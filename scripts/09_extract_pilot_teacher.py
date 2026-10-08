from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import torch
from PIL import Image

from common import dump_json, load_config, seed_everything
from src.data.crops import crop_metadata_to_dict, largest_component, save_crop, square_crop_box
from src.teacher.feature_extractor import extract_feature
from src.teacher.load_internvl import gpu_memory, load_teacher
from src.teacher.semantic_extractor import extract_semantic
from src.utils.checkpoint import atomic_json_dump, utc_now
from src.utils.logger import setup_logger
from src.utils.metrics import binary_classification_metrics


def _teacher_view(row: dict, config: dict, crop_dir: Path) -> tuple[Path, str, dict | None]:
    image_path = Path(row["image_path"])
    if row["selection_group"] != "hard_positive":
        return image_path, "full_image", None
    component = largest_component(row["mask_path"])
    with Image.open(image_path) as image:
        crop_box = square_crop_box(
            image.size,
            component.bbox,
            int(config["hard_crop"]["min_crop_size"]),
            float(config["hard_crop"]["context_scale"]),
        )
    crop_path = crop_dir / f"{row['cache_key']}__target.png"
    metadata = save_crop(
        image_path,
        crop_box,
        crop_path,
        component.bbox,
        int(config["hard_crop"]["teacher_input_size"]),
    )
    return crop_path, "mask_guided_target_crop", crop_metadata_to_dict(metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kaggle_pv_pilot_distill.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_dir:
        override_dir = Path(args.output_dir)
        if not override_dir.is_absolute():
            raise ValueError(f"输出路径必须是绝对路径: {override_dir}")
        config["runtime"]["output_dir"] = str(override_dir)
    seed_everything(int(config["seed"]))
    output_dir = Path(config["runtime"]["output_dir"])
    manifest_path = output_dir / "reports" / "pilot_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"未找到Pilot清单: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_rows = manifest["splits"]["train"]
    cache_dir = output_dir / "pilot_teacher_cache"
    feature_dir = cache_dir / "features"
    semantic_dir = cache_dir / "semantic"
    crop_dir = cache_dir / "crops"
    log_dir = cache_dir / "logs"
    for directory in (feature_dir, semantic_dir, crop_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir / "pilot_teacher.log", "pilot_teacher")
    logger.info("GPU before load: %s", gpu_memory())
    bundle = load_teacher(config, logger)
    results = []
    failures = []
    started = time.perf_counter()
    for index, row in enumerate(train_rows, 1):
        feature_path = feature_dir / f"{row['cache_key']}.pt"
        semantic_path = semantic_dir / f"{row['cache_key']}.json"
        if feature_path.is_file() and semantic_path.is_file():
            results.append(json.loads(semantic_path.read_text(encoding="utf-8")))
            logger.info("[%d/%d] resume %s", index, len(train_rows), row["cache_key"])
            continue
        try:
            view_path, view_type, crop = _teacher_view(row, config, crop_dir)
            view_config = copy.deepcopy(config)
            prompt_key = "crop_prompt" if view_type != "full_image" else "full_prompt"
            view_config["teacher"]["prompt"] = view_config["teacher"][prompt_key]
            feature, feature_stats = extract_feature(bundle.model, view_path, view_config)
            pooled_feature = feature.float().mean(dim=tuple(range(feature.ndim - 1))).to(torch.float16)
            torch.save(pooled_feature, feature_path)
            semantic, _ = extract_semantic(bundle, view_path, view_config)
            ground_truth = bool(row["is_positive"])
            prediction = bool(semantic["pv_exists"])
            semantic_weight = 1.0 if prediction == ground_truth else 0.0
            result = {
                "cache_key": row["cache_key"],
                "filename": row["filename"],
                "selection_group": row["selection_group"],
                "ground_truth": ground_truth,
                "view_type": view_type,
                "view_path": str(view_path),
                "crop": crop,
                "feature_path": str(feature_path),
                "feature_dim": int(pooled_feature.numel()),
                "feature_stats": feature_stats,
                "semantic": semantic,
                "semantic_weight": semantic_weight,
                "correct": prediction == ground_truth,
            }
            atomic_json_dump(result, semantic_path)
            results.append(result)
            logger.info(
                "[%d/%d] %s group=%s prediction=%s probability=%.3f semantic_weight=%.1f",
                index,
                len(train_rows),
                row["cache_key"],
                row["selection_group"],
                prediction,
                float(semantic["pv_probability"]),
                semantic_weight,
            )
            del feature, pooled_feature
        except Exception as exc:
            failures.append(
                {
                    "cache_key": row["cache_key"],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            logger.exception("Failed %s", row["cache_key"])
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    metrics = binary_classification_metrics(
        [result["ground_truth"] for result in results],
        [result["semantic"]["pv_exists"] for result in results],
    )
    report = {
        "created_at": utc_now(),
        "sample_count": len(train_rows),
        "success_count": len(results),
        "failure_count": len(failures),
        "metrics": metrics,
        "semantic_disabled_count": sum(result["semantic_weight"] == 0 for result in results),
        "feature_dim": results[0]["feature_dim"] if results else None,
        "quantization": bundle.quantization,
        "runtime_seconds": time.perf_counter() - started,
        "gpu": gpu_memory(),
        "failures": failures,
    }
    dump_json(report, output_dir / "reports" / "pilot_teacher_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise RuntimeError(f"Teacher缓存失败 {len(failures)} 张，请重跑以断点续传")


if __name__ == "__main__":
    main()
