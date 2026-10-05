from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from PIL import Image

from common import dump_json, load_config, seed_everything
from src.data.crops import (
    crop_metadata_to_dict,
    fixed_negative_crop_boxes,
    largest_component,
    save_crop,
    square_crop_box,
)
from src.data.dataset import inspect_dataset, sample_to_dict
from src.data.selection import select_distillation_subset
from src.teacher.feature_extractor import extract_feature
from src.teacher.load_internvl import gpu_memory, load_teacher
from src.teacher.semantic_extractor import extract_semantic
from src.utils.checkpoint import atomic_json_dump, utc_now
from src.utils.logger import setup_logger
from src.utils.metrics import binary_classification_metrics


def _apply_overrides(config: dict, values: list[str], output_dir: str | None) -> None:
    sources = {source["name"]: source for source in config["data"]["sources"]}
    for value in values:
        name, root = value.split("=", 1)
        root_path = Path(root)
        if not root_path.is_absolute():
            raise ValueError(f"数据源路径必须是绝对路径: {root}")
        sources[name]["root_candidates"] = [str(root_path)]
    if output_dir:
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            raise ValueError(f"输出路径必须是绝对路径: {output_dir}")
        config["runtime"]["output_dir"] = str(output_path)


def _pv_probability(semantic: dict) -> float:
    confidence = float(semantic["confidence"])
    return confidence if semantic["pv_exists"] else 1.0 - confidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kaggle_pv_hard_crop_distill.yaml")
    parser.add_argument("--source-root", action="append", default=[])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    _apply_overrides(config, args.source_root, args.output_dir)
    seed_everything(int(config["seed"]))

    output_dir = Path(config["runtime"]["output_dir"])
    crop_dir = output_dir / "hard_crop_cache" / "crops"
    feature_dir = output_dir / "hard_crop_cache" / "features"
    semantic_dir = output_dir / "hard_crop_cache" / "semantic"
    log_dir = output_dir / "hard_crop_cache" / "logs"
    for directory in (crop_dir, feature_dir, semantic_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    _, samples, dataset_report = inspect_dataset(config)
    selected, selection = select_distillation_subset(samples, config)
    selected = [sample for sample in selected if sample.label_origin.startswith("synthetic") or sample.is_positive]
    logger = setup_logger(log_dir / "hard_crop_teacher.log", "hard_crop_teacher")
    logger.info("GPU before load: %s", gpu_memory())
    bundle = load_teacher(config, logger)
    teacher_config = copy.deepcopy(config)
    results = []
    failures = []
    started = time.perf_counter()

    for sample_index, sample in enumerate(selected, 1):
        try:
            with Image.open(sample.image_path) as image:
                image_size = image.size
            target_bbox = None
            component = None
            if sample.is_positive:
                component = largest_component(sample.mask_path)
                target_bbox = component.bbox
                crop_boxes = [
                    square_crop_box(
                        image_size,
                        target_bbox,
                        int(config["hard_crop"]["min_crop_size"]),
                        float(config["hard_crop"]["context_scale"]),
                    )
                ]
            else:
                crop_boxes = fixed_negative_crop_boxes(
                    image_size,
                    int(config["hard_crop"]["min_crop_size"]),
                    int(config["hard_crop"]["negative_view_count"]),
                )

            views = []
            for view_index, crop_box in enumerate(crop_boxes):
                view_key = f"{sample.cache_key}__view{view_index}"
                crop_path = crop_dir / f"{view_key}.png"
                metadata = save_crop(
                    sample.image_path,
                    crop_box,
                    crop_path,
                    target_bbox,
                    int(config["hard_crop"]["teacher_input_size"]),
                )
                feature, feature_stats = extract_feature(bundle.model, crop_path, teacher_config)
                semantic, _ = extract_semantic(bundle, crop_path, teacher_config)
                feature_path = feature_dir / f"{view_key}.pt"
                semantic_path = semantic_dir / f"{view_key}.json"
                torch.save(feature, feature_path)
                view = {
                    "view_key": view_key,
                    "crop_path": str(crop_path),
                    "crop": crop_metadata_to_dict(metadata),
                    "feature": feature_stats,
                    "feature_path": str(feature_path),
                    "semantic": semantic,
                    "pv_probability": _pv_probability(semantic),
                }
                atomic_json_dump(view, semantic_path)
                views.append(view)
                del feature
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            probability = max(view["pv_probability"] for view in views)
            prediction = probability >= 0.5
            result = {
                "sample": sample_to_dict(sample),
                "ground_truth": sample.is_positive,
                "prediction": prediction,
                "pv_probability": probability,
                "correct": prediction == sample.is_positive,
                "component": {
                    "bbox": component.bbox,
                    "pixels": component.pixels,
                    "component_count": component.component_count,
                } if component else None,
                "views": views,
            }
            results.append(result)
            logger.info(
                "[%d/%d] %s truth=%s prediction=%s probability=%.3f views=%d",
                sample_index,
                len(selected),
                sample.cache_key,
                sample.is_positive,
                prediction,
                probability,
                len(views),
            )
        except Exception as exc:
            failures.append(
                {
                    "cache_key": sample.cache_key,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            logger.exception("Failed %s", sample.cache_key)

    if not results:
        raise RuntimeError("局部Teacher测试没有成功样本")
    metrics = binary_classification_metrics(
        [result["ground_truth"] for result in results],
        [result["prediction"] for result in results],
    )
    metrics_by_label = {}
    for label, expected in (("hard_positive", True), ("negative", False)):
        subset = [result for result in results if result["ground_truth"] is expected]
        if subset:
            metrics_by_label[label] = binary_classification_metrics(
                [result["ground_truth"] for result in subset],
                [result["prediction"] for result in subset],
            )
    report = {
        "created_at": utc_now(),
        "evaluation_scope": "oracle mask-guided crops for distillation training; not deployable inference",
        "parameters": {
            "seed": config["seed"],
            "model": config["model"],
            "hard_crop": config["hard_crop"],
            "quantization_used": bundle.quantization,
            "dtype_used": str(bundle.dtype),
        },
        "dataset_summary": dataset_report,
        "selection": selection,
        "metrics": metrics,
        "metrics_by_label": metrics_by_label,
        "results": results,
        "failures": failures,
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "success_count": len(results),
            "failure_count": len(failures),
            "gpu": gpu_memory(),
        },
    }
    report_path = output_dir / "reports" / "hard_crop_teacher_report.json"
    dump_json(report, report_path)
    print("Metrics:", metrics)
    print("Metrics by label:", metrics_by_label)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
