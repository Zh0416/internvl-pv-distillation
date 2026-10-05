from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from PIL import Image

from common import dump_json, load_config, seed_everything
from src.data.confusers import select_challenging_negatives
from src.data.crops import crop_metadata_to_dict, fixed_negative_crop_boxes, largest_component, save_crop, square_crop_box
from src.data.dataset import inspect_dataset, sample_to_dict
from src.data.hard_samples import select_valid_hard_positives
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


def _run_view(
    bundle,
    teacher_config: dict,
    image_path: Path,
    view_key: str,
    view_type: str,
    crop_metadata: dict | None,
    feature_dir: Path,
    semantic_dir: Path,
    save_feature: bool,
) -> dict:
    view_config = copy.deepcopy(teacher_config)
    prompt_key = "full_prompt" if view_type == "full_image" else "crop_prompt"
    view_config["teacher"]["prompt"] = view_config["teacher"].get(
        prompt_key,
        view_config["teacher"]["prompt"],
    )
    feature_path = None
    feature_stats = None
    if save_feature:
        feature, feature_stats = extract_feature(bundle.model, image_path, view_config)
        feature_path = feature_dir / f"{view_key}.pt"
        torch.save(feature, feature_path)
        del feature
    semantic, _ = extract_semantic(bundle, image_path, view_config)
    view = {
        "view_key": view_key,
        "view_type": view_type,
        "image_path": str(image_path),
        "crop": crop_metadata,
        "feature": feature_stats,
        "feature_path": str(feature_path) if feature_path else None,
        "semantic": semantic,
        "pv_probability": _pv_probability(semantic),
        "prediction": bool(semantic["pv_exists"]),
    }
    atomic_json_dump(view, semantic_dir / f"{view_key}.json")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return view


def _metrics(results: list[dict], prediction_key: str) -> dict:
    return binary_classification_metrics(
        [result["ground_truth"] for result in results],
        [result[prediction_key] for result in results],
    )


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
    general_selected, general_selection = select_distillation_subset(samples, config)
    selected_groups = {
        row["cache_key"]: row["selection_group"]
        for row in general_selection["selected"]
    }
    regular_samples = [sample for sample in general_selected if selected_groups[sample.cache_key] == "regular_positive"]
    hard_samples, hard_selection = select_valid_hard_positives(samples, config)
    negative_samples, negative_selection = select_challenging_negatives(
        samples,
        int(config["selection"]["negative_count"]),
    )
    grouped_samples = {
        "regular_positive": regular_samples,
        "hard_positive": hard_samples,
        "hard_negative": negative_samples,
    }
    selected = [
        (group, sample)
        for group, group_samples in grouped_samples.items()
        for sample in group_samples
    ]
    expected_count = sum(
        int(config["selection"][key])
        for key in ("regular_positive_count", "hard_positive_count", "negative_count")
    )
    if len(selected) != expected_count:
        raise RuntimeError(f"测试样本数量错误: expected={expected_count}, actual={len(selected)}")

    logger = setup_logger(log_dir / "hard_crop_teacher.log", "hard_crop_teacher")
    logger.info("GPU before load: %s", gpu_memory())
    bundle = load_teacher(config, logger)
    teacher_config = copy.deepcopy(config)
    results = []
    failures = []
    started = time.perf_counter()

    for sample_index, (group, sample) in enumerate(selected, 1):
        try:
            full_view = _run_view(
                bundle,
                teacher_config,
                Path(sample.image_path),
                f"{sample.cache_key}__full",
                "full_image",
                None,
                feature_dir,
                semantic_dir,
                save_feature=group != "hard_positive",
            )
            views = [full_view]
            component = None
            primary_view = full_view
            if group == "hard_positive":
                component = largest_component(sample.mask_path)
                with Image.open(sample.image_path) as image:
                    image_size = image.size
                crop_box = square_crop_box(
                    image_size,
                    component.bbox,
                    int(config["hard_crop"]["min_crop_size"]),
                    float(config["hard_crop"]["context_scale"]),
                )
                crop_path = crop_dir / f"{sample.cache_key}__target.png"
                metadata = save_crop(
                    sample.image_path,
                    crop_box,
                    crop_path,
                    component.bbox,
                    int(config["hard_crop"]["teacher_input_size"]),
                )
                primary_view = _run_view(
                    bundle,
                    teacher_config,
                    crop_path,
                    f"{sample.cache_key}__target",
                    "mask_guided_target_crop",
                    crop_metadata_to_dict(metadata),
                    feature_dir,
                    semantic_dir,
                    save_feature=True,
                )
                views.append(primary_view)
            elif group == "hard_negative":
                with Image.open(sample.image_path) as image:
                    image_size = image.size
                crop_boxes = fixed_negative_crop_boxes(
                    image_size,
                    int(config["hard_crop"]["negative_crop_size"]),
                    int(config["hard_crop"]["negative_view_count"]),
                )
                for view_index, crop_box in enumerate(crop_boxes):
                    crop_path = crop_dir / f"{sample.cache_key}__negative{view_index}.png"
                    metadata = save_crop(
                        sample.image_path,
                        crop_box,
                        crop_path,
                        None,
                        int(config["hard_crop"]["teacher_input_size"]),
                    )
                    views.append(
                        _run_view(
                            bundle,
                            teacher_config,
                            crop_path,
                            f"{sample.cache_key}__negative{view_index}",
                            "fixed_negative_crop",
                            crop_metadata_to_dict(metadata),
                            feature_dir,
                            semantic_dir,
                            save_feature=False,
                        )
                    )

            result = {
                "sample": sample_to_dict(sample),
                "selection_group": group,
                "ground_truth": sample.is_positive,
                "full_prediction": full_view["prediction"],
                "primary_prediction": primary_view["prediction"],
                "full_pv_probability": full_view["pv_probability"],
                "primary_pv_probability": primary_view["pv_probability"],
                "full_correct": full_view["prediction"] == sample.is_positive,
                "primary_correct": primary_view["prediction"] == sample.is_positive,
                "component": {
                    "bbox": component.bbox,
                    "pixels": component.pixels,
                    "component_count": component.component_count,
                } if component else None,
                "views": views,
            }
            results.append(result)
            logger.info(
                "[%d/%d] %s group=%s full=%s primary=%s",
                sample_index,
                len(selected),
                sample.cache_key,
                group,
                full_view["prediction"],
                primary_view["prediction"],
            )
        except Exception as exc:
            failures.append(
                {
                    "cache_key": sample.cache_key,
                    "selection_group": group,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            logger.exception("Failed %s", sample.cache_key)

    if not results:
        raise RuntimeError("Teacher门槛测试没有成功样本")
    metrics_by_group = {}
    for group in grouped_samples:
        subset = [result for result in results if result["selection_group"] == group]
        if subset:
            metrics_by_group[group] = {
                "full_image": _metrics(subset, "full_prediction"),
                "primary_view": _metrics(subset, "primary_prediction"),
            }
    negative_crop_views = [
        view
        for result in results if result["selection_group"] == "hard_negative"
        for view in result["views"] if view["view_type"] == "fixed_negative_crop"
    ]
    negative_crop_metrics = binary_classification_metrics(
        [False] * len(negative_crop_views),
        [view["prediction"] for view in negative_crop_views],
    )
    report = {
        "created_at": utc_now(),
        "training_performed": False,
        "test_type": "50-sample InternVL teacher gate before knowledge distillation",
        "evaluation_scope": "full-image evaluation plus oracle mask-guided crops for hard-positive training only",
        "parameters": {
            "seed": config["seed"],
            "model": config["model"],
            "hard_crop": config["hard_crop"],
            "quantization_used": bundle.quantization,
            "dtype_used": str(bundle.dtype),
        },
        "dataset_summary": dataset_report,
        "selection": {
            "general": general_selection,
            "hard_positive": hard_selection,
            "hard_negative": negative_selection,
            "selected_by_group": {group: len(items) for group, items in grouped_samples.items()},
        },
        "metrics": {
            "full_image_50": _metrics(results, "full_prediction"),
            "primary_view_50": _metrics(results, "primary_prediction"),
            "by_group": metrics_by_group,
            "negative_crop_views": negative_crop_metrics,
        },
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
    print("Full-image metrics:", report["metrics"]["full_image_50"])
    print("Primary-view metrics:", report["metrics"]["primary_view_50"])
    print("Metrics by group:", metrics_by_group)
    print("Negative crop metrics:", negative_crop_metrics)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
