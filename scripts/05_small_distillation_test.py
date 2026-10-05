from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch

from common import dump_json, load_config, seed_everything
from src.data.dataset import config_hash, inspect_dataset, sample_to_dict
from src.data.selection import select_distillation_subset
from src.teacher.feature_extractor import extract_feature
from src.teacher.load_internvl import gpu_memory, load_teacher
from src.teacher.semantic_extractor import extract_semantic
from src.utils.checkpoint import atomic_json_dump, utc_now
from src.utils.logger import setup_logger
from src.utils.metrics import binary_classification_metrics


def _run_sample(sample, bundle, config: dict, features_dir: Path, semantic_dir: Path) -> dict:
    started = time.perf_counter()
    feature, feature_stats = extract_feature(bundle.model, sample.image_path, config)
    semantic, _ = extract_semantic(bundle, sample.image_path, config)
    feature_path = features_dir / f"{sample.cache_key}.pt"
    semantic_path = semantic_dir / f"{sample.cache_key}.json"
    torch.save(feature, feature_path)
    result = {
        "sample": sample_to_dict(sample),
        "ground_truth": sample.is_positive,
        "prediction": bool(semantic["pv_exists"]),
        "correct": bool(semantic["pv_exists"]) == sample.is_positive,
        "feature": feature_stats,
        "semantic": semantic,
        "feature_path": str(feature_path),
        "semantic_path": str(semantic_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json_dump(result, semantic_path)
    del feature
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kaggle_pv_small_distill.yaml")
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument(
        "--source-root",
        action="append",
        default=[],
        metavar="NAME=ABSOLUTE_PATH",
        help="覆盖指定数据源根目录，可重复使用",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    source_overrides = {}
    for value in args.source_root:
        if "=" not in value:
            raise ValueError(f"--source-root格式必须是NAME=ABSOLUTE_PATH，实际为: {value}")
        name, root = value.split("=", 1)
        root_path = Path(root)
        if not root_path.is_absolute():
            raise ValueError(f"数据源路径必须是绝对路径: {root}")
        source_overrides[name] = str(root_path)
    configured_sources = {source["name"]: source for source in config["data"].get("sources", [])}
    unknown_sources = sorted(set(source_overrides) - set(configured_sources))
    if unknown_sources:
        raise ValueError(f"未知数据源: {unknown_sources}")
    for name, root in source_overrides.items():
        configured_sources[name]["root_candidates"] = [root]
    if args.output_dir:
        output_path = Path(args.output_dir)
        if not output_path.is_absolute():
            raise ValueError(f"输出路径必须是绝对路径: {args.output_dir}")
        config["runtime"]["output_dir"] = str(output_path)
    seed_everything(int(config["seed"]))
    output_dir = Path(config["runtime"]["output_dir"])
    reports_dir = output_dir / "reports"
    cache_dir = output_dir / "teacher_cache"
    features_dir = cache_dir / "features"
    semantic_dir = cache_dir / "semantic"
    logs_dir = cache_dir / "logs"
    for directory in (reports_dir, features_dir, semantic_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(logs_dir / "small_distillation.log", "small_distillation")
    _, samples, dataset_report = inspect_dataset(config)
    selected, selection = select_distillation_subset(samples, config)
    if not selected:
        raise RuntimeError("没有可用于小样本蒸馏测试的数据")
    dump_json(dataset_report, reports_dir / "dataset_report.json")
    dump_json(selection, reports_dir / "selection_manifest.json")
    print(
        "Dataset:",
        {
            "images": dataset_report["total_images"],
            "positive": dataset_report["positive_count"],
            "negative": dataset_report["negative_count"],
            "selected": selection["selected_count"],
        },
    )
    if args.selection_only:
        print(f"Selection manifest: {reports_dir / 'selection_manifest.json'}")
        return

    logger.info("GPU before load: %s", gpu_memory())
    bundle = load_teacher(config, logger)
    results: list[dict] = []
    failures: list[dict] = []
    started = time.perf_counter()
    for index, sample in enumerate(selected, 1):
        try:
            result = _run_sample(sample, bundle, config, features_dir, semantic_dir)
            results.append(result)
            logger.info(
                "[%d/%d] %s truth=%s prediction=%s confidence=%.3f",
                index,
                len(selected),
                sample.cache_key,
                sample.is_positive,
                result["prediction"],
                float(result["semantic"]["confidence"]),
            )
        except Exception as exc:
            failures.append(
                {
                    "cache_key": sample.cache_key,
                    "filename": sample.filename,
                    "source": sample.source,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "timestamp": utc_now(),
                }
            )
            logger.exception("Failed %s", sample.cache_key)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not results:
        dump_json({"failures": failures}, reports_dir / "small_distillation_report.json")
        raise RuntimeError("所有小样本测试均失败，请查看small_distillation.log")

    metrics = binary_classification_metrics(
        [result["ground_truth"] for result in results],
        [result["prediction"] for result in results],
    )
    single_test = results[0]
    report = {
        "created_at": utc_now(),
        "config_hash": config_hash(config),
        "training_performed": False,
        "test_type": "teacher cache extraction and image-level semantic validation for distillation",
        "parameters": {
            "seed": config["seed"],
            "model": config["model"],
            "teacher": config["teacher"],
            "selection": config["selection"],
            "quantization_used": bundle.quantization,
            "dtype_used": str(bundle.dtype),
        },
        "dataset_summary": {
            "total_images": dataset_report["total_images"],
            "positive_count": dataset_report["positive_count"],
            "negative_count": dataset_report["negative_count"],
            "sources": dataset_report["sources"],
        },
        "selection": selection,
        "metrics": metrics,
        "single_test": single_test,
        "samples": results,
        "failures": failures,
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "gpu": gpu_memory(),
            "success_count": len(results),
            "failure_count": len(failures),
        },
    }
    report_path = reports_dir / "small_distillation_report.json"
    dump_json(report, report_path)
    print("Metrics:", metrics)
    print(
        "Single test:",
        {
            "cache_key": single_test["sample"]["cache_key"],
            "truth": single_test["ground_truth"],
            "prediction": single_test["prediction"],
            "confidence": single_test["semantic"]["confidence"],
            "correct": single_test["correct"],
        },
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
