from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from common import dump_json, load_config, seed_everything
from src.data.dataset import config_hash, inspect_dataset
from src.teacher.feature_extractor import extract_feature
from src.teacher.load_internvl import gpu_memory, load_teacher
from src.teacher.semantic_extractor import extract_semantic
from src.utils.checkpoint import atomic_json_dump, load_json, utc_now
from src.utils.logger import setup_logger


def _complete(feature_path: Path, semantic_path: Path) -> bool:
    if not feature_path.exists() or feature_path.stat().st_size == 0 or not semantic_path.exists(): return False
    try:
        return isinstance(torch.load(feature_path, map_location="cpu", weights_only=True), torch.Tensor)
    except Exception:
        return False


def _error_type(exc: Exception) -> str:
    text = str(exc).lower()
    if "out of memory" in text or isinstance(exc, torch.cuda.OutOfMemoryError): return "OOM"
    if isinstance(exc, OSError): return "image_read_or_save_error"
    return "inference_error"


def run(config: dict, max_samples: int | None) -> None:
    output = Path(config["runtime"]["output_dir"]); cache = output / "teacher_cache"
    features_dir = cache / "features"; semantic_dir = cache / "semantic"; logs_dir = cache / "logs"
    features_dir.mkdir(parents=True, exist_ok=True); semantic_dir.mkdir(parents=True, exist_ok=True); logs_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(logs_dir / "extraction.log")
    _, samples, report = inspect_dataset(config)
    selected = samples if max_samples is None else samples[:max_samples]
    manifest_path = cache / "manifest.json"; failed_path = logs_dir / "failed_samples.json"
    failed = load_json(failed_path, [])
    complete_names = {sample.cache_key for sample in samples if _complete(features_dir / f"{sample.cache_key}.pt", semantic_dir / f"{sample.cache_key}.json")}
    failed = [item for item in failed if item.get("cache_key", Path(item.get("filename", "")).stem) not in complete_names]
    manifest = load_json(manifest_path, {"total": len(samples), "success": 0, "failed": 0, "skipped": 0, "remaining": len(samples), "updated_at": utc_now()})
    manifest.update({"total": len(samples), "success": len(complete_names), "failed": len({item.get("filename") for item in failed}),
                    "skipped": 0, "run_started_at": utc_now(), "config_hash": config_hash(config)})
    dump_json({"dataset_report": report, "config": config}, output / "reports" / "run_context.json")
    bundle = load_teacher(config, logger)
    times = []; run_success = run_failed = run_skipped = 0; started = time.perf_counter()
    for index, sample in enumerate(selected, 1):
        feature_path = features_dir / f"{sample.cache_key}.pt"; semantic_path = semantic_dir / f"{sample.cache_key}.json"
        if _complete(feature_path, semantic_path):
            run_skipped += 1
            manifest["skipped"] = manifest.get("skipped", 0) + 1
            manifest["updated_at"] = utc_now()
            atomic_json_dump(manifest, manifest_path)
            logger.info("[%d/%d] skip complete %s", index, len(selected), sample.filename)
            continue
        success = False
        for attempt in range(int(config["runtime"].get("retry_count", 1)) + 1):
            try:
                image_started = time.perf_counter()
                feature, feature_stats = extract_feature(bundle.model, sample.image_path, config)
                semantic, _ = extract_semantic(bundle, sample.image_path, config)
                torch.save(feature, feature_path)
                atomic_json_dump({"filename": sample.filename, "cache_key": sample.cache_key,
                                  "source": sample.source, "feature": feature_stats, "semantic": semantic,
                                  "saved_at": utc_now(), "config_hash": config_hash(config)}, semantic_path)
                manifest["feature_shape"] = feature_stats["shape"]
                times.append(time.perf_counter() - image_started); run_success += 1; success = True
                logger.info("[%d/%d] success %s %.2fs", index, len(selected), sample.filename, times[-1]); del feature
                failed = [item for item in failed if item.get("filename") != sample.filename]
                break
            except Exception as exc:
                if attempt < int(config["runtime"].get("retry_count", 1)):
                    logger.warning("Retry %s after %s: %s", sample.filename, _error_type(exc), exc)
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    continue
                failed.append({"filename": sample.filename, "cache_key": sample.cache_key,
                               "source": sample.source, "error_type": _error_type(exc),
                               "error_message": str(exc), "timestamp": utc_now()})
                run_failed += 1; logger.exception("Failed %s", sample.filename)
        manifest.update({"success": manifest.get("success", 0) + (1 if success else 0),
                         "failed": manifest.get("failed", 0) + (0 if success else 1)})
        completed = manifest["success"] + manifest["failed"]
        manifest["remaining"] = max(0, manifest["total"] - completed); manifest["updated_at"] = utc_now()
        atomic_json_dump(manifest, manifest_path); atomic_json_dump(failed, failed_path)
    manifest["run_elapsed_seconds"] = time.perf_counter() - started; manifest["average_time_per_image"] = sum(times) / len(times) if times else None
    manifest["estimated_remaining_seconds"] = manifest["remaining"] * manifest["average_time_per_image"] if times else None
    manifest["gpu"] = gpu_memory(); atomic_json_dump(manifest, manifest_path); atomic_json_dump(failed, failed_path)
    dump_json({"seed": config["seed"], "model": config["model"], "quantization": bundle.quantization, "dtype": str(bundle.dtype),
               "gpu": gpu_memory(), "successful_samples": manifest["success"], "failed_samples": manifest["failed"],
               "feature_shape": manifest.get("feature_shape"), "image_size": config["data"]["image_size"],
               "dynamic_patch_limit": config["model"].get("max_dynamic_patch"),
               "average_inference_time": manifest["average_time_per_image"], "prompt": config["teacher"]["prompt"]}, cache / "experiment_config.json")
    logger.info("Finished: success=%s failed=%s skipped=%s remaining=%s", run_success, run_failed, run_skipped, manifest["remaining"])


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/internvl3_5_teacher.yaml"); parser.add_argument("--max-samples", default=None)
    args = parser.parse_args(); config = load_config(args.config); seed_everything(int(config["seed"]))
    value = config["runtime"].get("max_samples", 20) if args.max_samples is None else args.max_samples
    max_samples = None if str(value).lower() == "none" else int(value); run(config, max_samples)


if __name__ == "__main__": main()
