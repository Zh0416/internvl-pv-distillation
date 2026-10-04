from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from common import dump_json, load_config, seed_everything
from src.data.dataset import inspect_dataset
from src.teacher.feature_extractor import extract_feature
from src.teacher.load_internvl import gpu_memory, load_teacher
from src.teacher.semantic_extractor import extract_semantic
from src.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/internvl3_5_teacher.yaml")
    args = parser.parse_args(); config = load_config(args.config); seed_everything(int(config["seed"]))
    cache = Path(config["runtime"]["output_dir"]) / "teacher_cache"; logger = setup_logger(cache / "logs" / "extraction.log", "teacher_test")
    _, samples, _ = inspect_dataset(config)
    if not samples:
        raise RuntimeError("没有可测试的有效影像")
    logger.info("GPU before load: %s", gpu_memory()); bundle = load_teacher(config, logger)
    started = time.perf_counter(); feature, feature_stats = extract_feature(bundle.model, samples[0].image_path, config)
    semantic, semantic_time = extract_semantic(bundle, samples[0].image_path, config)
    result = {"filename": samples[0].filename, "feature": feature_stats, "semantic": semantic,
              "model": config["model"], "quantization": bundle.quantization, "gpu": gpu_memory(),
              "total_elapsed_seconds": time.perf_counter() - started}
    dump_json(result, cache / "single_test.json")
    logger.info("Single test passed: %s", result)
    print(f"Feature shape: {feature_stats['shape']}; semantic: {semantic}; GPU: {result['gpu']}")
    print(f"Saved: {cache / 'single_test.json'}")
    del feature
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
