from __future__ import annotations

import argparse
from pathlib import Path

from common import dump_json, load_config, seed_everything
from src.data.dataset import inspect_dataset, sample_to_dict
from src.data.split_dataset import create_splits, save_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/internvl3_5_teacher.yaml")
    args = parser.parse_args()
    config = load_config(args.config); seed_everything(int(config["seed"]))
    output = Path(config["runtime"]["output_dir"])
    root, samples, report = inspect_dataset(config)
    report["samples"] = [sample_to_dict(sample) for sample in samples]
    dump_json(report, output / "reports" / "dataset_report.json")
    splits = create_splits(samples, config); save_splits(splits, output / "splits")
    print(f"Dataset root: {root}")
    print(f"Images={report['total_images']} valid_masks={report['valid_mask_count']} positive={report['positive_count']} negative={report['negative_count']}")
    print(f"Splits: { {name: len(items) for name, items in splits.items()} }")
    print(f"Report: {output / 'reports' / 'dataset_report.json'}")


if __name__ == "__main__":
    main()
