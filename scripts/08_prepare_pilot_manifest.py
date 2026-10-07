from __future__ import annotations

import argparse
from pathlib import Path

from common import dump_json, load_config, seed_everything
from src.data.dataset import inspect_dataset
from src.data.pilot import build_pilot_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kaggle_pv_pilot_distill.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            raise ValueError(f"输出路径必须是绝对路径: {output_dir}")
        config["runtime"]["output_dir"] = str(output_dir)
    seed_everything(int(config["seed"]))
    output_dir = Path(config["runtime"]["output_dir"])
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _, samples, dataset_report = inspect_dataset(config)
    _, manifest = build_pilot_splits(samples, config)
    dump_json(dataset_report, reports_dir / "pilot_dataset_report.json")
    dump_json(manifest, reports_dir / "pilot_manifest.json")
    print("Pilot counts:", manifest["counts"])
    print("Unique samples:", sum(sum(groups.values()) for groups in manifest["counts"].values()))
    print("Manifest:", reports_dir / "pilot_manifest.json")


if __name__ == "__main__":
    main()
