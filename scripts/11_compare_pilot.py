from __future__ import annotations

import json
from pathlib import Path

from common import dump_json, load_config
from src.utils.checkpoint import utc_now


def main() -> None:
    config = load_config("configs/kaggle_pv_pilot_distill.yaml")
    output_dir = Path(config["runtime"]["output_dir"])
    reports_dir = output_dir / "reports"
    baseline = json.loads((reports_dir / "pilot_baseline_report.json").read_text(encoding="utf-8"))
    kd = json.loads((reports_dir / "pilot_kd_report.json").read_text(encoding="utf-8"))

    def metrics(report: dict, group: str) -> dict:
        return report["test"]["pixel_metrics"][group]

    comparison = {
        "created_at": utc_now(),
        "baseline": {
            "best_epoch": baseline["best_epoch"],
            "validation": baseline["validation"]["pixel_metrics"],
            "test": baseline["test"]["pixel_metrics"],
            "image_presence": baseline["test"]["image_presence_metrics"],
        },
        "kd": {
            "best_epoch": kd["best_epoch"],
            "validation": kd["validation"]["pixel_metrics"],
            "test": kd["test"]["pixel_metrics"],
            "image_presence": kd["test"]["image_presence_metrics"],
        },
        "delta": {
            group: {
                metric: metrics(kd, group)[metric] - metrics(baseline, group)[metric]
                for metric in ("dice", "iou", "precision", "recall", "specificity")
            }
            for group in ("overall", "regular_positive", "hard_positive", "hard_negative")
        },
    }
    dump_json(comparison, reports_dir / "pilot_comparison.json")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
