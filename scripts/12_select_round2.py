from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import dump_json
from src.utils.checkpoint import utc_now


BASELINE_RUNS = ("reference", "hard_weight", "positive_weight")
KD_RUNS = ("semantic_only", "feature_only", "combined_low")


def _load_report(reports_dir: Path, run_name: str) -> dict:
    return json.loads((reports_dir / f"pilot_{run_name}_report.json").read_text(encoding="utf-8"))


def _metrics(evaluation: dict, group: str) -> dict:
    return evaluation["pixel_metrics"][group]


def select_threshold(report: dict) -> dict:
    evaluations = report["validation_thresholds"]
    reference = evaluations.get("0.5")
    if reference is None:
        raise ValueError("阈值扫描缺少0.5基准")
    minimum_overall = _metrics(reference, "overall")["dice"] - 0.01
    candidates = [
        (float(threshold), evaluation)
        for threshold, evaluation in evaluations.items()
        if _metrics(evaluation, "hard_negative")["false_positive"] == 0
        and _metrics(evaluation, "overall")["dice"] >= minimum_overall
    ]
    if not candidates:
        raise RuntimeError(f"没有满足验证集约束的阈值: {report['run_name']}")
    threshold, evaluation = max(
        candidates,
        key=lambda item: (
            _metrics(item[1], "hard_positive")["dice"],
            _metrics(item[1], "hard_positive")["recall"],
            _metrics(item[1], "overall")["dice"],
            -abs(item[0] - 0.5),
        ),
    )
    return {
        "run_name": report["run_name"],
        "threshold": threshold,
        "parameters": report["parameters"],
        "validation": evaluation["pixel_metrics"],
    }


def _eligible(candidate: dict, baseline: dict) -> bool:
    candidate_metrics = candidate["validation"]
    baseline_metrics = baseline["validation"]
    return (
        candidate_metrics["hard_negative"]["false_positive"] == 0
        and candidate_metrics["overall"]["dice"] >= baseline_metrics["overall"]["dice"] - 0.01
        and candidate_metrics["hard_positive"]["recall"] >= baseline_metrics["hard_positive"]["recall"] - 0.005
    )


def _rank(candidate: dict) -> tuple[float, float, float]:
    metrics = candidate["validation"]
    return (
        metrics["hard_positive"]["dice"],
        metrics["hard_positive"]["recall"],
        metrics["overall"]["dice"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", required=True, choices=("baseline", "kd", "finalize"))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        raise ValueError(f"输出路径必须是绝对路径: {output_dir}")
    reports_dir = output_dir / "reports"
    selection_path = reports_dir / "round2_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {"created_at": utc_now()}

    if args.stage == "baseline":
        candidates = {name: select_threshold(_load_report(reports_dir, name)) for name in BASELINE_RUNS}
        reference = candidates["reference"]
        eligible = [candidate for candidate in candidates.values() if _eligible(candidate, reference)]
        selection["baseline_candidates"] = candidates
        selection["baseline"] = max(eligible, key=_rank) if eligible else reference
    elif args.stage == "kd":
        if "baseline" not in selection:
            raise RuntimeError("必须先选择Baseline")
        candidates = {name: select_threshold(_load_report(reports_dir, name)) for name in KD_RUNS}
        eligible = [candidate for candidate in candidates.values() if _eligible(candidate, selection["baseline"])]
        selection["kd_candidates"] = candidates
        selection["kd"] = max(eligible or candidates.values(), key=_rank)
        selection["kd_validation_guard_passed"] = bool(eligible)
    else:
        baseline = _load_report(reports_dir, selection["baseline"]["run_name"])
        kd = _load_report(reports_dir, selection["kd"]["run_name"])
        if not baseline.get("test") or not kd.get("test"):
            raise RuntimeError("最终模型尚未在固定测试集上评估")
        baseline_metrics = baseline["test"]["pixel_metrics"]
        kd_metrics = kd["test"]["pixel_metrics"]
        selection["test_comparison"] = {
            "baseline": baseline_metrics,
            "kd": kd_metrics,
            "delta": {
                group: {
                    metric: kd_metrics[group][metric] - baseline_metrics[group][metric]
                    for metric in ("dice", "iou", "precision", "recall", "specificity")
                }
                for group in ("overall", "regular_positive", "hard_positive", "hard_negative")
            },
            "accepted": (
                selection["kd_validation_guard_passed"]
                and kd_metrics["hard_positive"]["dice"] - baseline_metrics["hard_positive"]["dice"] >= 0.02
                and kd_metrics["hard_positive"]["iou"] - baseline_metrics["hard_positive"]["iou"] >= 0.02
                and kd_metrics["hard_positive"]["recall"] >= baseline_metrics["hard_positive"]["recall"]
                and kd_metrics["hard_negative"]["false_positive"] == 0
            ),
        }
    dump_json(selection, selection_path)
    print(json.dumps({"stage": args.stage, "baseline": selection.get("baseline", {}).get("run_name"), "kd": selection.get("kd", {}).get("run_name"), "accepted": selection.get("test_comparison", {}).get("accepted"), "report": str(selection_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
