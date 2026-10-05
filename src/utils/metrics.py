from __future__ import annotations

from collections.abc import Iterable


def _safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_classification_metrics(y_true: Iterable[bool], y_pred: Iterable[bool]) -> dict:
    truth = [bool(value) for value in y_true]
    prediction = [bool(value) for value in y_pred]
    if len(truth) != len(prediction):
        raise ValueError("y_true与y_pred长度必须一致")
    if not truth:
        raise ValueError("至少需要一个预测样本")

    true_positive = sum(expected and actual for expected, actual in zip(truth, prediction))
    true_negative = sum(not expected and not actual for expected, actual in zip(truth, prediction))
    false_positive = sum(not expected and actual for expected, actual in zip(truth, prediction))
    false_negative = sum(expected and not actual for expected, actual in zip(truth, prediction))
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {
        "sample_count": len(truth),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy": _safe_divide(true_positive + true_negative, len(truth)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": f1,
        "iou": _safe_divide(true_positive, true_positive + false_positive + false_negative),
        "specificity": _safe_divide(true_negative, true_negative + false_positive),
        "metric_scope": "image-level PV presence classification; not pixel-level segmentation",
    }
