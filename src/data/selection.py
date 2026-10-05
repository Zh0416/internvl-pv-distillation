from __future__ import annotations

import numpy as np

from .dataset import Sample, sample_to_dict


def _take_random(items: list[Sample], count: int, rng: np.random.Generator) -> list[Sample]:
    if count <= 0 or not items:
        return []
    indexes = np.arange(len(items))
    rng.shuffle(indexes)
    return [items[index] for index in indexes[:count]]


def _take_evenly(items: list[Sample], count: int) -> list[Sample]:
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    indexes = np.linspace(0, len(items) - 1, num=count, dtype=int)
    return [items[index] for index in indexes]


def select_distillation_subset(samples: list[Sample], config: dict) -> tuple[list[Sample], dict]:
    selection_cfg = config.get("selection", {})
    rng = np.random.default_rng(int(config["seed"]))

    synthetic_negatives = [sample for sample in samples if sample.label_origin.startswith("synthetic")]
    labeled_negatives = [sample for sample in samples if sample.mask_path is not None and not sample.is_positive]
    positives = sorted((sample for sample in samples if sample.is_positive), key=lambda sample: sample.pv_ratio)

    hard_positive_count = int(selection_cfg.get("hard_positive_count", 2))
    hard_positives = positives[:hard_positive_count]
    hard_keys = {sample.cache_key for sample in hard_positives}
    regular_candidates = [sample for sample in positives if sample.cache_key not in hard_keys]
    regular_positives = _take_evenly(
        regular_candidates,
        int(selection_cfg.get("regular_positive_count", 2)),
    )
    selected = (
        _take_random(synthetic_negatives, int(selection_cfg.get("negative_count", 2)), rng)
        + _take_random(labeled_negatives, int(selection_cfg.get("hard_negative_count", 1)), rng)
        + hard_positives
        + regular_positives
    )
    rng.shuffle(selected)
    manifest = {
        "requested": {
            "negative_count": int(selection_cfg.get("negative_count", 2)),
            "hard_negative_count": int(selection_cfg.get("hard_negative_count", 1)),
            "hard_positive_count": hard_positive_count,
            "regular_positive_count": int(selection_cfg.get("regular_positive_count", 2)),
        },
        "available": {
            "synthetic_negatives": len(synthetic_negatives),
            "labeled_negatives": len(labeled_negatives),
            "positives": len(positives),
        },
        "selected_count": len(selected),
        "selected": [sample_to_dict(sample) for sample in selected],
    }
    return selected, manifest
