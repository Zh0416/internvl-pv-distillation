from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .confusers import score_pv_confuser
from .dataset import Sample, sample_to_dict
from .hard_samples import mask_geometry, target_component_geometry
from .selection import select_regular_positives


def _hard_positive_candidates(samples: list[Sample], config: dict) -> list[Sample]:
    hard_cfg = config["hard_crop"]
    min_pixels = int(hard_cfg.get("min_mask_pixels", 128))
    exclude_border = bool(hard_cfg.get("exclude_border_touching", True))
    min_border_margin = int(hard_cfg.get("min_border_margin", 0))
    candidates = []
    for sample in sorted((item for item in samples if item.is_positive), key=lambda item: item.pv_pixels):
        geometry = mask_geometry(sample.mask_path)
        component = target_component_geometry(sample.mask_path)
        if geometry.pixels < min_pixels:
            continue
        if exclude_border and component["touches_border"]:
            continue
        if component["border_distance"] < min_border_margin:
            continue
        candidates.append(sample)
    return candidates


def _ranked_negatives(samples: list[Sample]) -> list[Sample]:
    return [
        sample
        for sample, _ in sorted(
            (
                (sample, score_pv_confuser(sample.image_path))
                for sample in samples
                if not sample.is_positive
            ),
            key=lambda item: (-item[1].score, item[0].cache_key),
        )
    ]


def _take_evenly(items: list[Sample], count: int) -> list[Sample]:
    if len(items) < count:
        raise RuntimeError(f"候选样本不足: requested={count}, found={len(items)}")
    indexes = np.linspace(0, len(items) - 1, num=count + 2, dtype=int)[1:-1]
    return [items[index] for index in indexes]


def build_pilot_splits(samples: list[Sample], config: dict) -> tuple[dict[str, list[Sample]], dict]:
    counts = config["pilot"]["counts"]
    sample_by_key = {sample.cache_key: sample for sample in samples}
    fixed_test = config["pilot"].get("fixed_test", {})

    def fixed_group(group: str) -> list[Sample]:
        keys = list(fixed_test.get(group, []))
        missing = [key for key in keys if key not in sample_by_key]
        if missing:
            raise RuntimeError(f"固定测试样本不存在: {missing}")
        expected = int(counts["test"][group])
        if len(keys) != expected:
            raise RuntimeError(f"固定测试组数量错误: group={group}, expected={expected}, actual={len(keys)}")
        return [sample_by_key[key] for key in keys]

    hard_candidates = _hard_positive_candidates(samples, config)
    negative_candidates = _ranked_negatives(samples)

    hard_train_count = int(counts["train"]["hard_positive"])
    hard_val_count = int(counts["validation"]["hard_positive"])
    hard_test = fixed_group("hard_positive")
    regular_test = fixed_group("regular_positive")
    negative_test = fixed_group("hard_negative")
    fixed_test_keys = {
        sample.cache_key for sample in [*hard_test, *regular_test, *negative_test]
    }
    hard_remaining = [
        sample for sample in hard_candidates
        if sample.cache_key not in fixed_test_keys
    ]
    hard_remaining_count = hard_train_count + hard_val_count
    if len(hard_remaining) < hard_remaining_count:
        raise RuntimeError(
            f"困难正样本不足: requested={hard_remaining_count}, found={len(hard_remaining)}"
        )
    hard_train = hard_remaining[:hard_train_count]
    hard_validation = hard_remaining[hard_train_count:hard_remaining_count]
    selected_hard = [*hard_test, *hard_train, *hard_validation]
    hard_threshold = max(sample.pv_ratio for sample in selected_hard)
    reserved_positive_keys = {
        sample.cache_key
        for sample in [*selected_hard, *regular_test]
    }
    regular_train_count = int(counts["train"]["regular_positive"])
    regular_validation_count = int(counts["validation"]["regular_positive"])
    regular_pool, _ = select_regular_positives(
        samples,
        regular_train_count + regular_validation_count,
        reserved_positive_keys,
        hard_threshold,
    )
    rng = np.random.default_rng(int(config["seed"]))
    rng.shuffle(regular_pool)
    regular_train = regular_pool[:regular_train_count]
    regular_validation = regular_pool[regular_train_count:]

    negative_test_count = int(counts["test"]["hard_negative"])
    negative_train_count = int(counts["train"]["hard_negative"])
    negative_validation_count = int(counts["validation"]["hard_negative"])
    if len(negative_test) != negative_test_count:
        raise RuntimeError("固定负样本测试组数量错误")
    negative_remaining = [
        sample for sample in negative_candidates
        if sample.cache_key not in fixed_test_keys
    ]
    negative_remaining_count = negative_train_count + negative_validation_count
    if len(negative_remaining) < negative_remaining_count:
        raise RuntimeError(
            f"困难负样本不足: requested={negative_remaining_count}, found={len(negative_remaining)}"
        )
    negative_train = negative_remaining[:negative_train_count]
    negative_validation = negative_remaining[
        negative_train_count:negative_remaining_count
    ]

    grouped = {
        "train": {
            "regular_positive": regular_train,
            "hard_positive": hard_train,
            "hard_negative": negative_train,
        },
        "validation": {
            "regular_positive": regular_validation,
            "hard_positive": hard_validation,
            "hard_negative": negative_validation,
        },
        "test": {
            "regular_positive": regular_test,
            "hard_positive": hard_test,
            "hard_negative": negative_test,
        },
    }
    splits = {
        split: [sample for group_samples in groups.values() for sample in group_samples]
        for split, groups in grouped.items()
    }
    all_keys = [sample.cache_key for split_samples in splits.values() for sample in split_samples]
    if len(all_keys) != len(set(all_keys)):
        raise RuntimeError("Pilot train/validation/test 样本存在重叠")

    manifest = {
        "seed": int(config["seed"]),
        "counts": {
            split: {group: len(items) for group, items in groups.items()}
            for split, groups in grouped.items()
        },
        "hard_positive_ratio_ceiling": hard_threshold,
        "splits": {
            split: [
                {
                    **sample_to_dict(sample),
                    "selection_group": group,
                    "sample_weight": float(config["pilot"]["sample_weights"][group]),
                }
                for group, items in grouped[split].items()
                for sample in items
            ]
            for split in grouped
        },
        "hard_positive_audit": [
            {
                "cache_key": sample.cache_key,
                "geometry": asdict(mask_geometry(sample.mask_path)),
                "target_component": target_component_geometry(sample.mask_path),
            }
            for sample in selected_hard
        ],
    }
    return splits, manifest
