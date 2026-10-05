from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import Sample


def _partition(indexes: list[int], config: dict, rng: np.random.Generator) -> dict[str, list[int]]:
    shuffled = np.asarray(indexes, dtype=int)
    rng.shuffle(shuffled)
    train_end = int(len(shuffled) * float(config["split"]["train"]))
    val_end = train_end + int(len(shuffled) * float(config["split"]["val"]))
    return {
        "train": shuffled[:train_end].tolist(),
        "val": shuffled[train_end:val_end].tolist(),
        "test": shuffled[val_end:].tolist(),
    }


def create_splits(samples: list[Sample], config: dict) -> dict[str, list[Sample]]:
    rng = np.random.default_rng(int(config["seed"]))
    if config["split"].get("stratify", True):
        strata = [
            [index for index, sample in enumerate(samples) if sample.is_positive],
            [index for index, sample in enumerate(samples) if not sample.is_positive],
        ]
        split_indexes = {"train": [], "val": [], "test": []}
        for stratum in strata:
            partitioned = _partition(stratum, config, rng)
            for name in split_indexes:
                split_indexes[name].extend(partitioned[name])
    else:
        split_indexes = _partition(list(range(len(samples))), config, rng)

    result: dict[str, list[Sample]] = {}
    for name, indexes in split_indexes.items():
        rng.shuffle(indexes)
        result[name] = [samples[index] for index in indexes]
    return result


def save_splits(splits: dict[str, list[Sample]], output_dir: str | Path) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, samples in splits.items():
        (directory / f"{name}.txt").write_text(
            "\n".join(sample.cache_key for sample in samples) + "\n",
            encoding="utf-8",
        )
