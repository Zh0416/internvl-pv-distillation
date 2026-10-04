from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import Sample


def create_splits(samples: list[Sample], config: dict) -> dict[str, list[Sample]]:
    rng = np.random.default_rng(int(config["seed"]))
    indexes = np.arange(len(samples))
    if config["split"].get("stratify", True):
        positive = [i for i, sample in enumerate(samples) if sample.is_positive]
        negative = [i for i, sample in enumerate(samples) if not sample.is_positive]
        rng.shuffle(positive); rng.shuffle(negative)
        indexes = np.array(positive + negative, dtype=int)
    else:
        rng.shuffle(indexes)
    train_end = int(len(indexes) * float(config["split"]["train"]))
    val_end = train_end + int(len(indexes) * float(config["split"]["val"]))
    return {"train": [samples[i] for i in indexes[:train_end]],
            "val": [samples[i] for i in indexes[train_end:val_end]],
            "test": [samples[i] for i in indexes[val_end:]]}


def save_splits(splits: dict[str, list[Sample]], output_dir: str | Path) -> None:
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True)
    for name, samples in splits.items():
        (directory / f"{name}.txt").write_text("\n".join(s.filename for s in samples) + "\n", encoding="utf-8")
