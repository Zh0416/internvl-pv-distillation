from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

from .dataset import Sample, sample_to_dict


@dataclass
class ConfuserScore:
    score: float
    dark_fraction: float
    cool_fraction: float
    edge_fraction: float


def score_pv_confuser(image_path: str) -> ConfuserScore:
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    red, green, blue = np.moveaxis(rgb, -1, 0)
    gray = rgb.mean(axis=2)
    dark_fraction = float(((gray >= 20) & (gray <= 125)).mean())
    cool_fraction = float(((blue >= red * 1.03) & (blue >= green * 0.93) & (gray <= 190)).mean())
    horizontal = np.abs(np.diff(gray, axis=1))
    vertical = np.abs(np.diff(gray, axis=0))
    edge_fraction = float((horizontal > 24).mean() + (vertical > 24).mean()) / 2.0
    score = 0.4 * cool_fraction + 0.35 * dark_fraction + 0.25 * edge_fraction
    return ConfuserScore(
        score=float(score),
        dark_fraction=dark_fraction,
        cool_fraction=cool_fraction,
        edge_fraction=edge_fraction,
    )


def select_challenging_negatives(samples: list[Sample], count: int) -> tuple[list[Sample], dict]:
    candidates = [sample for sample in samples if not sample.is_positive]
    ranked = sorted(
        ((sample, score_pv_confuser(sample.image_path)) for sample in candidates),
        key=lambda item: (-item[1].score, item[0].cache_key),
    )
    selected = ranked[:count]
    if len(selected) < count:
        raise RuntimeError(f"困难负样本不足: requested={count}, found={len(selected)}")
    return [sample for sample, _ in selected], {
        "requested": count,
        "available": len(candidates),
        "ranking": "0.40*cool_fraction + 0.35*dark_fraction + 0.25*edge_fraction",
        "selected": [
            {"sample": sample_to_dict(sample), "confuser": asdict(score)}
            for sample, score in selected
        ],
    }
