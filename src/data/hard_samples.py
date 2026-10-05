from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

from .dataset import Sample, sample_to_dict


@dataclass
class MaskGeometry:
    pixels: int
    bbox: tuple[int, int, int, int]
    touches_border: bool
    border_distance: int
    bbox_width: int
    bbox_height: int


def mask_geometry(mask_path: str) -> MaskGeometry:
    with Image.open(mask_path) as mask:
        array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    foreground = array > 0
    coordinates = np.argwhere(foreground)
    if not len(coordinates):
        raise ValueError(f"掩膜没有前景像素: {mask_path}")
    top, left = coordinates.min(axis=0).tolist()
    bottom, right = (coordinates.max(axis=0) + 1).tolist()
    height, width = foreground.shape
    touches_border = left == 0 or top == 0 or right == width or bottom == height
    border_distance = min(left, top, width - right, height - bottom)
    return MaskGeometry(
        pixels=int(foreground.sum()),
        bbox=(int(left), int(top), int(right), int(bottom)),
        touches_border=touches_border,
        border_distance=int(border_distance),
        bbox_width=int(right - left),
        bbox_height=int(bottom - top),
    )


def select_valid_hard_positives(samples: list[Sample], config: dict) -> tuple[list[Sample], dict]:
    hard_cfg = config["hard_crop"]
    requested = int(config["selection"]["hard_positive_count"])
    min_pixels = int(hard_cfg.get("min_mask_pixels", 128))
    exclude_border = bool(hard_cfg.get("exclude_border_touching", True))
    min_border_margin = int(hard_cfg.get("min_border_margin", 0))
    candidates = sorted((sample for sample in samples if sample.is_positive), key=lambda sample: sample.pv_pixels)
    selected: list[Sample] = []
    excluded: list[dict] = []
    for sample in candidates:
        geometry = mask_geometry(sample.mask_path)
        reason = None
        if geometry.pixels < min_pixels:
            reason = "below_min_mask_pixels"
        elif exclude_border and geometry.touches_border:
            reason = "touches_tile_border"
        elif geometry.border_distance < min_border_margin:
            reason = "within_border_margin"
        if reason:
            excluded.append(
                {
                    "sample": sample_to_dict(sample),
                    "geometry": asdict(geometry),
                    "reason": reason,
                    "semantic_distillation_weight": 0.0,
                }
            )
            continue
        selected.append(sample)
        if len(selected) >= requested:
            break
    if len(selected) < requested:
        raise RuntimeError(f"符合困难样本质量规则的正样本不足: requested={requested}, found={len(selected)}")
    report = {
        "requested": requested,
        "selected_count": len(selected),
        "min_mask_pixels": min_pixels,
        "exclude_border_touching": exclude_border,
        "min_border_margin": min_border_margin,
        "excluded_before_selection": excluded,
        "selected": [
            {"sample": sample_to_dict(sample), "geometry": asdict(mask_geometry(sample.mask_path))}
            for sample in selected
        ],
    }
    return selected, report
