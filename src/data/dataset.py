from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass
class Sample:
    filename: str
    image_path: str
    mask_path: str
    stem: str
    pv_pixels: int
    pv_ratio: float
    is_positive: bool
    group: str | None


def resolve_dataset_root(candidates: Iterable[str | Path], image_name: str = "images") -> Path:
    checked = []
    for candidate in candidates:
        root = Path(candidate)
        checked.append(str(root))
        if (root / image_name).is_dir() and any((root / image_name).iterdir()):
            return root
    raise FileNotFoundError(f"未找到包含 {image_name}/ 的数据根目录，已检查: {checked}")


def find_mask_dir(root: Path, names: Iterable[str]) -> Path:
    names = list(names)
    for name in names:
        path = root / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"未找到mask目录: {names} under {root}")


def _group_from_stem(stem: str, prefix_parts: int) -> str | None:
    parts = stem.split("_")
    return "_".join(parts[:prefix_parts]) if len(parts) >= prefix_parts else None


def inspect_dataset(config: dict) -> tuple[Path, list[Sample], dict]:
    data_cfg = config["data"]
    root = resolve_dataset_root(data_cfg["root_candidates"], data_cfg["image_dir_name"])
    image_dir = root / data_cfg["image_dir_name"]
    mask_dir = find_mask_dir(root, data_cfg["mask_dir_names"])
    extensions = {str(ext).lower() for ext in data_cfg["image_extensions"]}
    suffix = str(data_cfg.get("mask_suffix", "_label"))
    samples: list[Sample] = []
    errors: list[dict] = []
    missing_masks: list[str] = []
    invalid_images: list[str] = []
    invalid_masks: list[str] = []
    image_paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    all_masks = {p.stem: p for p in mask_dir.iterdir() if p.is_file()}
    for image_path in image_paths:
        mask_path = all_masks.get(f"{image_path.stem}{suffix}")
        if mask_path is None:
            missing_masks.append(image_path.name)
            continue
        try:
            with Image.open(image_path) as image:
                if image.size != tuple(data_cfg["image_size"]) or image.mode not in {"RGB", "RGBA"}:
                    invalid_images.append(image_path.name)
            with Image.open(mask_path) as mask:
                mask_array = np.asarray(mask)
                if mask_array.ndim == 3:
                    mask_array = mask_array[..., 0]
                foreground = mask_array > 0
                height, width = foreground.shape
                pv_pixels = int(foreground.sum())
                ratio = float(pv_pixels / (height * width)) if height and width else 0.0
                if (width, height) != tuple(data_cfg["image_size"]):
                    invalid_masks.append(mask_path.name)
            samples.append(Sample(image_path.name, str(image_path), str(mask_path), image_path.stem,
                                  pv_pixels, ratio, pv_pixels > 0,
                                  _group_from_stem(image_path.stem, int(config["split"].get("group_prefix_parts", 1)))))
        except Exception as exc:
            errors.append({"filename": image_path.name, "error": str(exc)})
    ratios = [sample.pv_ratio for sample in samples]
    report = {
        "root": str(root), "image_dir": str(image_dir), "mask_dir": str(mask_dir),
        "total_images": len(image_paths), "valid_mask_count": len(samples),
        "missing_mask_count": len(missing_masks), "empty_mask_count": sum(not s.is_positive for s in samples),
        "positive_count": sum(s.is_positive for s in samples), "negative_count": sum(not s.is_positive for s in samples),
        "pv_pixels": {s.filename: s.pv_pixels for s in samples},
        "pv_ratio": {s.filename: s.pv_ratio for s in samples},
        "pv_ratio_summary": {"min": float(np.min(ratios)) if ratios else 0.0, "max": float(np.max(ratios)) if ratios else 0.0,
                             "mean": float(np.mean(ratios)) if ratios else 0.0, "median": float(np.median(ratios)) if ratios else 0.0},
        "image_mask_one_to_one": len(image_paths) == len(samples) and not missing_masks,
        "invalid_images": invalid_images, "invalid_masks": invalid_masks, "missing_masks": missing_masks,
        "read_errors": errors,
        "filename_rule_note": "group字段按文件名前缀解析；是否为真实scene需结合数据集语义确认，默认不依赖该猜测进行划分。",
    }
    return root, samples, report


def sample_to_dict(sample: Sample) -> dict:
    return asdict(sample)


def config_hash(config: dict) -> str:
    serialized = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:16]
