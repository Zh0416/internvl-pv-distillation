from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass
class Sample:
    filename: str
    image_path: str
    mask_path: str | None
    stem: str
    cache_key: str
    source: str
    label_origin: str
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


def _cache_key(source: str, stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{source}__{stem}")


def _source_configs(data_cfg: dict) -> list[dict]:
    configured = data_cfg.get("sources")
    if configured:
        return [dict(source) for source in configured]
    return [
        {
            "name": "dataset",
            "role": "labeled",
            "root_candidates": data_cfg["root_candidates"],
            "image_dir_name": data_cfg.get("image_dir_name", "images"),
            "mask_dir_names": data_cfg.get("mask_dir_names", ["labels", "mask", "masks"]),
            "mask_suffixes": [data_cfg.get("mask_suffix", "_label")],
        }
    ]


def _matching_mask(image_stem: str, masks: dict[str, Path], suffixes: Iterable[str]) -> Path | None:
    for suffix in suffixes:
        candidate = masks.get(f"{image_stem}{suffix}")
        if candidate is not None:
            return candidate
    return None


def inspect_dataset(config: dict) -> tuple[Path, list[Sample], dict]:
    data_cfg = config["data"]
    extensions = {str(ext).lower() for ext in data_cfg["image_extensions"]}
    expected_size = tuple(data_cfg["image_size"])
    prefix_parts = int(config["split"].get("group_prefix_parts", 1))
    samples: list[Sample] = []
    source_reports: list[dict] = []
    roots: list[Path] = []
    cache_keys: set[str] = set()

    for source_cfg in _source_configs(data_cfg):
        source_name = str(source_cfg["name"])
        role = str(source_cfg.get("role", "labeled")).lower()
        image_dir_name = str(source_cfg.get("image_dir_name", data_cfg.get("image_dir_name", "images")))
        root = resolve_dataset_root(source_cfg["root_candidates"], image_dir_name)
        roots.append(root)
        image_dir = root / image_dir_name
        image_paths = sorted(
            path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions
        )
        mask_dir: Path | None = None
        masks: dict[str, Path] = {}
        if role != "negative":
            mask_names = source_cfg.get(
                "mask_dir_names",
                data_cfg.get("mask_dir_names", ["labels", "mask", "masks"]),
            )
            mask_dir = find_mask_dir(root, mask_names)
            masks = {path.stem: path for path in mask_dir.iterdir() if path.is_file()}

        suffixes = source_cfg.get(
            "mask_suffixes",
            data_cfg.get("mask_suffixes", [data_cfg.get("mask_suffix", "_label")]),
        )
        missing_masks: list[str] = []
        invalid_images: list[str] = []
        invalid_masks: list[str] = []
        read_errors: list[dict] = []
        image_modes: set[str] = set()
        image_sizes: set[tuple[int, int]] = set()
        image_formats: set[str] = set()
        mask_modes: set[str] = set()
        mask_sizes: set[tuple[int, int]] = set()
        label_values: set[int] = set()
        source_samples: list[Sample] = []

        for image_path in image_paths:
            mask_path = None if role == "negative" else _matching_mask(image_path.stem, masks, suffixes)
            if role != "negative" and mask_path is None:
                missing_masks.append(image_path.name)
                continue
            try:
                with Image.open(image_path) as image:
                    image_modes.add(image.mode)
                    image_sizes.add(image.size)
                    image_formats.add(str(image.format))
                    if image.size != expected_size or image.mode not in {"RGB", "RGBA"}:
                        invalid_images.append(image_path.name)

                if mask_path is None:
                    pv_pixels = 0
                    ratio = 0.0
                    label_origin = "synthetic_zero_from_negative_source"
                else:
                    with Image.open(mask_path) as mask:
                        mask_modes.add(mask.mode)
                        mask_sizes.add(mask.size)
                        mask_array = np.asarray(mask)
                        if mask_array.ndim == 3:
                            mask_array = mask_array[..., 0]
                        if mask_array.ndim != 2:
                            raise ValueError(f"标签必须是单通道，实际shape={mask_array.shape}")
                        label_values.update(int(value) for value in np.unique(mask_array))
                        foreground = mask_array > 0
                        height, width = foreground.shape
                        pv_pixels = int(foreground.sum())
                        ratio = float(pv_pixels / (height * width)) if height and width else 0.0
                        if (width, height) != expected_size:
                            invalid_masks.append(mask_path.name)
                    label_origin = "mask"

                cache_key = _cache_key(source_name, image_path.stem)
                if cache_key in cache_keys:
                    raise ValueError(f"重复cache_key: {cache_key}")
                cache_keys.add(cache_key)
                source_samples.append(
                    Sample(
                        filename=image_path.name,
                        image_path=str(image_path),
                        mask_path=str(mask_path) if mask_path else None,
                        stem=image_path.stem,
                        cache_key=cache_key,
                        source=source_name,
                        label_origin=label_origin,
                        pv_pixels=pv_pixels,
                        pv_ratio=ratio,
                        is_positive=pv_pixels > 0,
                        group=_group_from_stem(image_path.stem, prefix_parts),
                    )
                )
            except Exception as exc:
                read_errors.append({"filename": image_path.name, "error": str(exc)})

        samples.extend(source_samples)
        source_reports.append(
            {
                "name": source_name,
                "role": role,
                "root": str(root),
                "image_dir": str(image_dir),
                "mask_dir": str(mask_dir) if mask_dir else None,
                "image_count": len(image_paths),
                "valid_count": len(source_samples),
                "positive_count": sum(sample.is_positive for sample in source_samples),
                "negative_count": sum(not sample.is_positive for sample in source_samples),
                "missing_mask_count": len(missing_masks),
                "invalid_image_count": len(invalid_images),
                "invalid_mask_count": len(invalid_masks),
                "image_modes": sorted(image_modes),
                "image_sizes": sorted([list(size) for size in image_sizes]),
                "image_formats": sorted(image_formats),
                "mask_modes": sorted(mask_modes),
                "mask_sizes": sorted([list(size) for size in mask_sizes]),
                "label_values": sorted(label_values),
                "missing_masks": missing_masks,
                "invalid_images": invalid_images,
                "invalid_masks": invalid_masks,
                "read_errors": read_errors,
            }
        )

    ratios = [sample.pv_ratio for sample in samples]
    report = {
        "root": str(roots[0]),
        "roots": [str(root) for root in roots],
        "sources": source_reports,
        "total_images": sum(source["image_count"] for source in source_reports),
        "valid_mask_count": sum(sample.mask_path is not None for sample in samples),
        "synthetic_negative_count": sum(sample.label_origin.startswith("synthetic") for sample in samples),
        "missing_mask_count": sum(source["missing_mask_count"] for source in source_reports),
        "empty_mask_count": sum(sample.mask_path is not None and not sample.is_positive for sample in samples),
        "positive_count": sum(sample.is_positive for sample in samples),
        "negative_count": sum(not sample.is_positive for sample in samples),
        "pv_pixels": {sample.cache_key: sample.pv_pixels for sample in samples},
        "pv_ratio": {sample.cache_key: sample.pv_ratio for sample in samples},
        "pv_ratio_summary": {
            "min": float(np.min(ratios)) if ratios else 0.0,
            "max": float(np.max(ratios)) if ratios else 0.0,
            "mean": float(np.mean(ratios)) if ratios else 0.0,
            "median": float(np.median(ratios)) if ratios else 0.0,
        },
        "image_mask_one_to_one": all(
            source["role"] == "negative" or source["image_count"] == source["valid_count"]
            for source in source_reports
        ),
        "filename_rule_note": "cache_key包含数据源名以避免跨数据集同名冲突；group仅用于可选的场景级划分。",
    }
    return roots[0], samples, report


def sample_to_dict(sample: Sample) -> dict:
    return asdict(sample)


def config_hash(config: dict) -> str:
    serialized = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:16]
