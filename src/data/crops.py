from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class ComponentStats:
    bbox: tuple[int, int, int, int]
    pixels: int
    component_count: int


@dataclass
class CropMetadata:
    original_size: tuple[int, int]
    target_bbox: tuple[int, int, int, int] | None
    crop_box: tuple[int, int, int, int]
    crop_size: int
    scale_to_teacher: float


def largest_component(mask_path: str | Path) -> ComponentStats:
    with Image.open(mask_path) as mask:
        array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    foreground = array > 0
    coordinates = {tuple(map(int, item)) for item in np.argwhere(foreground)}
    if not coordinates:
        raise ValueError(f"掩膜没有前景像素: {mask_path}")

    largest: list[tuple[int, int]] = []
    component_count = 0
    while coordinates:
        component_count += 1
        start = coordinates.pop()
        queue = deque([start])
        component = [start]
        while queue:
            row, column = queue.popleft()
            for row_offset in (-1, 0, 1):
                for column_offset in (-1, 0, 1):
                    if row_offset == 0 and column_offset == 0:
                        continue
                    neighbor = (row + row_offset, column + column_offset)
                    if neighbor in coordinates:
                        coordinates.remove(neighbor)
                        queue.append(neighbor)
                        component.append(neighbor)
        if len(component) > len(largest):
            largest = component

    rows = [item[0] for item in largest]
    columns = [item[1] for item in largest]
    bbox = (min(columns), min(rows), max(columns) + 1, max(rows) + 1)
    return ComponentStats(bbox=bbox, pixels=len(largest), component_count=component_count)


def square_crop_box(
    image_size: tuple[int, int],
    target_bbox: tuple[int, int, int, int],
    min_crop_size: int,
    context_scale: float,
) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    left, top, right, bottom = target_bbox
    target_width = max(1, right - left)
    target_height = max(1, bottom - top)
    crop_size = int(round(max(min_crop_size, max(target_width, target_height) * context_scale)))
    crop_size = min(crop_size, image_width, image_height)
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    crop_left = int(round(center_x - crop_size / 2))
    crop_top = int(round(center_y - crop_size / 2))
    crop_left = min(max(0, crop_left), image_width - crop_size)
    crop_top = min(max(0, crop_top), image_height - crop_size)
    return crop_left, crop_top, crop_left + crop_size, crop_top + crop_size


def fixed_negative_crop_boxes(image_size: tuple[int, int], crop_size: int, view_count: int) -> list[tuple[int, int, int, int]]:
    image_width, image_height = image_size
    crop_size = min(crop_size, image_width, image_height)
    anchors = [(0.15, 0.15), (0.5, 0.5), (0.85, 0.85), (0.15, 0.85), (0.85, 0.15)]
    boxes = []
    for x_fraction, y_fraction in anchors[:view_count]:
        center_x = image_width * x_fraction
        center_y = image_height * y_fraction
        left = min(max(0, int(round(center_x - crop_size / 2))), image_width - crop_size)
        top = min(max(0, int(round(center_y - crop_size / 2))), image_height - crop_size)
        boxes.append((left, top, left + crop_size, top + crop_size))
    return boxes


def save_crop(
    image_path: str | Path,
    crop_box: tuple[int, int, int, int],
    output_path: str | Path,
    target_bbox: tuple[int, int, int, int] | None = None,
    teacher_size: int = 448,
) -> CropMetadata:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        original_size = rgb.size
        crop = rgb.crop(crop_box)
        crop.save(output_path, format="PNG")
    crop_size = crop_box[2] - crop_box[0]
    return CropMetadata(
        original_size=original_size,
        target_bbox=target_bbox,
        crop_box=crop_box,
        crop_size=crop_size,
        scale_to_teacher=float(teacher_size / crop_size),
    )


def save_diagnostic_panel(
    image_path: str | Path,
    mask_path: str | Path,
    component: ComponentStats,
    crop_box: tuple[int, int, int, int],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        rgb = image.convert("RGB")
        overlay = rgb.copy()
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(component.bbox, outline=(0, 255, 0), width=3)
        draw.rectangle(crop_box, outline=(255, 255, 0), width=3)
        mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        red = np.zeros((mask_array.shape[0], mask_array.shape[1], 4), dtype=np.uint8)
        red[..., 0] = 255
        red[..., 3] = np.where(mask_array > 0, 180, 0).astype(np.uint8)
        mask_overlay = Image.alpha_composite(rgb.convert("RGBA"), Image.fromarray(red, mode="RGBA")).convert("RGB")
        crop = rgb.crop(crop_box).resize(rgb.size, Image.Resampling.NEAREST)
        panel = Image.new("RGB", (rgb.width * 4, rgb.height))
        panel.paste(rgb, (0, 0))
        panel.paste(overlay, (rgb.width, 0))
        panel.paste(mask_overlay, (rgb.width * 2, 0))
        panel.paste(crop, (rgb.width * 3, 0))
        panel.save(output_path, format="PNG")


def crop_metadata_to_dict(metadata: CropMetadata) -> dict:
    return asdict(metadata)
