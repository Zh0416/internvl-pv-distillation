from __future__ import annotations

import time
from pathlib import Path

import torch
from PIL import Image


def _dynamic_preprocess(image: Image.Image, image_size: int, max_num: int) -> list[Image.Image]:
    width, height = image.size
    aspect = width / height
    candidates = [(1, 1)]
    for rows in range(1, max_num + 1):
        for cols in range(1, max_num + 1):
            count = rows * cols
            if 1 < count <= max_num:
                candidates.append((rows, cols))
    rows, cols = min(candidates, key=lambda item: abs(aspect - item[1] / item[0]))
    resized = image.resize((image_size * cols, image_size * rows), Image.Resampling.BICUBIC)
    tiles = [resized.crop((col * image_size, row * image_size, (col + 1) * image_size, (row + 1) * image_size))
             for row in range(rows) for col in range(cols)]
    return tiles


def load_pixel_values(image_path: str | Path, model, config: dict) -> tuple[torch.Tensor, int]:
    from torchvision import transforms
    image_size = int(getattr(model.config, "force_image_size", 448) or 448)
    max_num = int(config["model"].get("max_dynamic_patch", 6))
    image = Image.open(image_path).convert("RGB")
    tiles = _dynamic_preprocess(image, image_size, max_num)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    values = torch.stack([transform(tile) for tile in tiles])
    device = model.device if hasattr(model, "device") else torch.device("cuda")
    dtype = next((parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()), torch.float16)
    return values.to(device=device, dtype=dtype), len(tiles)


@torch.inference_mode()
def extract_feature(model, image_path: str | Path, config: dict) -> tuple[torch.Tensor, dict]:
    started = time.perf_counter()
    pixel_values, patch_count = load_pixel_values(image_path, model, config)
    feature = model.extract_feature(pixel_values)
    feature_cpu = feature.detach().to(device="cpu", dtype=torch.float16).contiguous()
    stats = {"shape": list(feature_cpu.shape), "dtype": str(feature_cpu.dtype), "device": str(feature.device),
             "min": float(feature_cpu.min().item()), "max": float(feature_cpu.max().item()),
             "mean": float(feature_cpu.float().mean().item()), "std": float(feature_cpu.float().std().item()),
             "patch_count": patch_count, "elapsed_seconds": time.perf_counter() - started}
    del pixel_values, feature
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return feature_cpu, stats
