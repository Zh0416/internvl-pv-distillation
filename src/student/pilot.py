from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def pool_teacher_view_features(
    feature_map: torch.Tensor,
    crop_boxes: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    feature_height, feature_width = feature_map.shape[-2:]
    image_height, image_width = image_size
    local_features = []
    for feature, box in zip(feature_map, crop_boxes):
        left, top, right, bottom = box.tolist()
        start_x = max(0, min(feature_width - 1, left * feature_width // image_width))
        start_y = max(0, min(feature_height - 1, top * feature_height // image_height))
        end_x = max(start_x + 1, min(feature_width, (right * feature_width + image_width - 1) // image_width))
        end_y = max(start_y + 1, min(feature_height, (bottom * feature_height + image_height - 1) // image_height))
        local_features.append(feature[:, start_y:end_y, start_x:end_x].mean(dim=(-2, -1)))
    return torch.stack(local_features)


class PilotDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        image_size: int,
        augment: bool,
        teacher_cache_dir: str | Path | None = None,
    ) -> None:
        self.rows = rows
        self.image_size = image_size
        self.augment = augment
        self.teacher_cache_dir = Path(teacher_cache_dir) if teacher_cache_dir else None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        crop_box = None
        if self.teacher_cache_dir:
            import json

            semantic_path = self.teacher_cache_dir / "semantic" / f"{row['cache_key']}.json"
            semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
            crop = semantic.get("crop")
            crop_box = tuple(crop["crop_box"]) if crop else (0, 0, self.image_size, self.image_size)
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
        if row["mask_path"]:
            with Image.open(row["mask_path"]) as mask_image:
                mask = mask_image.convert("L").resize(
                    (self.image_size, self.image_size), Image.Resampling.NEAREST
                )
        else:
            mask = Image.new("L", (self.image_size, self.image_size), 0)
        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if crop_box:
                    left, top, right, bottom = crop_box
                    crop_box = (self.image_size - right, top, self.image_size - left, bottom)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                if crop_box:
                    left, top, right, bottom = crop_box
                    crop_box = (left, self.image_size - bottom, right, self.image_size - top)
            rotations = random.randrange(4)
            if rotations:
                angle = rotations * 90
                image = image.rotate(angle)
                mask = mask.rotate(angle)
                if crop_box:
                    for _ in range(rotations):
                        left, top, right, bottom = crop_box
                        crop_box = (top, self.image_size - right, bottom, self.image_size - left)
        image_array = np.asarray(image, dtype=np.float32).copy() / 255.0
        mask_array = (np.asarray(mask, dtype=np.uint8).copy() > 0).astype(np.float32)
        pixel_values = torch.from_numpy(image_array).permute(2, 0, 1)
        pixel_values = (pixel_values - IMAGENET_MEAN) / IMAGENET_STD
        result = {
            "pixel_values": pixel_values,
            "mask": torch.from_numpy(mask_array).unsqueeze(0),
            "presence": torch.tensor(float(row["is_positive"]), dtype=torch.float32),
            "sample_weight": torch.tensor(float(row["sample_weight"]), dtype=torch.float32),
            "cache_key": row["cache_key"],
            "selection_group": row["selection_group"],
        }
        if self.teacher_cache_dir:
            feature_path = self.teacher_cache_dir / "features" / f"{row['cache_key']}.pt"
            result["teacher_crop_box"] = torch.tensor(crop_box, dtype=torch.int64)
            result["teacher_probability"] = torch.tensor(
                float(semantic["semantic"]["pv_probability"]), dtype=torch.float32
            )
            result["semantic_weight"] = torch.tensor(
                float(semantic["semantic_weight"]), dtype=torch.float32
            )
            result["teacher_feature"] = torch.load(
                feature_path, map_location="cpu", weights_only=True
            ).float()
        return result


class SegFormerPilot(nn.Module):
    def __init__(self, model_name: str, teacher_dim: int | None = None) -> None:
        super().__init__()
        from transformers import SegformerForSemanticSegmentation

        self.segmenter = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=1,
            id2label={0: "photovoltaic"},
            label2id={"photovoltaic": 0},
            ignore_mismatched_sizes=True,
        )
        hidden_dim = int(self.segmenter.config.hidden_sizes[-1])
        self.presence_head = nn.Linear(hidden_dim, 1)
        self.feature_adapter = nn.Linear(hidden_dim, teacher_dim) if teacher_dim else None

    def forward(self, pixel_values: torch.Tensor, teacher_crop_boxes: torch.Tensor | None = None) -> dict:
        output = self.segmenter(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = F.interpolate(
            output.logits,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        final_feature = output.hidden_states[-1]
        pooled = final_feature.mean(dim=(-2, -1))
        result = {
            "logits": logits,
            "presence_logits": self.presence_head(pooled).squeeze(1),
        }
        if self.feature_adapter is not None:
            teacher_view = pooled
            if teacher_crop_boxes is not None:
                teacher_view = pool_teacher_view_features(
                    final_feature, teacher_crop_boxes, pixel_values.shape[-2:]
                )
            result["teacher_view_presence_logits"] = self.presence_head(teacher_view).squeeze(1)
            result["adapted_feature"] = self.feature_adapter(teacher_view)
        return result


def segmentation_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    sample_weights: torch.Tensor,
    focal_weight: float,
    dice_weight: float,
    positive_pixel_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    binary_cross_entropy = F.binary_cross_entropy_with_logits(
        logits,
        masks,
        reduction="none",
        pos_weight=torch.tensor(positive_pixel_weight, device=logits.device, dtype=logits.dtype),
    )
    probabilities = torch.sigmoid(logits)
    target_probabilities = probabilities * masks + (1.0 - probabilities) * (1.0 - masks)
    focal = (binary_cross_entropy * (1.0 - target_probabilities).pow(2)).mean(dim=(1, 2, 3))
    intersection = (probabilities * masks).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    per_sample = focal_weight * focal + dice_weight * dice
    weighted = (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
    return weighted, {
        "focal": float(focal.mean().detach()),
        "dice_loss": float(dice.mean().detach()),
    }


def distillation_losses(
    output: dict, batch: dict, feature_agreement_only: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    semantic = F.binary_cross_entropy_with_logits(
        output.get("teacher_view_presence_logits", output["presence_logits"]),
        batch["teacher_probability"],
        reduction="none",
    )
    semantic_weights = batch["semantic_weight"] * batch["sample_weight"]
    semantic_loss = (semantic * semantic_weights).sum() / semantic_weights.sum().clamp_min(1.0)
    adapted = F.normalize(output["adapted_feature"].float(), dim=1)
    teacher = F.normalize(batch["teacher_feature"].float(), dim=1)
    feature_per_sample = 1.0 - F.cosine_similarity(adapted, teacher, dim=1)
    feature_weights = batch["sample_weight"]
    if feature_agreement_only:
        feature_weights = feature_weights * batch["semantic_weight"]
    feature_loss = (
        feature_per_sample * feature_weights
    ).sum() / feature_weights.sum().clamp_min(1.0)
    return semantic_loss, feature_loss


def pixel_metrics(true_positive: int, true_negative: int, false_positive: int, false_negative: int) -> dict:
    def divide(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    precision = divide(true_positive, true_positive + false_positive)
    recall = divide(true_positive, true_positive + false_negative)
    dice = divide(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    return {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": dice,
        "dice": dice,
        "iou": divide(true_positive, true_positive + false_positive + false_negative),
        "specificity": divide(true_negative, true_negative + false_positive),
        "accuracy": divide(true_positive + true_negative, true_positive + true_negative + false_positive + false_negative),
        "metric_scope": "pixel-level binary segmentation",
    }
