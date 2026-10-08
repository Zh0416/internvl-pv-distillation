from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from common import dump_json, load_config, seed_everything
from src.student.pilot import (
    PilotDataset,
    SegFormerPilot,
    distillation_losses,
    pixel_metrics,
    segmentation_loss,
)
from src.teacher.load_internvl import gpu_memory
from src.utils.checkpoint import utc_now
from src.utils.logger import setup_logger
from src.utils.metrics import binary_classification_metrics


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _evaluate(model, loader, device, threshold: float, min_presence_pixels: int) -> dict:
    model.eval()
    counters = defaultdict(lambda: [0, 0, 0, 0])
    image_truth = []
    image_prediction = []
    image_groups = []
    sample_results = []
    with torch.inference_mode():
        for batch in loader:
            batch = _move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(batch["pixel_values"])
            predictions = torch.sigmoid(output["logits"]) >= threshold
            truth = batch["mask"] >= 0.5
            for index, group in enumerate(batch["selection_group"]):
                expected = truth[index]
                actual = predictions[index]
                true_positive = int((expected & actual).sum().item())
                true_negative = int((~expected & ~actual).sum().item())
                false_positive = int((~expected & actual).sum().item())
                false_negative = int((expected & ~actual).sum().item())
                values = [true_positive, true_negative, false_positive, false_negative]
                counters["overall"] = [left + right for left, right in zip(counters["overall"], values)]
                counters[group] = [left + right for left, right in zip(counters[group], values)]
                expected_presence = bool(batch["presence"][index].item())
                predicted_pixels = int(actual.sum().item())
                predicted_presence = predicted_pixels >= min_presence_pixels
                image_truth.append(expected_presence)
                image_prediction.append(predicted_presence)
                image_groups.append(group)
                sample_results.append(
                    {
                        "cache_key": batch["cache_key"][index],
                        "selection_group": group,
                        "ground_truth_presence": expected_presence,
                        "predicted_presence": predicted_presence,
                        "predicted_pixels": predicted_pixels,
                        "pixel_metrics": pixel_metrics(*values),
                    }
                )
    metrics_by_group = {
        group: pixel_metrics(*values)
        for group, values in counters.items()
    }
    image_metrics_by_group = {"overall": binary_classification_metrics(image_truth, image_prediction)}
    for group in sorted(set(image_groups)):
        indexes = [index for index, value in enumerate(image_groups) if value == group]
        image_metrics_by_group[group] = binary_classification_metrics(
            [image_truth[index] for index in indexes],
            [image_prediction[index] for index in indexes],
        )
    return {
        "pixel_metrics": metrics_by_group,
        "image_presence_metrics": image_metrics_by_group,
        "samples": sample_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kaggle_pv_pilot_distill.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--variant", required=True, choices=("baseline", "kd"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hard-positive-weight", type=float, default=None)
    parser.add_argument("--positive-pixel-weight", type=float, default=1.0)
    parser.add_argument("--semantic-kd-weight", type=float, default=None)
    parser.add_argument("--feature-kd-weight", type=float, default=None)
    parser.add_argument("--feature-agreement-only", action="store_true")
    parser.add_argument("--defer-test", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    run_name = args.run_name or args.variant
    if not re.fullmatch(r"[a-z0-9_]+", run_name):
        raise ValueError(f"无效的运行名称: {run_name}")
    config = load_config(args.config)
    if args.output_dir:
        override_dir = Path(args.output_dir)
        if not override_dir.is_absolute():
            raise ValueError(f"输出路径必须是绝对路径: {override_dir}")
        config["runtime"]["output_dir"] = str(override_dir)
    seed_everything(int(config["seed"]))
    student_cfg = dict(config["pilot"]["student"])
    if args.epochs is not None:
        student_cfg["epochs"] = args.epochs
    if args.semantic_kd_weight is not None:
        student_cfg["semantic_kd_weight"] = args.semantic_kd_weight
    if args.feature_kd_weight is not None:
        student_cfg["feature_kd_weight"] = args.feature_kd_weight
    if args.positive_pixel_weight <= 0 or (args.hard_positive_weight is not None and args.hard_positive_weight <= 0):
        raise ValueError("训练权重必须为正数")
    output_dir = Path(config["runtime"]["output_dir"])
    manifest = json.loads((output_dir / "reports" / "pilot_manifest.json").read_text(encoding="utf-8"))
    teacher_cache_dir = output_dir / "pilot_teacher_cache"
    teacher_dim = None
    if args.variant == "kd":
        teacher_report = json.loads(
            (output_dir / "reports" / "pilot_teacher_report.json").read_text(encoding="utf-8")
        )
        if teacher_report["failure_count"]:
            raise RuntimeError("Teacher缓存不完整，不能训练KD Student")
        teacher_dim = int(teacher_report["feature_dim"])
    run_dir = output_dir / "pilot_student" / run_name
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir / "training.log", f"pilot_{run_name}")
    image_size = int(student_cfg["image_size"])
    train_rows = [dict(row) for row in manifest["splits"]["train"]]
    if args.hard_positive_weight is not None:
        for row in train_rows:
            if row["selection_group"] == "hard_positive":
                row["sample_weight"] = args.hard_positive_weight
    train_dataset = PilotDataset(
        train_rows,
        image_size,
        augment=True,
        teacher_cache_dir=teacher_cache_dir if args.variant == "kd" else None,
    )
    validation_dataset = PilotDataset(manifest["splits"]["validation"], image_size, augment=False)
    test_dataset = PilotDataset(manifest["splits"]["test"], image_size, augment=False)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader_kwargs = {
        "batch_size": int(student_cfg["batch_size"]),
        "num_workers": int(student_cfg["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": int(student_cfg["num_workers"]) > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegFormerPilot(student_cfg["name"], teacher_dim=teacher_dim).to(device)
    report_path = output_dir / "reports" / f"pilot_{run_name}_report.json"
    best_path = checkpoint_dir / "best.pt"
    if args.evaluate_only:
        if not report_path.is_file() or not best_path.is_file():
            raise FileNotFoundError(f"缺少训练报告或最佳权重: {run_name}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        thresholds = [float(value) for value in args.thresholds.split(",")] if args.thresholds else [float(student_cfg["threshold"])]
        if any(not 0 < value < 1 for value in thresholds):
            raise ValueError("阈值必须在0和1之间")
        if args.test:
            if len(thresholds) != 1:
                raise ValueError("测试时只能指定一个已由验证集选定的阈值")
            if str(thresholds[0]) not in report.get("validation_thresholds", {}):
                raise ValueError("测试阈值必须先通过验证集扫描")
            report["selected_threshold"] = thresholds[0]
            train_eval_dataset = PilotDataset(manifest["splits"]["train"], image_size, augment=False)
            train_eval_loader = DataLoader(train_eval_dataset, shuffle=False, **loader_kwargs)
            report["train"] = _evaluate(model, train_eval_loader, device, thresholds[0], int(config["hard_crop"]["min_mask_pixels"]))
            report["test"] = _evaluate(model, test_loader, device, thresholds[0], int(config["hard_crop"]["min_mask_pixels"]))
        else:
            report["validation_thresholds"] = {
                str(threshold): _evaluate(model, validation_loader, device, threshold, int(config["hard_crop"]["min_mask_pixels"]))
                for threshold in thresholds
            }
        dump_json(report, report_path)
        print(json.dumps({"run_name": run_name, "validation_thresholds": {key: value["pixel_metrics"] for key, value in report["validation_thresholds"].items()}, "test": report.get("test", {}).get("pixel_metrics") if report.get("test") else None}, ensure_ascii=False, indent=2))
        return
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(student_cfg["learning_rate"]),
        weight_decay=float(student_cfg["weight_decay"]),
    )
    accumulation = int(student_cfg["gradient_accumulation_steps"])
    epochs = int(student_cfg["epochs"])
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = steps_per_epoch * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_dice = -1.0
    started = time.perf_counter()
    logger.info("variant=%s run_name=%s train=%d validation=%d test=%d gpu=%s", args.variant, run_name, len(train_dataset), len(validation_dataset), len(test_dataset), gpu_memory())
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_totals = defaultdict(float)
        for step, batch in enumerate(train_loader, 1):
            batch = _move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(batch["pixel_values"], batch.get("teacher_crop_box"))
                segmentation, components = segmentation_loss(
                    output["logits"],
                    batch["mask"],
                    batch["sample_weight"],
                    float(student_cfg["focal_weight"]),
                    float(student_cfg["dice_weight"]),
                    args.positive_pixel_weight,
                )
                presence = F.binary_cross_entropy_with_logits(
                    output["presence_logits"], batch["presence"]
                )
                semantic_kd = torch.zeros((), device=device)
                feature_kd = torch.zeros((), device=device)
                if args.variant == "kd":
                    semantic_kd, feature_kd = distillation_losses(output, batch, args.feature_agreement_only)
                total = (
                    segmentation
                    + float(student_cfg["presence_supervision_weight"]) * presence
                    + float(student_cfg["semantic_kd_weight"]) * semantic_kd
                    + float(student_cfg["feature_kd_weight"]) * feature_kd
                )
                scaled_loss = total / accumulation
            scaler.scale(scaled_loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            epoch_totals["total"] += float(total.detach())
            epoch_totals["segmentation"] += float(segmentation.detach())
            epoch_totals["presence"] += float(presence.detach())
            epoch_totals["semantic_kd"] += float(semantic_kd.detach())
            epoch_totals["feature_kd"] += float(feature_kd.detach())
            epoch_totals["focal"] += components["focal"]
            epoch_totals["dice_loss"] += components["dice_loss"]
        validation = _evaluate(
            model,
            validation_loader,
            device,
            float(student_cfg["threshold"]),
            int(config["hard_crop"]["min_mask_pixels"]),
        )
        validation_dice = validation["pixel_metrics"]["overall"]["dice"]
        averages = {key: value / len(train_loader) for key, value in epoch_totals.items()}
        epoch_record = {
            "epoch": epoch,
            "train_losses": averages,
            "validation_pixel_metrics": validation["pixel_metrics"],
            "validation_image_presence_metrics": validation["image_presence_metrics"],
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(epoch_record)
        logger.info("epoch=%d loss=%.4f val_dice=%.4f val_iou=%.4f", epoch, averages["total"], validation_dice, validation["pixel_metrics"]["overall"]["iou"])
        if validation_dice > best_dice:
            best_dice = validation_dice
            torch.save({"model": model.state_dict(), "epoch": epoch, "validation_dice": best_dice}, best_path)
        dump_json({"variant": args.variant, "run_name": run_name, "history": history}, run_dir / "history.json")
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    validation = _evaluate(
        model,
        validation_loader,
        device,
        float(student_cfg["threshold"]),
        int(config["hard_crop"]["min_mask_pixels"]),
    )
    test = None
    if not args.defer_test:
        test = _evaluate(
            model,
            test_loader,
            device,
            float(student_cfg["threshold"]),
            int(config["hard_crop"]["min_mask_pixels"]),
        )
    report = {
        "created_at": utc_now(),
        "variant": args.variant,
        "run_name": run_name,
        "training_performed": True,
        "parameters": {
            "student": student_cfg,
            "hard_positive_weight": args.hard_positive_weight,
            "positive_pixel_weight": args.positive_pixel_weight,
            "feature_agreement_only": args.feature_agreement_only,
            "teacher_dim": teacher_dim,
            "train_count": len(train_dataset),
            "validation_count": len(validation_dataset),
            "test_count": len(test_dataset),
        },
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_dice": float(checkpoint["validation_dice"]),
        "history": history,
        "validation": validation,
        "test": test,
        "runtime_seconds": time.perf_counter() - started,
        "gpu": gpu_memory(),
        "checkpoint": str(best_path),
    }
    dump_json(report, report_path)
    print(json.dumps({
        "variant": args.variant,
        "best_epoch": report["best_epoch"],
        "validation": validation["pixel_metrics"]["overall"],
        "test": test["pixel_metrics"]["overall"] if test else None,
        "test_by_group": test["pixel_metrics"] if test else None,
        "runtime_seconds": report["runtime_seconds"],
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
