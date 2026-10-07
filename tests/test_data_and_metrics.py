from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.dataset import inspect_dataset
from src.data.crops import fixed_negative_crop_boxes, largest_component, square_crop_box
from src.data.confusers import score_pv_confuser, select_challenging_negatives
from src.data.hard_samples import mask_geometry
from src.data.pilot import build_pilot_splits
from src.data.selection import select_distillation_subset, select_regular_positives
from src.data.split_dataset import create_splits
from src.student.pilot import distillation_losses, segmentation_loss
from src.teacher.semantic_extractor import parse_semantic_response
from src.utils.metrics import binary_classification_metrics


class DataAndMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.positive_root = root / "pv4026"
        self.negative_root = root / "pv100f"
        (self.positive_root / "images").mkdir(parents=True)
        (self.positive_root / "labels").mkdir()
        (self.negative_root / "images").mkdir(parents=True)
        for index, pixels in enumerate((1, 4, 16, 64), 1):
            image = np.full((8, 8, 3), index * 20, dtype=np.uint8)
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask.flat[:pixels] = 255
            Image.fromarray(image).save(self.positive_root / "images" / f"PV_{index}.tif")
            Image.fromarray(mask).save(self.positive_root / "labels" / f"PV_{index}.tif")
        empty = np.zeros((8, 8), dtype=np.uint8)
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(self.positive_root / "images" / "PV_hard.tif")
        Image.fromarray(empty).save(self.positive_root / "labels" / "PV_hard.tif")
        for index in range(3):
            image = np.full((8, 8, 3), 100 + index, dtype=np.uint8)
            Image.fromarray(image).save(self.negative_root / "images" / f"NEG_{index}.tif")

        self.config = {
            "seed": 42,
            "data": {
                "sources": [
                    {
                        "name": "pv4026",
                        "role": "labeled",
                        "root_candidates": [str(self.positive_root)],
                        "image_dir_name": "images",
                        "mask_dir_names": ["labels"],
                        "mask_suffixes": [""],
                    },
                    {
                        "name": "pv100f",
                        "role": "negative",
                        "root_candidates": [str(self.negative_root)],
                        "image_dir_name": "images",
                    },
                ],
                "image_extensions": [".tif"],
                "image_size": [8, 8],
            },
            "split": {"train": 0.5, "val": 0.25, "test": 0.25, "stratify": True, "group_prefix_parts": 1},
            "selection": {"negative_count": 2, "hard_negative_count": 1, "hard_positive_count": 1, "regular_positive_count": 2},
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_multi_source_loading_selection_and_split(self) -> None:
        _, samples, report = inspect_dataset(self.config)
        self.assertEqual(len(samples), 8)
        self.assertEqual(report["positive_count"], 4)
        self.assertEqual(report["negative_count"], 4)
        self.assertEqual(report["synthetic_negative_count"], 3)
        self.assertEqual(report["sources"][0]["label_values"], [0, 255])
        selected, manifest = select_distillation_subset(samples, self.config)
        self.assertEqual(len(selected), 6)
        self.assertEqual(manifest["available"]["labeled_negatives"], 1)
        self.assertEqual(
            manifest["selected_by_group"],
            {"negative": 2, "hard_negative": 1, "hard_positive": 1, "regular_positive": 2},
        )
        splits = create_splits(samples, self.config)
        self.assertEqual(sum(len(items) for items in splits.values()), 8)
        self.assertTrue(any(sample.is_positive for sample in splits["train"]))
        self.assertTrue(any(not sample.is_positive for sample in splits["train"]))

    def test_metrics_and_semantic_parser(self) -> None:
        metrics = binary_classification_metrics([True, True, False, False], [True, False, True, False])
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["iou"], 1 / 3)
        legacy = parse_semantic_response('{"pv_exists": "false", "confidence": 80, "reason": "none"}')
        self.assertFalse(legacy["pv_exists"])
        self.assertEqual(legacy["confidence"], 0.8)
        self.assertAlmostEqual(legacy["pv_probability"], 0.2)
        parsed = parse_semantic_response('{"pv_exists": false, "pv_probability": 20, "reason": "none"}')
        self.assertFalse(parsed["pv_exists"])
        self.assertEqual(parsed["confidence"], 0.8)
        self.assertEqual(parsed["pv_probability"], 0.2)

    def test_regular_positive_selection_excludes_hard_samples(self) -> None:
        _, samples, _ = inspect_dataset(self.config)
        positives = sorted(
            (sample for sample in samples if sample.is_positive),
            key=lambda sample: sample.pv_ratio,
        )
        hard_keys = {positives[0].cache_key}
        selected, report = select_regular_positives(samples, 2, hard_keys, positives[0].pv_ratio)
        self.assertEqual(len(selected), 2)
        self.assertTrue(hard_keys.isdisjoint(sample.cache_key for sample in selected))
        self.assertEqual(report["excluded_count"], 1)
        self.assertEqual(report["selected_count"], 2)
        self.assertTrue(all(sample.pv_ratio > positives[0].pv_ratio for sample in selected))

    def test_hard_crop_geometry(self) -> None:
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[3:5, 4:7] = 255
        mask[20, 20] = 255
        mask_path = Path(self.temp_dir.name) / "component.tif"
        Image.fromarray(mask).save(mask_path)
        component = largest_component(mask_path)
        geometry = mask_geometry(str(mask_path))
        self.assertEqual(component.pixels, 6)
        self.assertEqual(component.component_count, 2)
        self.assertEqual(component.bbox, (4, 3, 7, 5))
        self.assertFalse(geometry.touches_border)
        self.assertEqual(geometry.border_distance, 3)
        self.assertEqual(geometry.bbox, (4, 3, 21, 21))
        self.assertEqual(square_crop_box((32, 32), component.bbox, 16, 4.0), (0, 0, 16, 16))
        self.assertEqual(len(fixed_negative_crop_boxes((32, 32), 16, 3)), 3)

    def test_confuser_ranking(self) -> None:
        _, samples, _ = inspect_dataset(self.config)
        negatives = [sample for sample in samples if sample.label_origin.startswith("synthetic")]
        score = score_pv_confuser(negatives[0].image_path)
        self.assertGreaterEqual(score.score, 0.0)
        selected, report = select_challenging_negatives(samples, 2)
        self.assertEqual(len(selected), 2)
        self.assertEqual(report["requested"], 2)

    def test_pilot_splits_are_disjoint(self) -> None:
        _, samples, _ = inspect_dataset(self.config)
        config = dict(self.config)
        config["hard_crop"] = {
            "min_mask_pixels": 1,
            "exclude_border_touching": False,
            "min_border_margin": 0,
        }
        config["pilot"] = {
            "fixed_test": {
                "regular_positive": [],
                "hard_positive": ["pv4026__PV_1"],
                "hard_negative": ["pv100f__NEG_0"],
            },
            "counts": {
                "train": {"regular_positive": 1, "hard_positive": 1, "hard_negative": 1},
                "validation": {"regular_positive": 0, "hard_positive": 1, "hard_negative": 1},
                "test": {"regular_positive": 0, "hard_positive": 1, "hard_negative": 1},
            },
            "sample_weights": {
                "regular_positive": 1.0,
                "hard_positive": 1.5,
                "hard_negative": 2.0,
            },
        }
        splits, manifest = build_pilot_splits(samples, config)
        keys = [sample.cache_key for values in splits.values() for sample in values]
        self.assertEqual(len(keys), 7)
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(manifest["counts"]["train"]["regular_positive"], 1)

    def test_student_losses_are_finite(self) -> None:
        logits = torch.zeros((2, 1, 8, 8), requires_grad=True)
        masks = torch.zeros((2, 1, 8, 8))
        masks[0, :, 2:4, 2:4] = 1.0
        sample_weights = torch.tensor([1.5, 2.0])
        segmentation, _ = segmentation_loss(logits, masks, sample_weights, 0.6, 0.4)
        output = {
            "presence_logits": torch.zeros(2, requires_grad=True),
            "adapted_feature": torch.ones((2, 4), requires_grad=True),
        }
        batch = {
            "teacher_probability": torch.tensor([0.9, 0.1]),
            "semantic_weight": torch.tensor([1.0, 0.0]),
            "sample_weight": sample_weights,
            "teacher_feature": torch.ones((2, 4)),
        }
        semantic, feature = distillation_losses(output, batch)
        total = segmentation + semantic + feature
        total.backward()
        self.assertTrue(torch.isfinite(total))


if __name__ == "__main__":
    unittest.main()
