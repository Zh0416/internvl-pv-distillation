from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.dataset import inspect_dataset
from src.data.selection import select_distillation_subset
from src.data.split_dataset import create_splits
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
        parsed = parse_semantic_response('{"pv_exists": "false", "confidence": 80, "reason": "none"}')
        self.assertFalse(parsed["pv_exists"])
        self.assertEqual(parsed["confidence"], 0.8)


if __name__ == "__main__":
    unittest.main()
