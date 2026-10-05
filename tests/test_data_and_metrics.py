from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.dataset import inspect_dataset
from src.data.crops import fixed_negative_crop_boxes, largest_component, square_crop_box
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
        parsed = parse_semantic_response('{"pv_exists": "false", "confidence": 80, "reason": "none"}')
        self.assertFalse(parsed["pv_exists"])
        self.assertEqual(parsed["confidence"], 0.8)

    def test_hard_crop_geometry(self) -> None:
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[3:5, 4:7] = 255
        mask[20, 20] = 255
        mask_path = Path(self.temp_dir.name) / "component.tif"
        Image.fromarray(mask).save(mask_path)
        component = largest_component(mask_path)
        self.assertEqual(component.pixels, 6)
        self.assertEqual(component.component_count, 2)
        self.assertEqual(component.bbox, (4, 3, 7, 5))
        self.assertEqual(square_crop_box((32, 32), component.bbox, 16, 4.0), (0, 0, 16, 16))
        self.assertEqual(len(fixed_negative_crop_boxes((32, 32), 16, 3)), 3)


if __name__ == "__main__":
    unittest.main()
