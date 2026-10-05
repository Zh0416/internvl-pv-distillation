from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from common import dump_json, load_config, seed_everything
from src.data.crops import largest_component, save_diagnostic_panel, square_crop_box
from src.data.dataset import inspect_dataset
from src.data.selection import select_distillation_subset


def _apply_overrides(config: dict, values: list[str], output_dir: str | None) -> None:
    sources = {source["name"]: source for source in config["data"]["sources"]}
    for value in values:
        name, root = value.split("=", 1)
        root_path = Path(root)
        if not root_path.is_absolute():
            raise ValueError(f"数据源路径必须是绝对路径: {root}")
        sources[name]["root_candidates"] = [str(root_path)]
    if output_dir:
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            raise ValueError(f"输出路径必须是绝对路径: {output_dir}")
        config["runtime"]["output_dir"] = str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kaggle_pv_hard_crop_distill.yaml")
    parser.add_argument("--source-root", action="append", default=[])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    _apply_overrides(config, args.source_root, args.output_dir)
    seed_everything(int(config["seed"]))

    _, samples, _ = inspect_dataset(config)
    selected, _ = select_distillation_subset(samples, config)
    hard_samples = [sample for sample in selected if sample.is_positive]
    output_dir = Path(config["runtime"]["output_dir"]) / "reports" / "hard_sample_diagnostics"
    rows = []
    for sample in hard_samples:
        component = largest_component(sample.mask_path)
        with Image.open(sample.image_path) as image:
            image_size = image.size
        crop_box = square_crop_box(
            image_size,
            component.bbox,
            int(config["hard_crop"]["min_crop_size"]),
            float(config["hard_crop"]["context_scale"]),
        )
        panel_path = output_dir / f"{sample.cache_key}.png"
        save_diagnostic_panel(sample.image_path, sample.mask_path, component, crop_box, panel_path)
        rows.append(
            {
                "cache_key": sample.cache_key,
                "filename": sample.filename,
                "mask_pixels": sample.pv_pixels,
                "mask_ratio": sample.pv_ratio,
                "largest_component_pixels": component.pixels,
                "component_count": component.component_count,
                "component_bbox": component.bbox,
                "crop_box": crop_box,
                "crop_size": crop_box[2] - crop_box[0],
                "panel_path": str(panel_path),
            }
        )
    report_path = output_dir.parent / "hard_sample_diagnostics.json"
    dump_json({"count": len(rows), "samples": rows}, report_path)
    print(f"Hard samples: {len(rows)}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
