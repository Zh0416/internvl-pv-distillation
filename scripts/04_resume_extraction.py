from __future__ import annotations

import runpy
import sys

from common import load_config, seed_everything


if __name__ == "__main__":
    config_path = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "configs/internvl3_5_teacher.yaml"
    config = load_config(config_path); seed_everything(int(config["seed"]))
    sys.argv = ["03_extract_teacher_features.py", "--config", config_path, "--max-samples", "None"]
    runpy.run_path("scripts/03_extract_teacher_features.py", run_name="__main__")
