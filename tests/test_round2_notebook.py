from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path


class RoundTwoNotebookTest(unittest.TestCase):
    def test_stage_runner_records_success_and_timeout(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "notebook5b148ab676_round2.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
        checkout_cell = next(cell for cell in notebook["cells"] if cell["id"] == "github-checkout")
        function = next(
            node for node in ast.parse("".join(checkout_cell["source"])).body
            if isinstance(node, ast.FunctionDef) and node.name == "run_external"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage_log_dir = root / "stage_logs"
            stage_log_dir.mkdir()
            stage_status_path = root / "reports" / "round2_execution_status.json"
            stage_status_path.parent.mkdir()
            namespace = {
                "datetime": datetime,
                "timezone": timezone,
                "time": time,
                "json": json,
                "subprocess": subprocess,
                "Path": Path,
                "stage_log_dir": stage_log_dir,
                "stage_status_path": stage_status_path,
                "stage_records": [],
            }
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(notebook_path), "exec"), namespace)
            namespace["run_external"]("success", [sys.executable, "-c", "print('ok')"], 10)
            with self.assertRaises(subprocess.TimeoutExpired):
                namespace["run_external"]("timeout", [sys.executable, "-c", "import time; time.sleep(5)"], 0.1)
            records = json.loads(stage_status_path.read_text(encoding="utf-8"))
            self.assertEqual([record["status"] for record in records], ["complete", "timeout"])
            self.assertIn("ok", (stage_log_dir / "success.log").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
