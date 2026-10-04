from __future__ import annotations

from pathlib import Path


def save_feature_summary(summary: dict, path: str | Path) -> None:
    """Persist a lightweight feature summary without loading image data."""
    import json
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
