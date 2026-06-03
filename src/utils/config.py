from __future__ import annotations

from pathlib import Path

import yaml

from src.utils.paths import CONFIG_PATH, PROJECT_ROOT


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else CONFIG_PATH
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
