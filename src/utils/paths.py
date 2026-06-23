from __future__ import annotations

from pathlib import Path

# Raiz do repositório (pasta que contém src/, config/, data/, …)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
METADATA_PATH = PROJECT_ROOT / "config" / "metadata.json"
