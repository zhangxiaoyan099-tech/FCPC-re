from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text)
    except Exception:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FCPC reconstruction runner")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    parser.add_argument("--dry-run", action="store_true", help="Parse and build only; do not train")
    return parser.parse_args()

