from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping


class CSVLogger:
    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, row: Mapping[str, object]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({k: row.get(k, "") for k in self.fieldnames})

