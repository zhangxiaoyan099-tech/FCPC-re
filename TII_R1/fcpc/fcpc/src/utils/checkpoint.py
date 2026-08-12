from __future__ import annotations

from pathlib import Path


def save_checkpoint(state: object, path: str | Path) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path) -> object:
    import torch

    return torch.load(Path(path), map_location="cpu")

