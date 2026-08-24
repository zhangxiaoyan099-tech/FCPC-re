"""Dataset loader registry for public FCPC reference runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    input_channels: int


DATASET_SPECS = {
    "synthetic": DatasetSpec("synthetic", num_classes=10, input_channels=1),
    "uci_har": DatasetSpec("uci_har", num_classes=6, input_channels=9),
    "mnist": DatasetSpec("mnist", num_classes=10, input_channels=1),
    "emnist_byclass": DatasetSpec("emnist_byclass", num_classes=62, input_channels=1),
    "femnist_leaf": DatasetSpec("femnist_leaf", num_classes=62, input_channels=1),
    "cifar10": DatasetSpec("cifar10", num_classes=10, input_channels=3),
    "cifar100": DatasetSpec("cifar100", num_classes=100, input_channels=3),
    "tinyimagenet": DatasetSpec("tinyimagenet", num_classes=200, input_channels=3),
}


class SyntheticVisionDataset:
    """Small deterministic dataset for end-to-end pipeline smoke tests."""

    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        input_channels: int,
        image_size: int,
        seed: int,
    ):
        import torch

        generator = torch.Generator().manual_seed(int(seed))
        self.images = torch.randn(
            int(num_samples),
            int(input_channels),
            int(image_size),
            int(image_size),
            generator=generator,
        )
        self.targets = torch.randint(
            0,
            int(num_classes),
            (int(num_samples),),
            generator=generator,
        )

    def __len__(self):
        return int(self.targets.numel())

    def __getitem__(self, index):
        return self.images[index], int(self.targets[index])


class UCIHARDataset:
    """Raw 9-channel inertial windows from the UCI smartphone HAR dataset."""

    SIGNAL_NAMES = (
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z",
        "total_acc_x",
        "total_acc_y",
        "total_acc_z",
    )

    def __init__(self, root: str | Path, split: str = "train"):
        import torch

        self.root = Path(root)
        self.split = split
        split_dir = self.root / split
        signal_dir = split_dir / "Inertial Signals"
        label_path = split_dir / f"y_{split}.txt"
        subject_path = split_dir / f"subject_{split}.txt"
        required = [
            signal_dir / f"{name}_{split}.txt"
            for name in self.SIGNAL_NAMES
        ] + [label_path, subject_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "UCI HAR files are missing. Extract the official archive so "
                f"that root points to 'UCI HAR Dataset'. Missing: {missing[:3]}"
            )

        channels = [
            np.loadtxt(signal_dir / f"{name}_{split}.txt", dtype=np.float32)
            for name in self.SIGNAL_NAMES
        ]
        self.features = torch.from_numpy(np.stack(channels, axis=1))
        self.targets = torch.from_numpy(
            np.loadtxt(label_path, dtype=np.int64).reshape(-1) - 1
        )
        self.client_ids = np.loadtxt(subject_path, dtype=np.int64).reshape(-1)

    def __len__(self):
        return int(self.targets.numel())

    def __getitem__(self, index):
        return self.features[index], int(self.targets[index])


def get_dataset_spec(name: str) -> DatasetSpec:
    key = name.lower()
    if key not in DATASET_SPECS:
        raise ValueError(f"unsupported dataset: {name}")
    return DATASET_SPECS[key]


class LeafFEMNISTDataset:
    """Minimal LEAF FEMNIST JSON reader.

    Expected directory:
        root/train/*.json
        root/test/*.json

    Each JSON file follows the LEAF schema with `users` and `user_data`.
    """

    def __init__(self, root: str | Path, split: str = "train", transform=None):
        try:
            import torch
            from torch.utils.data import Dataset
        except Exception as exc:  # pragma: no cover
            raise ImportError("PyTorch is required for LeafFEMNISTDataset") from exc

        class _Dataset(Dataset):
            def __init__(self, outer):
                self.outer = outer

            def __len__(self):
                return len(self.outer.targets)

            def __getitem__(self, index):
                x = np.asarray(self.outer.images[index], dtype=np.float32).reshape(28, 28) / 255.0
                tensor = torch.from_numpy(x).unsqueeze(0)
                if self.outer.transform is not None:
                    tensor = self.outer.transform(tensor)
                return tensor, int(self.outer.targets[index])

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.images: list[Any] = []
        self.targets: list[int] = []
        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"LEAF FEMNIST split directory not found: {split_dir}. "
                "Prepare LEAF JSON files under data/femnist_leaf/{train,test}/."
            )
        for json_path in sorted(split_dir.glob("*.json")):
            data = json.loads(json_path.read_text(encoding="utf-8"))
            for user in data.get("users", []):
                user_data = data.get("user_data", {}).get(user, {})
                xs = user_data.get("x", [])
                ys = user_data.get("y", [])
                self.images.extend(xs)
                self.targets.extend(int(y) for y in ys)
        if not self.targets:
            raise ValueError(f"No FEMNIST samples found under {split_dir}")
        self._dataset = _Dataset(self)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._dataset[index]


def load_dataset(name: str, root: str | Path, train: bool = True, download: bool = False, **kwargs):
    key = name.lower()
    root = str(root)
    if key == "synthetic":
        num_samples = int(
            kwargs.get("train_samples", 256)
            if train
            else kwargs.get("test_samples", 64)
        )
        return SyntheticVisionDataset(
            num_samples=num_samples,
            num_classes=int(kwargs.get("num_classes", 10)),
            input_channels=int(kwargs.get("input_channels", 1)),
            image_size=int(kwargs.get("image_size", 28)),
            seed=int(kwargs.get("seed", 42)) + (0 if train else 100_000),
        )
    if key == "uci_har":
        return UCIHARDataset(root=root, split="train" if train else "test")
    if key == "femnist_leaf":
        return LeafFEMNISTDataset(root=root, split="train" if train else "test")
    import torchvision
    from torchvision import transforms

    if key == "mnist":
        transform = transforms.Compose([transforms.ToTensor()])
        return torchvision.datasets.MNIST(root=root, train=train, transform=transform, download=download)
    if key == "cifar10":
        transform = transforms.Compose([transforms.ToTensor()])
        return torchvision.datasets.CIFAR10(root=root, train=train, transform=transform, download=download)
    if key == "cifar100":
        transform = transforms.Compose([transforms.ToTensor()])
        return torchvision.datasets.CIFAR100(root=root, train=train, transform=transform, download=download)
    if key == "emnist_byclass":
        transform = transforms.Compose([transforms.ToTensor()])
        return torchvision.datasets.EMNIST(root=root, split="byclass", train=train, transform=transform, download=download)
    if key == "tinyimagenet":
        split = "train" if train else "val"
        image_root = Path(root) / split
        if not image_root.exists():
            raise FileNotFoundError(
                f"TinyImageNet ImageFolder split not found: {image_root}. "
                "Expected data/tiny-imagenet-200/train and data/tiny-imagenet-200/val."
            )
        transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
        return torchvision.datasets.ImageFolder(str(image_root), transform=transform)
    raise ValueError(f"unsupported dataset: {name}")


def load_torchvision_dataset(name: str, root: str | Path, train: bool = True, download: bool = False):
    return load_dataset(name=name, root=root, train=train, download=download)


def get_targets(dataset) -> list[int]:
    """Extract labels from common torchvision/LEAF datasets."""
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        if hasattr(targets, "tolist"):
            return [int(x) for x in targets.tolist()]
        return [int(x) for x in targets]
    if hasattr(dataset, "labels"):
        labels = getattr(dataset, "labels")
        if hasattr(labels, "tolist"):
            return [int(x) for x in labels.tolist()]
        return [int(x) for x in labels]
    if hasattr(dataset, "samples"):
        return [int(label) for _, label in getattr(dataset, "samples")]
    raise ValueError("Could not extract labels from dataset; provide a dataset with targets/labels/samples.")


def get_client_ids(dataset) -> list[int]:
    """Extract natural device/subject identifiers when a dataset provides them."""
    if not hasattr(dataset, "client_ids"):
        raise ValueError("Dataset does not expose natural client_ids")
    client_ids = getattr(dataset, "client_ids")
    if hasattr(client_ids, "tolist"):
        return [int(x) for x in client_ids.tolist()]
    return [int(x) for x in client_ids]
