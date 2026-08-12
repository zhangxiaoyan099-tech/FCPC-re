from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rearrange TinyImageNet validation images into ImageFolder layout."
    )
    parser.add_argument("--root", default="data/tiny-imagenet-200", help="TinyImageNet root directory")
    parser.add_argument(
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help="copy keeps the original val/images directory; move modifies it",
    )
    return parser.parse_args()


def read_annotations(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            image_name, class_id = parts[0], parts[1]
            mapping[image_name] = class_id
    return mapping


def preprocess(root: Path, mode: str = "copy") -> tuple[int, int]:
    val_dir = root / "val"
    image_dir = val_dir / "images"
    annotations = val_dir / "val_annotations.txt"
    if not root.exists():
        raise FileNotFoundError(f"TinyImageNet root not found: {root}")
    if not annotations.exists():
        raise FileNotFoundError(f"TinyImageNet annotation file not found: {annotations}")
    if not image_dir.exists():
        existing_class_dirs = [p for p in val_dir.iterdir() if p.is_dir()]
        if existing_class_dirs:
            return 0, len(existing_class_dirs)
        raise FileNotFoundError(f"TinyImageNet validation image directory not found: {image_dir}")

    mapping = read_annotations(annotations)
    processed = 0
    classes = set()
    for image_name, class_id in mapping.items():
        src = image_dir / image_name
        if not src.exists():
            continue
        dst_dir = val_dir / class_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / image_name
        if dst.exists():
            classes.add(class_id)
            continue
        if mode == "copy":
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))
        processed += 1
        classes.add(class_id)
    return processed, len(classes)


def main() -> None:
    args = parse_args()
    processed, n_classes = preprocess(Path(args.root), mode=args.mode)
    print(f"TinyImageNet validation preprocessing complete: processed={processed}, classes={n_classes}")


if __name__ == "__main__":
    main()

