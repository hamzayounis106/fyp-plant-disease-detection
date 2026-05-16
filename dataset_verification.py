#!/usr/bin/env python3
"""
Dataset verification utility for plant disease image datasets.

Supports both structures:
1) dataset_root/<crop>/<class>/<images>
2) dataset_root/<crop>/<split>/<class>/<images>

Outputs:
- Tree summary of crops, splits (if present), and classes
- Per-class image counts to surface imbalance
- Corrupted/unreadable image report
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image, UnidentifiedImageError
except ImportError as exc:
    raise SystemExit(
        "Pillow is required. Install it with: pip install pillow"
    ) from exc

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
KNOWN_SPLITS = {"train", "val", "valid", "validation", "test"}


@dataclass(frozen=True)
class ImageRecord:
    file_path: Path
    crop: str
    split: str
    disease_class: str


def looks_like_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def detect_layout(root: Path) -> str:
    """Detect whether dataset is crop/class or crop/split/class layout."""
    crop_dirs = [d for d in root.iterdir() if d.is_dir()]
    if not crop_dirs:
        return "unknown"

    split_hits = 0
    total_crop_checks = 0

    for crop_dir in crop_dirs:
        total_crop_checks += 1
        children = [d.name.lower() for d in crop_dir.iterdir() if d.is_dir()]
        if any(name in KNOWN_SPLITS for name in children):
            split_hits += 1

    if split_hits >= max(1, total_crop_checks // 2):
        return "crop_split_class"
    return "crop_class"


def collect_records(root: Path) -> Tuple[List[ImageRecord], str]:
    layout = detect_layout(root)
    records: List[ImageRecord] = []

    if layout == "crop_split_class":
        for crop_dir in sorted([d for d in root.iterdir() if d.is_dir()], key=lambda p: p.name.lower()):
            for split_dir in sorted([d for d in crop_dir.iterdir() if d.is_dir()], key=lambda p: p.name.lower()):
                split_name = split_dir.name
                for class_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()], key=lambda p: p.name.lower()):
                    for file_path in class_dir.rglob("*"):
                        if looks_like_image(file_path):
                            records.append(
                                ImageRecord(
                                    file_path=file_path,
                                    crop=crop_dir.name,
                                    split=split_name,
                                    disease_class=class_dir.name,
                                )
                            )
    elif layout == "crop_class":
        for crop_dir in sorted([d for d in root.iterdir() if d.is_dir()], key=lambda p: p.name.lower()):
            for class_dir in sorted([d for d in crop_dir.iterdir() if d.is_dir()], key=lambda p: p.name.lower()):
                for file_path in class_dir.rglob("*"):
                    if looks_like_image(file_path):
                        records.append(
                            ImageRecord(
                                file_path=file_path,
                                crop=crop_dir.name,
                                split="all",
                                disease_class=class_dir.name,
                            )
                        )
    else:
        raise ValueError(
            "Could not detect dataset layout. Expected either crop/class/images "
            "or crop/split/class/images."
        )

    return records, layout


def verify_image(file_path: Path) -> Optional[str]:
    """Return None if image is readable, otherwise a short error string."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            img.load()
        return None
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return str(exc)


def print_tree_summary(records: List[ImageRecord], layout: str) -> None:
    tree: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))

    for rec in records:
        tree[rec.crop][rec.split].add(rec.disease_class)

    print("\n=== DATASET TREE SUMMARY ===")
    print(f"Detected layout: {layout}")

    for crop in sorted(tree.keys(), key=str.lower):
        print(f"- {crop}")
        splits = tree[crop]
        for split in sorted(splits.keys(), key=str.lower):
            if split != "all":
                print(f"  - {split}")
                for cls in sorted(splits[split], key=str.lower):
                    print(f"    - {cls}")
            else:
                for cls in sorted(splits[split], key=str.lower):
                    print(f"  - {cls}")


def print_count_table(records: List[ImageRecord]) -> None:
    counts = Counter((r.crop, r.split, r.disease_class) for r in records)

    header = ["Crop", "Split", "Class", "ImageCount"]
    rows = sorted(
        [(crop, split, cls, count) for (crop, split, cls), count in counts.items()],
        key=lambda x: (x[0].lower(), x[1].lower(), x[2].lower()),
    )

    widths = [
        max(len(header[0]), *(len(r[0]) for r in rows)) if rows else len(header[0]),
        max(len(header[1]), *(len(r[1]) for r in rows)) if rows else len(header[1]),
        max(len(header[2]), *(len(r[2]) for r in rows)) if rows else len(header[2]),
        max(len(header[3]), *(len(str(r[3])) for r in rows)) if rows else len(header[3]),
    ]

    print("\n=== CLASS COUNT TABLE ===")
    print(
        f"{header[0]:<{widths[0]}} | {header[1]:<{widths[1]}} | "
        f"{header[2]:<{widths[2]}} | {header[3]:>{widths[3]}}"
    )
    print("-" * (sum(widths) + 9))

    for crop, split, cls, count in rows:
        print(f"{crop:<{widths[0]}} | {split:<{widths[1]}} | {cls:<{widths[2]}} | {count:>{widths[3]}}")

    class_counts = [r[3] for r in rows]
    if class_counts:
        min_count = min(class_counts)
        max_count = max(class_counts)
        imbalance_ratio = (max_count / min_count) if min_count > 0 else float("inf")
        print("\nImbalance stats:")
        print(f"- Min images in a class: {min_count}")
        print(f"- Max images in a class: {max_count}")
        print(f"- Imbalance ratio (max/min): {imbalance_ratio:.2f}")


def print_corruption_report(records: List[ImageRecord], max_preview: int) -> None:
    unreadable = []

    print("\n=== IMAGE INTEGRITY CHECK ===")
    print(f"Checking {len(records)} images. This may take some time...")

    for rec in records:
        err = verify_image(rec.file_path)
        if err is not None:
            unreadable.append((rec.file_path, err))

    if not unreadable:
        print("No corrupted/unreadable image files found.")
        return

    print(f"Found {len(unreadable)} corrupted/unreadable image files.")
    print("Preview:")
    for path, err in unreadable[:max_preview]:
        print(f"- {path} | error: {err}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify dataset structure, class counts, and image integrity.")
    parser.add_argument(
        "dataset_root",
        type=Path,
        nargs="?",
        default=None,
        help="Path to dataset root folder.",
    )
    parser.add_argument(
        "--max-corrupt-preview",
        type=int,
        default=50,
        help="Max number of corrupted file paths to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # If no dataset root is passed, try to auto-detect candidate dataset folders in cwd.
    if args.dataset_root is None:
        cwd = Path.cwd()
        candidates = [d for d in cwd.iterdir() if d.is_dir()]
        detected: List[Path] = []

        for candidate in sorted(candidates, key=lambda p: p.name.lower()):
            try:
                layout = detect_layout(candidate)
                if layout in {"crop_split_class", "crop_class"}:
                    detected.append(candidate)
            except OSError:
                continue

        if not detected:
            raise SystemExit(
                "No dataset_root argument provided and no valid dataset folders found in current directory.\n"
                "Usage: py dataset_verification.py <dataset_root>"
            )

        print("No dataset_root provided. Auto-detected dataset folders:")
        for d in detected:
            print(f"- {d}")

        for root in detected:
            print("\n" + "=" * 90)
            print(f"Analyzing: {root}")
            records, layout = collect_records(root)
            if not records:
                print("No images found in this dataset folder. Skipping.")
                continue
            print_tree_summary(records, layout)
            print_count_table(records)
            print_corruption_report(records, max_preview=max(0, args.max_corrupt_preview))
        return

    root: Path = args.dataset_root

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Dataset root does not exist or is not a directory: {root}")

    records, layout = collect_records(root)
    if not records:
        raise SystemExit("No images found. Check dataset path and folder structure.")

    print_tree_summary(records, layout)
    print_count_table(records)
    print_corruption_report(records, max_preview=max(0, args.max_corrupt_preview))


if __name__ == "__main__":
    main()
