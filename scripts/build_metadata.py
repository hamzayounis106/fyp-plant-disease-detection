from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_plant_village(dataset_root: Path) -> pd.DataFrame:
    rows = []
    for crop_dir in sorted([d for d in dataset_root.iterdir() if d.is_dir()], key=lambda p: p.name.lower()):
        for split_dir in sorted([d for d in crop_dir.iterdir() if d.is_dir()], key=lambda p: p.name.lower()):
            split = split_dir.name.lower()
            if split not in {"train", "val", "test"}:
                continue
            for class_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()], key=lambda p: p.name.lower()):
                class_name = f"{crop_dir.name}___{class_dir.name}"
                for img_path in class_dir.rglob("*"):
                    if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
                        rows.append(
                            {
                                "image_path": str(img_path.resolve()),
                                "crop": crop_dir.name,
                                "disease_name": class_dir.name,
                                "class_name": class_name,
                                "split": split,
                                "source_dataset": "plant_village",
                            }
                        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build metadata CSV for Plant Village v1.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("artifacts/metadata/plant_village_metadata.csv"))
    parser.add_argument("--class-map-json", type=Path, default=Path("artifacts/metadata/class_map.json"))
    args = parser.parse_args()

    df = collect_plant_village(args.dataset_root)
    if df.empty:
        raise SystemExit("No images found under provided dataset root.")

    classes = sorted(df["class_name"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    df["label_idx"] = df["class_name"].map(class_to_idx)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    args.class_map_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.class_map_json, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=2)

    print(f"Saved metadata: {args.output_csv}")
    print(f"Saved class map: {args.class_map_json}")
    print(df.groupby("split").size().to_string())
    print(f"Num classes: {len(classes)}")


if __name__ == "__main__":
    main()
