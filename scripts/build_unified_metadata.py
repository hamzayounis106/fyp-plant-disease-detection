"""Build unified metadata CSV from Plant Village + PlantDoc.

- Reads Plant Village 2 (crop/split/class layout) and PlantDoc 1 (split/class layout).
- Maps PlantDoc class folder names to Plant Village class names via plantdoc_alias.json.
  Aliases with `null` value are kept as new classes (their folder name is used as class_name).
- For PlantDoc, splits are 'train' and 'test' only. A 10% stratified val split is carved
  from PlantDoc train (deterministic via --seed).
- Emits one combined CSV (`unified_metadata.csv`) and one class_map.json (replaces the old).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_plant_village(root: Path) -> pd.DataFrame:
    rows = []
    for crop_dir in sorted([d for d in root.iterdir() if d.is_dir()], key=lambda p: p.name.lower()):
        for split_dir in sorted([d for d in crop_dir.iterdir() if d.is_dir()], key=lambda p: p.name.lower()):
            split = split_dir.name.lower()
            if split not in {"train", "val", "test"}:
                continue
            for class_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()], key=lambda p: p.name.lower()):
                class_name = f"{crop_dir.name}___{class_dir.name}"
                for p in class_dir.rglob("*"):
                    if p.is_file() and p.suffix.lower() in IMG_EXTS:
                        rows.append({
                            "image_path": str(p.resolve()),
                            "crop": crop_dir.name,
                            "disease_name": class_dir.name,
                            "class_name": class_name,
                            "split": split,
                            "source_dataset": "plant_village",
                        })
    return pd.DataFrame(rows)


def collect_plantdoc(root: Path, alias: dict, val_fraction: float, seed: int) -> pd.DataFrame:
    rows = []
    for split_dir in sorted([d for d in root.iterdir() if d.is_dir()], key=lambda p: p.name.lower()):
        split = split_dir.name.lower()
        if split not in {"train", "test"}:
            continue
        for class_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()], key=lambda p: p.name.lower()):
            folder = class_dir.name
            mapped = alias.get(folder)
            class_name = mapped if mapped else f"PlantDoc___{folder}"
            disease_name = mapped.split("___")[1] if mapped and "___" in mapped else folder
            crop = mapped.split("___")[0] if mapped and "___" in mapped else folder
            for p in class_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    rows.append({
                        "image_path": str(p.resolve()),
                        "crop": crop,
                        "disease_name": disease_name,
                        "class_name": class_name,
                        "split": split,
                        "source_dataset": "plantdoc",
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    rng = random.Random(seed)
    train_idx = df.index[df["split"] == "train"].tolist()
    val_idx = []
    for cls, group in df.loc[train_idx].groupby("class_name"):
        idxs = group.index.tolist()
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_fraction))
        val_idx.extend(idxs[:n_val])
    df.loc[val_idx, "split"] = "val"
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--plant-village-root", type=Path, required=True)
    p.add_argument("--plantdoc-root", type=Path, required=True)
    p.add_argument("--alias-json", type=Path, default=Path("artifacts/metadata/plantdoc_alias.json"))
    p.add_argument("--output-csv", type=Path, default=Path("artifacts/metadata/unified_metadata.csv"))
    p.add_argument("--class-map-json", type=Path, default=Path("artifacts/metadata/unified_class_map.json"))
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    with open(args.alias_json, "r", encoding="utf-8") as f:
        alias = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

    pv = collect_plant_village(args.plant_village_root)
    pd_ = collect_plantdoc(args.plantdoc_root, alias, args.val_fraction, args.seed)
    if pv.empty and pd_.empty:
        raise SystemExit("No images collected from either dataset.")
    df = pd.concat([pv, pd_], ignore_index=True)

    classes = sorted(df["class_name"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    df["label_idx"] = df["class_name"].map(class_to_idx)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    with open(args.class_map_json, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=2)

    print(f"Saved metadata: {args.output_csv}")
    print(f"Saved class map: {args.class_map_json}")
    print(f"Total images: {len(df)}")
    print(f"Num classes: {len(classes)}")
    print("Per-source per-split counts:")
    print(df.groupby(["source_dataset", "split"]).size().to_string())


if __name__ == "__main__":
    main()
