from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plant_disease.segmentation import segment_leaf


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-segment leaf images and cache processed outputs.")
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/processed"))
    parser.add_argument(
        "--output-metadata-csv",
        type=Path,
        default=Path("artifacts/metadata/plant_village_metadata_segmented.csv"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--force-grabcut", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_csv)
    if args.max_images > 0:
        df = df.head(args.max_images).copy()
    out_paths = []
    methods = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Segmenting"):
        img_path = Path(row["image_path"])
        source = row.get("source_dataset", "plant_village")
        # Sanitize class name for filesystem (PlantDoc fallback names contain only safe chars)
        safe_class = str(row["class_name"]).replace("/", "_")
        rel = Path(source) / row["split"] / safe_class / (img_path.stem + ".png")
        out_path = args.output_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        method = "cached"
        if args.overwrite or not out_path.exists():
            try:
                image = Image.open(img_path).convert("RGB")
                segmented, method = segment_leaf(image, use_rembg=not args.force_grabcut)
                segmented.save(out_path)
            except Exception as e:
                # Fallback: copy raw image as PNG so the pipeline still has a file at processed_path
                method = f"failed_{type(e).__name__}"
                try:
                    Image.open(img_path).convert("RGB").save(out_path)
                except Exception:
                    pass

        out_paths.append(str(out_path.resolve()))
        methods.append(method)

    df["processed_path"] = out_paths
    df["segmentation_method"] = methods

    args.output_metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_metadata_csv, index=False)

    print(f"Saved segmented metadata: {args.output_metadata_csv}")
    print(df["segmentation_method"].value_counts().to_string())


if __name__ == "__main__":
    main()
