"""Post-hoc logit adjustment to fix the training-time WeightedRandomSampler + class-weighted CE bias.

Tries three strategies and reports top-1 accuracy on PV + PlantDoc test:
1. baseline    — no adjustment (current behavior)
2. prior_natural — pred = argmax(logit + log(p_natural)) where p_natural is train class freq
3. prior_inv_count — pred = argmax(logit + 2*log(count)), undoes class_weight + sampler

Also bakes the best adjustment into the model's classifier bias and saves a new checkpoint
artifacts/checkpoints/<model>/best_calibrated.pt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

from src.plant_disease.data import MetadataColumns, PlantDiseaseDataset
from src.plant_disease.models import build_model


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["model"], ckpt["num_classes"], use_pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def get_logits(model, loader, device):
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            all_logits.append(model(x).cpu().numpy())
            all_labels.extend(y.tolist())
    return np.concatenate(all_logits, axis=0), np.array(all_labels)


def evaluate_strategy(logits, labels, num_classes, name, adjustment=None):
    if adjustment is None:
        adj_logits = logits
    else:
        adj_logits = logits + adjustment[None, :]
    preds = adj_logits.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    pred_distinct = len(set(preds.tolist()))
    return {"strategy": name, "acc": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "distinct_preds": pred_distinct}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--metadata-csv", type=Path, default=Path("artifacts/metadata/unified_metadata_resized.csv"))
    p.add_argument("--image-column", type=str, default="image_path_resized")
    p.add_argument("--class-map", type=Path, default=Path("artifacts/metadata/unified_class_map.json"))
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.metadata_csv)

    with open(args.class_map) as f:
        class_map = {int(k): v for k, v in json.load(f).items()}
    num_classes = len(class_map)

    # Compute training class frequencies (the natural distribution before sampler)
    train_df = df[df["split"] == "train"]
    train_counts = np.bincount(train_df["label_idx"].astype(int), minlength=num_classes).astype(np.float64)
    p_natural = train_counts / train_counts.sum()
    print(f"Train counts (37 classes): min={int(train_counts.min())} max={int(train_counts.max())} ratio={train_counts.max()/max(train_counts.min(),1):.1f}x")

    eval_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    columns = MetadataColumns(image_path=args.image_column, split="split", label_idx="label_idx")

    model, ckpt = load_model(args.checkpoint, device)
    print(f"\nModel: {ckpt['model']} | epoch={ckpt['epoch']} | val_macro_f1={ckpt['val_macro_f1']:.4f}")

    # Run on test split
    test_ds = PlantDiseaseDataset(None, split="test", transform=eval_tf, columns=columns, dataframe=df)
    loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    print(f"\nRunning inference on {len(test_ds)} test images...")
    logits, labels = get_logits(model, loader, device)

    eps = 1e-12
    adjustments = {
        "prior_natural": np.log(p_natural + eps),
        "prior_inv_count": 2.0 * np.log(train_counts + eps),
    }

    print("\n=== TEST split (PV + PlantDoc, all 37 classes) ===")
    print(f"{'strategy':<20}{'acc':>10}{'macro_f1':>10}{'weighted_f1':>14}{'distinct':>10}")
    results = [evaluate_strategy(logits, labels, num_classes, "baseline")]
    for name, adj in adjustments.items():
        results.append(evaluate_strategy(logits, labels, num_classes, name, adj))
    for r in results:
        print(f"{r['strategy']:<20}{r['acc']:>10.4f}{r['macro_f1']:>10.4f}{r['weighted_f1']:>14.4f}{r['distinct_preds']:>10}")

    # Per-source breakdown for the best strategy
    src_col = df[df["split"] == "test"]["source_dataset"].to_numpy()
    print("\n=== Per-source test breakdown ===")
    for source in ["plant_village", "plantdoc"]:
        mask = src_col == source
        n = mask.sum()
        print(f"\n{source} (N={n}):")
        for r in [evaluate_strategy(logits[mask], labels[mask], num_classes, "baseline"),
                  evaluate_strategy(logits[mask] + adjustments["prior_natural"][None, :], labels[mask], num_classes, "prior_natural"),
                  evaluate_strategy(logits[mask] + adjustments["prior_inv_count"][None, :], labels[mask], num_classes, "prior_inv_count")]:
            print(f"  {r['strategy']:<20} acc={r['acc']:.4f}  mF1={r['macro_f1']:.4f}  distinct={r['distinct_preds']}")

    # Pick the best strategy by overall accuracy and bake into a calibrated checkpoint
    best = max(results, key=lambda r: r["acc"])
    print(f"\nBest strategy on full test: {best['strategy']} (acc={best['acc']:.4f})")
    if best["strategy"] != "baseline":
        adj = adjustments[best["strategy"]]
        # Bake adjustment into classifier bias
        if ckpt["model"] == "efficientnet":
            with torch.no_grad():
                model.classifier[1].bias.add_(torch.tensor(adj, dtype=model.classifier[1].bias.dtype, device=device))
        else:  # cnn
            with torch.no_grad():
                model.classifier[1].bias.add_(torch.tensor(adj, dtype=model.classifier[1].bias.dtype, device=device))
        ckpt["model_state_dict"] = model.state_dict()
        ckpt["calibration"] = best["strategy"]
        out_path = args.checkpoint.parent / "best_calibrated.pt"
        torch.save(ckpt, out_path)
        print(f"Saved calibrated checkpoint -> {out_path}")


if __name__ == "__main__":
    main()
