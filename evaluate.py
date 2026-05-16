from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from src.plant_disease.data import MetadataColumns, PlantDiseaseDataset
from src.plant_disease.models import build_model, count_parameters


def topk_accuracy(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float((topk == labels[:, None]).any(axis=1).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint and emit JSON metrics.")
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, default=Path("artifacts/metadata/class_map.json"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--image-column", type=str, default="processed_path")
    parser.add_argument("--source-filter", type=str, default="",
                        help="If set, only include rows where source_dataset == this value")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.metadata_csv)
    if args.image_column not in df.columns:
        args.image_column = "image_path"
    if args.source_filter:
        df = df[df["source_dataset"] == args.source_filter].reset_index(drop=True)

    num_classes = int(df["label_idx"].max()) + 1
    eval_tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    columns = MetadataColumns(image_path=args.image_column, split="split", label_idx="label_idx")
    ds = PlantDiseaseDataset(None, split=args.split, transform=eval_tf, columns=columns, dataframe=df)
    if len(ds) == 0:
        raise SystemExit(f"No rows for split={args.split} after filters.")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = ckpt.get("model", "efficientnet")
    ckpt_classes = ckpt.get("num_classes", num_classes)
    model = build_model(model_name, ckpt_classes, use_pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    class_map = {}
    if args.class_map.exists():
        with open(args.class_map, "r", encoding="utf-8") as f:
            class_map = {int(k): v for k, v in json.load(f).items()}

    all_probs, all_labels = [], []
    n_seen, total_time = 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            t0 = time.time()
            logits = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.time() - t0
            n_seen += x.size(0)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(y.tolist())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.array(all_labels)
    preds = probs.argmax(axis=1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    macro_p, macro_r, _, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    per_p, per_r, per_f, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(ckpt_classes)), zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=list(range(ckpt_classes)))

    top3 = topk_accuracy(probs, labels, k=min(3, ckpt_classes))
    top5 = topk_accuracy(probs, labels, k=min(5, ckpt_classes))

    ckpt_size_mb = args.checkpoint.stat().st_size / (1024 * 1024)
    n_params = count_parameters(model)
    mean_latency_ms = (total_time / max(n_seen, 1)) * 1000

    out = {
        "checkpoint": str(args.checkpoint),
        "model": model_name,
        "split": args.split,
        "image_column": args.image_column,
        "source_filter": args.source_filter,
        "n_samples": int(n_seen),
        "device": str(device),
        "num_classes": ckpt_classes,
        "metrics": {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "top3_accuracy": top3,
            "top5_accuracy": top5,
        },
        "per_class": [
            {
                "idx": i,
                "name": class_map.get(i, str(i)),
                "precision": float(per_p[i]),
                "recall": float(per_r[i]),
                "f1": float(per_f[i]),
                "support": int(support[i]),
            }
            for i in range(ckpt_classes)
        ],
        "confusion_matrix": cm.tolist(),
        "model_info": {
            "parameters": n_params,
            "checkpoint_size_mb": ckpt_size_mb,
            "mean_latency_ms_per_image": mean_latency_ms,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"split={args.split} model={model_name} acc={acc:.4f} macro_f1={macro_f1:.4f} "
          f"weighted_f1={weighted_f1:.4f} top5={top5:.4f} latency={mean_latency_ms:.2f}ms/img")
    print(classification_report(labels, preds, digits=4, zero_division=0))
    print(f"Saved metrics to {args.output_json}")


if __name__ == "__main__":
    main()
