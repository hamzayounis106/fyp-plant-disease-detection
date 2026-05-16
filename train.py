from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from src.plant_disease.data import MetadataColumns, PlantDiseaseDataset
from src.plant_disease.models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int):
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def evaluate(model, loader, criterion, device, use_channels_last: bool = False):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if use_channels_last:
                x = x.to(memory_format=torch.channels_last)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train plant disease classifier.")
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--image-column", type=str, default="processed_path")
    parser.add_argument("--model", type=str, default="efficientnet", choices=["efficientnet", "cnn"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--use-pretrained", action="store_true")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true", help="Mixed-precision training (recommended on RTX GPUs)")
    parser.add_argument("--channels-last", action="store_true", help="Use channels_last memory format (faster on Ampere+)")
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from <output-dir>/last.pt if it exists (loads model+optimizer+scheduler+scaler+epoch).")
    parser.add_argument("--fresh-schedule", action="store_true",
                        help="When resuming, keep model+optimizer state but rebuild LR scheduler with new --epochs as T_max. "
                             "Use for adding extra fine-tune epochs after a completed run.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model} | AMP: {args.amp} | channels_last: {args.channels_last}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | cuda {torch.version.cuda}")
        torch.backends.cudnn.benchmark = True

    df = pd.read_csv(args.metadata_csv)
    if args.image_column not in df.columns:
        args.image_column = "image_path"

    if args.max_images > 0:
        parts = []
        for _, g in df.groupby("split"):
            n = max(1, int(args.max_images * len(g) / len(df)))
            parts.append(g.sample(min(len(g), n), random_state=args.seed))
        df = pd.concat(parts).reset_index(drop=True)
        print(f"Smoke test: using {len(df)} images")

    num_classes = int(df["label_idx"].nunique())
    print(f"Classes: {num_classes}")

    train_tf, eval_tf = build_transforms(args.image_size)
    columns = MetadataColumns(image_path=args.image_column, split="split", label_idx="label_idx")

    csv_arg = args.metadata_csv if args.max_images == 0 else None
    df_arg = df if args.max_images > 0 else None
    train_ds = PlantDiseaseDataset(csv_arg, split="train", transform=train_tf, columns=columns, dataframe=df_arg)
    val_ds = PlantDiseaseDataset(csv_arg, split="val", transform=eval_tf, columns=columns, dataframe=df_arg)
    test_ds = PlantDiseaseDataset(csv_arg, split="test", transform=eval_tf, columns=columns, dataframe=df_arg)

    train_labels = train_ds.df["label_idx"].to_numpy()
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[train_labels]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = build_model(args.model, num_classes, args.use_pretrained)
    model.to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device), label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_f1 = -1.0
    start_epoch = 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    last_ckpt_path = args.output_dir / "last.pt"
    if args.resume and last_ckpt_path.exists():
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if args.fresh_schedule:
            # Rebuild scheduler over (epochs - start_epoch + 1) for an extra fine-tune phase.
            for g in optimizer.param_groups:
                g["lr"] = args.lr
                g["initial_lr"] = args.lr
            remaining = max(1, args.epochs - int(ckpt.get("epoch", 0)))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)
        elif "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_f1 = float(ckpt.get("best_f1", ckpt.get("val_macro_f1", -1.0)))
        sched_msg = f"fresh schedule (cosine over {args.epochs - start_epoch + 1} epochs at lr={args.lr})" \
            if args.fresh_schedule else "schedule restored"
        print(f"Resumed from epoch {ckpt.get('epoch', '?')} -> starting at epoch {start_epoch} | "
              f"best_f1_so_far={best_f1:.4f} | {sched_msg}")

    if start_epoch > args.epochs:
        print(f"Already completed {args.epochs} epochs (resumed at {start_epoch}). Skipping training loop.")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if args.channels_last:
                x = x.to(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.size(0)

        scheduler.step()
        train_loss /= max(len(train_loader.dataset), 1)
        val_loss, val_f1 = evaluate(model, val_loader, criterion, device, args.channels_last)
        epoch_time = time.time() - t0

        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f} time={epoch_time:.1f}s"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if args.amp and device.type == "cuda" else None,
            "val_macro_f1": val_f1,
            "best_f1": max(best_f1, val_f1),
            "image_size": args.image_size,
            "num_classes": num_classes,
            "image_column": args.image_column,
            "model": args.model,
        }
        torch.save(ckpt, args.output_dir / "last.pt")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(ckpt, args.output_dir / "best.pt")

    test_loss, test_f1 = evaluate(model, test_loader, criterion, device, args.channels_last)
    print(f"FINAL test_loss={test_loss:.4f} test_macro_f1={test_f1:.4f}")

    with open(args.output_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_macro_f1": best_f1,
                "final_test_macro_f1": test_f1,
                "num_classes": num_classes,
                "model": args.model,
                "epochs": args.epochs,
                "image_size": args.image_size,
                "amp": args.amp,
            },
            f,
            indent=2,
        )

    print(f"Training complete. Best val macro F1: {best_f1:.4f} | test macro F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()
