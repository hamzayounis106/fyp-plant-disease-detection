"""Surgically merge two output classes in a trained model's classifier head.

When two classes are duplicates (e.g., "Common Rust" and "Common Rust_" from a
metadata bug), we don't need to retrain. We can collapse the heads:
  new_logit_target = logsumexp(old_logit_src, old_logit_target)
which is exact for softmax probabilities. Practically, since the rest of the
network is unchanged, we replace the classifier rows so:
  new_weight[target] = old_weight[target] + old_weight[src]   (approx — combines features)
  new_bias[target]   = log(exp(old_bias[target]) + exp(old_bias[src]))
  drop row[src]

This is a close approximation to true logit merging when the merged classes
have similar feature embeddings (which they do — same disease).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from src.plant_disease.models import build_model


def merge_rows(weight: torch.Tensor, bias: torch.Tensor, src: int, target: int):
    new_w = torch.cat([weight[:src], weight[src + 1:]], dim=0).clone()
    new_b = torch.cat([bias[:src], bias[src + 1:]], dim=0).clone()
    # If src < target, target shifts down by 1 in the new tensors.
    new_target = target if target < src else target - 1
    # Combine: bias via logsumexp (exact for softmax merge); weight by sum.
    new_b[new_target] = torch.logsumexp(torch.stack([bias[target], bias[src]]), dim=0)
    new_w[new_target] = weight[target] + weight[src]
    return new_w, new_b, new_target


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-checkpoint", type=Path, required=True)
    p.add_argument("--src-idx", type=int, required=True, help="Old index to drop (its mass is merged into target)")
    p.add_argument("--target-idx", type=int, required=True, help="Old index that absorbs src")
    args = p.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    old_classes = ckpt["num_classes"]
    model = build_model(ckpt["model"], old_classes, use_pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Both architectures use classifier[1] = nn.Linear as the final layer.
    head = model.classifier[1] if hasattr(model.classifier, "__getitem__") else None
    if head is None or not isinstance(head, torch.nn.Linear):
        raise SystemExit("Could not locate final Linear layer.")

    new_w, new_b, new_target = merge_rows(head.weight.data, head.bias.data, args.src_idx, args.target_idx)
    new_classes = old_classes - 1
    in_features = head.in_features
    new_head = torch.nn.Linear(in_features, new_classes)
    new_head.weight.data = new_w
    new_head.bias.data = new_b

    # Build a fresh model with new num_classes and copy backbone + new head
    new_model = build_model(ckpt["model"], new_classes, use_pretrained=False)
    # Copy backbone state by loading old state dict into a temporary model and replacing head
    tmp_state = model.state_dict()
    # Strip old head keys and replace with new head keys
    new_state = new_model.state_dict()
    for k in list(new_state.keys()):
        if k.startswith("classifier.1."):
            continue
        new_state[k] = tmp_state[k]
    new_state["classifier.1.weight"] = new_head.weight.data
    new_state["classifier.1.bias"] = new_head.bias.data
    new_model.load_state_dict(new_state)

    out_ckpt = {
        **ckpt,
        "model_state_dict": new_model.state_dict(),
        "num_classes": new_classes,
        "merge_op": {"src": args.src_idx, "target": args.target_idx, "old_classes": old_classes, "new_classes": new_classes},
    }
    args.out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, args.out_checkpoint)
    print(f"Merged class {args.src_idx} -> {args.target_idx} | {old_classes} -> {new_classes} classes")
    print(f"New head bias[{new_target}] = {new_head.bias.data[new_target].item():+.4f}")
    print(f"Saved: {args.out_checkpoint}")


if __name__ == "__main__":
    main()
