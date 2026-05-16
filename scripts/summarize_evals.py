"""Print a comparison table from all eval_*.json under artifacts/checkpoints/."""
import json
import glob
import os

rows = []
for path in sorted(glob.glob("artifacts/checkpoints/*/eval_*.json")):
    with open(path) as f:
        d = json.load(f)
    norm = path.replace(os.sep, "/")
    parts = norm.split("/")
    model = parts[-2]
    name = parts[-1].replace("eval_", "").replace(".json", "")
    m = d["metrics"]
    info = d["model_info"]
    rows.append({
        "model": model,
        "eval": name,
        "n": d["n_samples"],
        "acc": m["accuracy"],
        "mF1": m["macro_f1"],
        "wF1": m["weighted_f1"],
        "top5": m["top5_accuracy"],
        "lat_ms": info["mean_latency_ms_per_image"],
        "params": info["parameters"],
        "ckpt_mb": info["checkpoint_size_mb"],
    })

print(f"{'model':<14}{'eval':<32}{'N':>5}{'acc':>9}{'mF1':>9}{'wF1':>9}{'top5':>9}{'lat_ms':>8}")
print("-" * 95)
for r in rows:
    print(f"{r['model']:<14}{r['eval']:<32}{r['n']:>5}{r['acc']:>9.4f}{r['mF1']:>9.4f}{r['wF1']:>9.4f}{r['top5']:>9.4f}{r['lat_ms']:>8.2f}")

print("\nModel info:")
seen = set()
for r in rows:
    if r["model"] in seen:
        continue
    seen.add(r["model"])
    print(f"  {r['model']:<14} params={r['params']:>11,}  ckpt={r['ckpt_mb']:.1f} MB")
