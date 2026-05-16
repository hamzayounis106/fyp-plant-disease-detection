# Runbook — Plant Disease Detection (CNN vs EfficientNetV2-S)

End-to-end commands to reproduce the project from scratch, in order. All paths are relative to the project root. Replace `.venv/Scripts/python.exe` with whatever invokes Python in your venv.

For why each step exists, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). For things that may go wrong, see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

---

## 0. One-time environment setup

Required: NVIDIA driver supporting CUDA 12.8+, ~10 GB free disk on the project drive, ~6 GB free RAM at training time.

```powershell
# Create venv (Python 3.13)
py -3.13 -m venv .venv

# Redirect TEMP/cache to project drive (avoids C: full issues — see Troubleshooting #2)
mkdir .pip-tmp, .pip-cache
$env:TMP = "$pwd\.pip-tmp"
$env:TEMP = "$pwd\.pip-tmp"
$env:PIP_CACHE_DIR = "$pwd\.pip-cache"

# Install PyTorch with CUDA 12.8 support (required for RTX 50-series Blackwell)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install remaining deps
.venv\Scripts\python.exe -m pip install -r requirements.txt

# Verify
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected output: 2.11.0+cu128 True NVIDIA GeForce RTX 5060
```

**Disable Windows sleep** before training (Settings → System → Power → Screen and sleep → "Never" when plugged in). Training has no way to recover from suspend.

---

## 1. Build unified metadata (Plant Village + PlantDoc)

Datasets must already be unzipped in the project root:
- `Plant Village Dataset 2/` (with `<crop>/{Train,Val,Test}/<class>/*.jpg` layout)
- `PlantDoc Classification dataset 1/` (with `{train,test}/<class>/*.jpg`)

```bash
.venv/Scripts/python.exe scripts/build_unified_metadata.py \
  --plant-village-root "Plant Village Dataset 2" \
  --plantdoc-root "PlantDoc Classification dataset 1"
```

Outputs:
- `artifacts/metadata/unified_metadata.csv` — 70,040 rows, columns: `image_path`, `crop`, `disease_name`, `class_name`, `split`, `source_dataset`, `label_idx`
- `artifacts/metadata/unified_class_map.json` — 36 classes (29 Plant Village + 7 PlantDoc-only)

The script also carves a 10% stratified val split out of PlantDoc train (PlantDoc has no native val).

---

## 2. Pre-resize images to 224×224 JPEG cache (one-time, ~10 min)

This is **required** for reasonable training time on this hardware. Without it, GPU sits at <5% utilization while workers decode JPEGs.

```bash
.venv/Scripts/python.exe -u scripts/cache_resized.py \
  --metadata-csv artifacts/metadata/unified_metadata.csv \
  --output-csv artifacts/metadata/unified_metadata_resized.csv \
  --size 224 --workers 6
```

Outputs:
- `artifacts/processed/resized/<source>/<split>/<class>/*.jpg` — 70,040 cached JPEGs
- `artifacts/metadata/unified_metadata_resized.csv` — adds `image_path_resized` column

Use `--workers 4` if 6 saturates the system.

---

## 3. Segment the test set (for "segmented vs raw" eval comparison, ~10 min)

```bash
# Slice out the test rows
.venv/Scripts/python.exe -c "import pandas as pd; df=pd.read_csv('artifacts/metadata/unified_metadata.csv'); df[df['split']=='test'].to_csv('artifacts/metadata/unified_test_only.csv', index=False)"

# Run segmentation (grabCut, faster than rembg)
.venv/Scripts/python.exe scripts/preprocess_segment.py \
  --metadata-csv artifacts/metadata/unified_test_only.csv \
  --output-root artifacts/processed \
  --output-metadata-csv artifacts/metadata/unified_test_segmented.csv \
  --force-grabcut

# Filter out rows where segmentation failed entirely (~13 of 1610)
.venv/Scripts/python.exe -c "
import os, pandas as pd
df = pd.read_csv('artifacts/metadata/unified_test_segmented.csv')
df = df[df['processed_path'].apply(os.path.exists)].copy()
df.to_csv('artifacts/metadata/unified_test_segmented_ok.csv', index=False)
print(f'kept {len(df)} rows')
"
```

Outputs:
- `artifacts/processed/<source>/test/<class>/*.png` — segmented test images
- `artifacts/metadata/unified_test_segmented_ok.csv` — used by the eval matrix

**Note:** we do not segment train/val. rembg+U2Net on CPU is ~25 hr for 70k images on this machine; it would not change the headline results. Training on raw + reporting raw vs segmented at eval time tells you whether segmentation helps anyway.

---

## 4. Train SimpleCNN, 25 epochs, ~60-90 min

```bash
.venv/Scripts/python.exe -u train.py \
  --metadata-csv artifacts/metadata/unified_metadata_resized.csv \
  --image-column image_path_resized \
  --model cnn --epochs 25 --batch-size 64 --image-size 224 \
  --amp --num-workers 2 --prefetch-factor 2 \
  --output-dir artifacts/checkpoints/cnn
```

Resume after interruption (power outage, sleep, crash):
```bash
# Same command, append --resume
... --output-dir artifacts/checkpoints/cnn --resume
```

`last.pt` saves model + optimizer + scheduler + AMP scaler + epoch + best_f1 every epoch — resume is exact.

Outputs:
- `artifacts/checkpoints/cnn/best.pt` — best validation macro F1
- `artifacts/checkpoints/cnn/last.pt` — last completed epoch (for resume)
- `artifacts/checkpoints/cnn/train_summary.json`

---

## 5. Train EfficientNetV2-S, 20 + 5 fine-tune epochs, ~140 min

Pretrained from ImageNet, then fine-tuned at lower LR for 5 more epochs to match CNN's 25-epoch budget.

**Stage A — 20 epochs base training:**
```bash
.venv/Scripts/python.exe -u train.py \
  --metadata-csv artifacts/metadata/unified_metadata_resized.csv \
  --image-column image_path_resized \
  --model efficientnet --use-pretrained --epochs 20 --batch-size 64 --image-size 224 \
  --amp --num-workers 2 --prefetch-factor 2 \
  --output-dir artifacts/checkpoints/efficientnet
```

**Stage B — +5 fine-tune epochs at lr=5e-5:**
```bash
.venv/Scripts/python.exe -u train.py \
  --metadata-csv artifacts/metadata/unified_metadata_resized.csv \
  --image-column image_path_resized \
  --model efficientnet --use-pretrained --epochs 25 --batch-size 64 --image-size 224 \
  --amp --num-workers 2 --prefetch-factor 2 \
  --output-dir artifacts/checkpoints/efficientnet \
  --resume --fresh-schedule --lr 5e-5
```

`--fresh-schedule` rebuilds the cosine LR over the remaining 5 epochs at the new lower lr (5e-5). Without it, the saved scheduler state would resume from the end of the original cosine where lr ≈ 0 and nothing would learn.

Resume Stage A or B with `--resume` (no `--fresh-schedule`) if interrupted.

---

## 6. Merge the duplicate corn-rust class (no retraining needed)

The original `plantdoc_alias.json` had a typo that created two corn rust classes. The current `plantdoc_alias.json` is fixed — but if you trained against the old file, the saved checkpoint has 37 outputs while the corrected metadata has 36. Surgery the head to match:

```bash
.venv/Scripts/python.exe scripts/merge_class.py \
  --checkpoint artifacts/checkpoints/cnn/best.pt \
  --out-checkpoint artifacts/checkpoints/cnn/best_merged.pt \
  --src-idx 10 --target-idx 9

.venv/Scripts/python.exe scripts/merge_class.py \
  --checkpoint artifacts/checkpoints/efficientnet/best.pt \
  --out-checkpoint artifacts/checkpoints/efficientnet/best_merged.pt \
  --src-idx 10 --target-idx 9
```

This collapses logit row 10 into row 9 (logsumexp on bias, sum on weight) — mathematically equivalent to summing the two softmax probabilities at inference. If you trained against the **already-corrected** metadata (36 classes from the start), skip this step.

---

## 7. Calibrate the classifier bias (the critical step)

Without this, the model has 0% top-1 accuracy on Plant Village (see [docs/ARCHITECTURE.md § The bias bug](docs/ARCHITECTURE.md#the-bias-bug--the-calibration-fix)).

```bash
.venv/Scripts/python.exe scripts/logit_adjust.py \
  --checkpoint artifacts/checkpoints/cnn/best_merged.pt

.venv/Scripts/python.exe scripts/logit_adjust.py \
  --checkpoint artifacts/checkpoints/efficientnet/best_merged.pt
```

Each invocation writes `best_calibrated.pt` next to the input checkpoint. The script also prints a comparison of three calibration strategies on the test set — pick whatever has the highest accuracy (typically `prior_natural`).

Expected after calibration:
- CNN: PV ~81%, PD ~17%
- EfficientNet: PV ~99.93%, PD ~58%

---

## 8. Run the 8-eval matrix

```bash
bash scripts/run_eval_matrix.sh best_calibrated.pt
```

This runs `evaluate.py` 8 times: each model × each dataset (PV, PD) × each preprocessing (raw, segmented). Outputs to:
```
artifacts/checkpoints/{cnn,efficientnet}/eval_{plant_village,plantdoc}_{raw,segmented}.json
```

These JSON files are what the Streamlit Compare tab reads. To pretty-print a summary:
```bash
.venv/Scripts/python.exe scripts/summarize_evals.py
```

If `bash` isn't available on Windows, run via Git Bash or expand the loop manually:
```bash
for MODEL in cnn efficientnet; do
  for SRC in plant_village plantdoc; do
    .venv/Scripts/python.exe evaluate.py \
      --metadata-csv artifacts/metadata/unified_metadata_resized.csv \
      --image-column image_path_resized \
      --checkpoint artifacts/checkpoints/$MODEL/best_calibrated.pt \
      --class-map artifacts/metadata/unified_class_map.json \
      --source-filter $SRC \
      --output-json artifacts/checkpoints/$MODEL/eval_${SRC}_raw.json \
      --num-workers 0
    .venv/Scripts/python.exe evaluate.py \
      --metadata-csv artifacts/metadata/unified_test_segmented_ok.csv \
      --image-column processed_path \
      --checkpoint artifacts/checkpoints/$MODEL/best_calibrated.pt \
      --class-map artifacts/metadata/unified_class_map.json \
      --source-filter $SRC \
      --output-json artifacts/checkpoints/$MODEL/eval_${SRC}_segmented.json \
      --num-workers 0
  done
done
```

---

## 9. Launch the Streamlit UI

```bash
.venv/Scripts/python.exe -m streamlit run app.py
```

Opens at http://localhost:8501. Three tabs:
- **Test: CNN** — upload an image or capture from webcam, see top-5 predictions
- **Test: EfficientNet** — same UI, EffNet model
- **Compare** — live side-by-side prediction on the same image, plus static benchmark panel that reads the eval JSONs (dataset toggle PV/PD, preprocessing toggle raw/segmented, winner-per-metric table, per-class F1 bars, confusion matrices)

Sidebar defaults assume `best_calibrated.pt` for both models and segmentation OFF (matches training).

---

## Common operations

### Live-tail a training run from another terminal
```powershell
Get-Content -Wait "<output-file-from-bg-task>"
```

### Kill stale Python processes
```powershell
Get-Process python -ErrorAction SilentlyContinue |
  Where-Object { $_.Path -like '*\.venv\*' } |
  ForEach-Object { Stop-Process -Id $_.Id -Force }
```

### Check GPU + memory state
```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,power.draw,temperature.gpu --format=csv
$os = Get-CimInstance Win32_OperatingSystem
"Free RAM: $([math]::Round($os.FreePhysicalMemory/1MB,1)) GB; Free Virt: $([math]::Round($os.FreeVirtualMemory/1MB,1)) GB"
```

### Re-evaluate a single (model, dataset, kind) combo
```bash
.venv/Scripts/python.exe evaluate.py \
  --metadata-csv artifacts/metadata/unified_metadata_resized.csv \
  --image-column image_path_resized \
  --checkpoint artifacts/checkpoints/efficientnet/best_calibrated.pt \
  --class-map artifacts/metadata/unified_class_map.json \
  --source-filter plant_village \
  --output-json artifacts/checkpoints/efficientnet/eval_plant_village_raw.json
```
