# Troubleshooting

Real issues we hit during development on the target machine (RTX 5060 8 GB, i5-13gen, 16 GB DDR5, Windows 11) with concrete fixes. Sorted roughly in the order you'll encounter them.

---

## 1. `torch.cuda.is_available()` returns False on RTX 50-series

**Symptom:** `import torch; torch.cuda.is_available()` is `False`, or you get `CUDA error: no kernel image is available for execution on the device`.

**Cause:** RTX 50-series is Blackwell architecture, compute capability **sm_120**. Stock PyTorch wheels (built against CUDA 11.8 or 12.1) don't ship sm_120 kernels.

**Fix:** Install the CUDA 12.8 PyTorch wheel:

```powershell
.venv\Scripts\python.exe -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

You should end up with `torch>=2.7+cu128`. Verify:

```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
# expected: 2.11.0+cu128 True NVIDIA GeForce RTX 5060
```

The cu128 wheel ships its own CUDA runtime — no need to install the system CUDA Toolkit. The NVIDIA display driver (≥ 555) is sufficient.

---

## 2. `pip install` fails with `[Errno 28] No space left on device`

**Symptom:** PyTorch wheel is ~2.75 GB. `pip` extracts to `%TEMP%` (on C:) before installing. If C: is nearly full, install fails partway through.

**Fix:** Redirect pip's temp dir and cache to a drive with space (the project drive is fine):

```bash
cd "/d/work/MachineLearning proejcts/plant"
mkdir -p .pip-tmp .pip-cache
export TMP="$(pwd)/.pip-tmp" TEMP="$(pwd)/.pip-tmp" TMPDIR="$(pwd)/.pip-tmp" PIP_CACHE_DIR="$(pwd)/.pip-cache"
.venv/Scripts/python.exe -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

The `--no-cache-dir` flag avoids touching the user's pip cache on C:.

---

## 3. `OSError: [WinError 1455] The paging file is too small for this operation to complete`

**Symptom:** Training crashes immediately after `Device: cuda | Model: ...` line, before epoch 1. Or DataLoader workers fail to spawn with this error.

**Cause:** Each PyTorch DataLoader worker on Windows spawns a full child process that re-imports torch + sklearn + scipy + numpy. With the default Windows automatic pagefile (~16 GB), 4 workers × ~2 GB committed each = pagefile exhaustion.

**Fix in priority order:**

1. **Close memory-heavy apps** (Chrome, IDEs with many windows, etc.) until `Get-CimInstance Win32_OperatingSystem | Select FreePhysicalMemory` shows >5 GB free.
2. **Drop `--num-workers`** to 2 (works on this machine), or 1, or 0 if needed.
3. **Permanent fix:** increase Windows pagefile to 32 GB+ via System Properties → Advanced → Performance → Advanced → Virtual memory. Requires reboot.

Used config that consistently works on this hardware:
```bash
.venv/Scripts/python.exe -u train.py ... --num-workers 2 --prefetch-factor 2 --batch-size 64 --amp
```

---

## 4. `RuntimeError: bad allocation` mid-training

**Symptom:** Training runs for a few epochs then crashes with this generic error. Pre-trained model loaded successfully, but a forward pass deep into training fails to allocate.

**Cause:** Stale Python processes from previous failed runs holding committed virtual memory. We saw 9 zombie Python processes at one point holding ~17 GB committed. Even after `Ctrl+C`, child processes from `multiprocessing.Pool` and dataloader workers can outlive the parent.

**Fix:** kill all stray python processes from the project venv before relaunching:

```powershell
Get-Process python -ErrorAction SilentlyContinue |
  Where-Object { $_.Path -like '*\.venv\*' } |
  ForEach-Object { Stop-Process -Id $_.Id -Force }
```

Verify free RAM/virt memory:
```powershell
$os = Get-CimInstance Win32_OperatingSystem
"Free RAM:  $([math]::Round($os.FreePhysicalMemory/1MB,1)) GB"
"Free Virt: $([math]::Round($os.FreeVirtualMemory/1MB,1)) GB"
"Commit:    $([math]::Round((Get-Counter '\Memory\Committed Bytes').CounterSamples[0].CookedValue/1GB, 1)) / $([math]::Round((Get-Counter '\Memory\Commit Limit').CounterSamples[0].CookedValue/1GB, 1)) GB"
```

Should show >5 GB free, commit/limit ratio <80%.

---

## 5. `torch.OutOfMemoryError: CUDA out of memory` despite plenty of free VRAM

**Symptom:** OOM message says e.g. *"Tried to allocate 294 MiB. GPU 0 has a total capacity of 7.96 GiB of which 6.07 GiB is free"* — i.e. there's plenty of free VRAM but allocation still fails.

**Cause:** Allocator fragmentation. The PyTorch caching allocator can hold reserved-but-unused memory in chunks too small for the next allocation. Common with `channels_last` memory format on Blackwell.

**Fix:** drop `--channels-last` (the speedup is minimal on this hardware anyway). On Linux you'd also try `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — but **`expandable_segments` is not supported on Windows** (PyTorch will print a warning and ignore it). The fix is just batch size + no channels_last:

```bash
.venv/Scripts/python.exe -u train.py ... --batch-size 64 --amp   # no --channels-last
```

---

## 6. `cv2.error: bad allocation` in `cv2.grabCut`

**Symptom:** Segmentation script crashes when processing PlantDoc images.

**Cause:** PlantDoc has natural-photo images, some 4000×3000+. cv2's grabCut allocates roughly `(W*H)^1.5` working memory and OOMs on huge inputs.

**Fix:** [`src/plant_disease/segmentation.py`](../src/plant_disease/segmentation.py) caps the input at `max_side=1024` before calling grabCut. Already applied in current code.

Plus [`scripts/preprocess_segment.py`](../scripts/preprocess_segment.py) wraps each image in try/except and falls back to writing the raw image as PNG when segmentation fails. Resulting metadata column `segmentation_method` shows `failed_<ErrorType>` for these.

---

## 7. Training takes forever / GPU sits at <5% utilization

**Symptom:** `nvidia-smi` shows GPU utilization near 0% during training. Per-epoch time is enormous (>20 min).

**Cause:** PIL.open + resize is the bottleneck. Plant Village JPEGs are small (<300 KB) but PlantDoc images are 1-5 MB and decode/resize slowly.

**Fix:** pre-resize once into a 224×224 JPEG cache:

```bash
.venv/Scripts/python.exe -u scripts/cache_resized.py \
  --metadata-csv artifacts/metadata/unified_metadata.csv \
  --output-csv artifacts/metadata/unified_metadata_resized.csv \
  --size 224 --workers 6
```

70k images, ~10 minutes. Then train with `--image-column image_path_resized`. Per-epoch wall time drops from ~20 min to ~3 min on this machine; GPU util climbs to ~85%.

---

## 8. Power outage / sleep / Claude Code crash kills training mid-run

**Symptom:** You come back to find training process gone, only some checkpoints present.

**Fix:** `train.py` saves full state (model + optimizer + scheduler + AMP scaler + epoch + best_f1) to `last.pt` after each epoch. To resume, append `--resume`:

```bash
.venv/Scripts/python.exe -u train.py \
  --metadata-csv artifacts/metadata/unified_metadata_resized.csv \
  --image-column image_path_resized \
  --model cnn --epochs 25 --batch-size 64 --image-size 224 \
  --amp --num-workers 2 --prefetch-factor 2 \
  --output-dir artifacts/checkpoints/cnn \
  --resume
```

LR schedule, AMP scaler, optimizer momentum — all restored exactly. Verify the resume by looking for the `Resumed from epoch N -> starting at epoch N+1` line.

To prevent the issue: Settings → System → Power → "Screen and sleep" → "When plugged in, put my device to sleep after" → **Never**.

---

## 9. EfficientNet always predicts the same class for every image

**Symptom:** EffNet predicts `PlantDoc___Tomato_two_spotted_spider_mites_leaf` (or similar minority class) for any uploaded image. Eval matrix shows top-1 = 0% but top-5 = 99.99% on Plant Village.

**Cause:** Combination of `WeightedRandomSampler(weights=1/count)` + `CrossEntropyLoss(weight=1/count)`. The two amplify each other and inflate the bias term for minority classes during training.

**Fix:** post-hoc logit calibration:

```bash
.venv/Scripts/python.exe scripts/logit_adjust.py \
  --checkpoint artifacts/checkpoints/efficientnet/best.pt
```

This bakes `log(p_natural)` into the classifier bias and saves `best_calibrated.pt`. The Streamlit app sidebar already defaults to `best_calibrated.pt`. PV accuracy: 0% → 99.93% with no retraining.

For deeper analysis see [ARCHITECTURE.md § The bias bug](ARCHITECTURE.md#the-bias-bug--the-calibration-fix).

**Lesson for future training runs:** pick **one** balancing mechanism, not both. Either:
- WeightedRandomSampler with vanilla CE (recommended), or
- Natural sampling with class-weighted CE

Never combine them.

---

## 10. Streamlit shows wrong predictions even after using calibrated checkpoint

**Symptom:** App is loading `best_calibrated.pt` but predictions are still off (worse than the eval matrix numbers suggest).

**Cause:** The "Use segmentation-first preprocessing" sidebar checkbox is on, but the model was trained on raw resized images (not segmented). Inference-time segmentation creates a distribution shift the model never saw.

**Fix:** turn off the checkbox. It defaults to off in the current `app.py`. Only turn it on if you want to evaluate how segmentation affects predictions.

---

## 11. Predictions return wrong class names

**Symptom:** Top-k labels in the UI don't match what you'd expect from the source dataset.

**Causes & fixes:**

- **Stale class_map.** Make sure the sidebar `Class map` field points to `artifacts/metadata/unified_class_map.json` (36 classes), not the old 29-class `class_map.json`.
- **Wrong checkpoint num_classes.** `Predictor.__init__` reads `ckpt["num_classes"]` and builds the model with that head size. If the checkpoint expects 37 classes and the class_map has 36, top-1 indices will misalign. Use `best_calibrated.pt` which has the merged corn-rust class (36 classes) baked in.
- **Folder name mismatch in alias map.** `plantdoc_alias.json` keys must match the actual PlantDoc folder names exactly (case + spelling). One typo creates a duplicate class. We hit this with `Common Rust_` vs `Common Rust`.

---

## 12. `scripts/run_eval_matrix.sh` fails on Windows

**Symptom:** `bash: command not found` or path errors when running the shell script.

**Fix:** The shell script needs Git Bash or WSL. From PowerShell, you can either:
1. Use Git Bash directly: `& "C:\Program Files\Git\bin\bash.exe" scripts/run_eval_matrix.sh`
2. Or run the 8 evals manually — see [RUNBOOK.md § Step 6](../RUNBOOK.md).

---

## 13. `evaluate.py` crashes with `FileNotFoundError` on segmented images

**Symptom:** Eval crashes mid-run when reading from `unified_test_segmented.csv`. A few rows have `processed_path` pointing to files that don't exist.

**Cause:** A handful of PlantDoc images failed both rembg and grabCut (oversized or corrupted). Their `segmentation_method` is `failed_*` and the fallback "save raw as PNG" also failed.

**Fix:** filter the CSV to existing files and use the cleaned version:

```bash
.venv/Scripts/python.exe -c "
import os, pandas as pd
df = pd.read_csv('artifacts/metadata/unified_test_segmented.csv')
df = df[df['processed_path'].apply(os.path.exists)].copy()
df.to_csv('artifacts/metadata/unified_test_segmented_ok.csv', index=False)
print(f'kept {len(df)} rows')
"
```

The eval matrix script already uses `_ok.csv`.

---

## 14. Two outputs for what looks like the same disease class

**Symptom:** `unified_class_map.json` shows two adjacent indices for the same disease, e.g. `Corn (Maize)___Common Rust` and `Corn (Maize)___Common Rust_`.

**Cause:** Typo in `plantdoc_alias.json` mapping a PlantDoc class to a slightly different Plant Village name (trailing underscore, capitalization, etc.).

**Fix without retraining:** classifier head surgery via [`scripts/merge_class.py`](../scripts/merge_class.py):

```bash
.venv/Scripts/python.exe scripts/merge_class.py \
  --checkpoint artifacts/checkpoints/efficientnet/best.pt \
  --out-checkpoint artifacts/checkpoints/efficientnet/best_merged.pt \
  --src-idx 10 --target-idx 9
```

This collapses idx 10 into idx 9 in the trained classifier head:
- `new_weight[9] = old_weight[9] + old_weight[10]`
- `new_bias[9] = log(exp(old_bias[9]) + exp(old_bias[10]))`  (logsumexp — exact for softmax)

Output is a 36-class checkpoint mathematically equivalent to running the original 37-class model and summing the two probabilities. Then re-run `logit_adjust.py` on the merged checkpoint to recalibrate.

**Fix with retraining:** edit `plantdoc_alias.json`, re-run `build_unified_metadata.py`, re-run `cache_resized.py` (idempotent — only re-resizes the affected rows), retrain.
