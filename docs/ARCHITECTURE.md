# Architecture

This document explains how the pipeline is wired together and why specific choices were made. For step-by-step commands, see [RUNBOOK.md](../RUNBOOK.md). For known issues and fixes, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## High-level flow

```
                Plant Village 2/                PlantDoc 1/
                (crop/split/class)              (split/class)
                       │                              │
                       │   build_unified_metadata.py  │
                       └──────────────┬───────────────┘
                                      ▼
                      artifacts/metadata/unified_metadata.csv
                      (70,040 rows × 36 classes, with label_idx)
                                      │
                                      ▼
                      ┌──────────────────────────────┐
                      │ cache_resized.py             │
                      │ (parallel JPEG resize → 224) │
                      └──────────────┬───────────────┘
                                     ▼
                  artifacts/processed/resized/<source>/<split>/<class>/*.jpg
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │ train.py --model {cnn,effnet}│ (AMP, --resume)
                      └──────────────┬───────────────┘
                                     ▼
                      best.pt → merge_class.py → best_merged.pt
                                                       │
                                                       ▼
                                            logit_adjust.py
                                                       │
                                                       ▼
                                          best_calibrated.pt  (production)
                                                       │
                                                       ▼
                          ┌─────────────────────────────────────────────────┐
                          │ evaluate.py × 8                                 │
                          │ (cnn, effnet) × (PV, PD) × (raw, segmented)     │
                          └──────────────┬──────────────────────────────────┘
                                         ▼
                        artifacts/checkpoints/<model>/eval_*.json
                                         │
                                         ▼
                                   app.py (Streamlit)
                                   • Tab 1: Test CNN
                                   • Tab 2: Test EfficientNet
                                   • Tab 3: Compare (live + static benchmarks)
```

## Data

### Datasets

| | Plant Village 2 | PlantDoc 1 |
|---|---|---|
| Layout | `<crop>/{Train,Val,Test}/<class>/*.jpg` | `{train,test}/<class>/*.jpg` |
| Conditions | Studio, neutral background | Real-world, cluttered backgrounds |
| Counts (train/val/test) | 53,693 / 12,067 / 1,358 | 2,414 / 256* / 252 |
| Classes | 29 (multiple crops, healthy + diseases) | 28 in train, 27 in test |

*PlantDoc has no native val split; we carve a stratified 10% from train at metadata-build time (deterministic via `--seed 42` in `build_unified_metadata.py`).

### Class unification (the alias map)

PlantDoc class folder names are looser than Plant Village's. We map them via [`artifacts/metadata/plantdoc_alias.json`](../artifacts/metadata/plantdoc_alias.json):

```json
{
  "Apple_Scab_Leaf":         "Apple___Apple Scab",
  "Corn_rust_leaf":          "Corn (Maize)___Common Rust",
  "Tomato_leaf_yellow_virus":"Tomato___Yellow Leaf Curl Virus",
  ...
  "Blueberry_leaf":  null,    // no PV equivalent → kept as PlantDoc___Blueberry_leaf
  "Soyabean_leaf":   null,
  ...
}
```

Aliases with `null` keep the PlantDoc folder name (prefixed `PlantDoc___`) as a brand-new class. Result: **36 classes** in the unified label space (29 from Plant Village + 7 PlantDoc-only).

### Pre-resize cache (why this exists)

Original plan: train directly off `image_path` with on-the-fly resize. Reality on this hardware:
- Plant Village images: ~100-300 KB JPEG, ~4ms each to PIL.open + resize
- PlantDoc images: 1-5 MB JPEG (some 4000×3000), ~50-200ms each
- With `num_workers=2` and 70k train images, **per-epoch dataloading wall time ≈ 25 min** while GPU sits at <5% util

[`scripts/cache_resized.py`](../scripts/cache_resized.py) parallel-resizes everything to 224×224 JPEG (quality 92) once. Output paths:
```
artifacts/processed/resized/{plant_village,plantdoc}/{train,val,test}/<class>/<stem>.jpg
```
After caching: per-epoch dataloading ≈ 30s, GPU utilization ~85%.

### Segmented test set

For the "segmented vs raw" evaluation comparison we run [`scripts/preprocess_segment.py`](../scripts/preprocess_segment.py) **only on the test split** (1,610 images, ~5 min on grabCut). Output PNGs land at:
```
artifacts/processed/{plant_village,plantdoc}/test/<class>/<stem>.png
```
13 of 1,610 segmentations failed (oversized PlantDoc images crash cv2's grabCut allocator). The script wraps each call in try/except and falls back to writing the raw image — `unified_test_segmented_ok.csv` then filters to the 1,603 that produced files.

We do **not** segment the training set. rembg+U2Net runs at ~1.3 s/img on CPU on this machine (GPU rembg requires `onnxruntime-gpu` and stricter setup) → 25+ hours for 70k images. Training on raw and reporting raw vs segmented test metrics tells you whether segmentation helps; it doesn't.

## Models

[`src/plant_disease/models.py`](../src/plant_disease/models.py) defines two architectures dispatched via `build_model(name, num_classes, use_pretrained)`:

### SimpleCNN (399K params, 4.6 MB)

```python
Conv(3→32) → BN → ReLU → MaxPool ×4 stages (32→64→128→256)
GlobalAvgPool → Dropout(0.5) → Linear(256, num_classes)
```

A from-scratch baseline. No pretraining. Lightweight enough for CPU inference (~0.6 ms/img).

### EfficientNetV2-S (20.2M params, 232.7 MB)

`torchvision.models.efficientnet_v2_s(weights=DEFAULT)` initialized from ImageNet, with the final `classifier[1]` Linear replaced to output `num_classes`. ~1.5 ms/img on RTX 5060.

## Training

[`train.py`](../train.py) handles both architectures.

### Fixed across both runs
- 224×224 input, ImageNet mean/std normalization
- AdamW(lr=2e-4 base, weight_decay=1e-4)
- CosineAnnealingLR(T_max=epochs)
- CrossEntropyLoss with `label_smoothing=0.05`
- Augmentation: HorizontalFlip(0.5), RandomRotation(±20°), ColorJitter(0.15/0.15/0.15/0.03)
- Validation transform: just Resize+Normalize (deterministic)
- AMP (mixed precision) on GPU
- WeightedRandomSampler with `weight = 1/class_count` (balance batches across classes)
- Per-class CE loss weights = `1/class_count` (amplify minority loss)

### Differences

| | CNN | EfficientNet |
|---|---|---|
| Pretrained | No | Yes (ImageNet) |
| Epochs | 25 | 20 + 5 fine-tune at lr=5e-5 |
| Batch size | 64 | 64 |

The EffNet 5-epoch fine-tune was added for parity. It uses `--resume --fresh-schedule --lr 5e-5` to start a fresh 5-step cosine over the last 5 epochs at a lower LR — proper transfer-learning second stage rather than just continuing the original schedule.

### Resume after interruption

`last.pt` saves at every epoch boundary with **full state**: model, optimizer, scheduler, AMP scaler, epoch number, best_f1. Re-running the same command with `--resume` continues from epoch+1 with the exact LR schedule. `--fresh-schedule` forces a new LR cosine over the remaining epochs (used for the EffNet fine-tune).

## The bias bug & the calibration fix

### Symptom

After training, the EfficientNet model predicted `PlantDoc___Tomato_two_spotted_spider_mites_leaf` for **every** uploaded image. Eval matrix showed:
- top-1 accuracy on Plant Village test: **0.000%**
- top-5 accuracy on Plant Village test: **99.99%**

The model knew the right answer (top-5) but always picked the wrong top-1.

### Root cause

The training config combined two class-balancing mechanisms:

1. **`WeightedRandomSampler`** with weight `1/class_count` per sample → each batch is roughly balanced across classes (minority classes upsampled).
2. **Class-weighted `CrossEntropyLoss`** with weight `1/class_count` → loss is amplified for minority classes.

These compound. The effective gradient signal per minority-class example is `(1/N) × (1/count) × (gradient)` — minority classes contribute **10-20× more** to the loss than majority classes.

The cheapest way for the optimizer to reduce this amplified loss is to **inflate the bias term** of minority-class outputs. Inspection of the trained EfficientNet classifier confirmed:

| class index | bias | weight norm | meaning |
|---|---:|---:|---|
| 24 (Tomato_two_spotted_spider_mites) | **+0.137** | 0.56 | huge bias, weak features |
| 18-23 (other PlantDoc-only) | +0.005 to +0.017 | 0.55-0.67 | small bias, weak features |
| 0-17, 25-35 (Plant Village classes) | -0.07 to -0.08 | 1.0-1.1 | negative bias, strong features |

The model encoded great Plant Village representations (large weight norms), but the inflated PlantDoc bias dominated argmax for any input. With 100% probability on the same idx for 100 random PV inputs, accuracy is exactly 0.

### Fix: post-hoc logit adjustment

[`scripts/logit_adjust.py`](../scripts/logit_adjust.py) applies the [Balanced Softmax / Logit Adjustment](https://arxiv.org/abs/2007.07314) inference rule:

```
adjusted_logit = original_logit + log(p_natural)
```

Where `p_natural` is the natural training class frequency (before sampler).

The intuition: WeightedRandomSampler made the model implicitly assume uniform priors at training time. At inference on real-world data with the natural frequency distribution, we add `log(p_natural)` to compensate.

**Result on the test split:**

| strategy | overall acc | PV acc | PD acc |
|---|---:|---:|---:|
| baseline | 1.99% | 0.00% | 12.70% |
| **prior_natural** | **93.42%** | **99.93%** | **58.33%** |

The fix is **free** — no retraining, just a single tensor add. The `logit_adjust.py` script bakes the adjustment directly into the classifier bias and saves a new checkpoint (`best_calibrated.pt`) so downstream code (inference, evaluation, UI) is oblivious.

### Why the bug stayed hidden during training

`val_macro_f1` during training was 0.08-0.15 — not great, but not 0%. Macro F1 averages per-class F1, and with the model nailing the upsampled minority classes (which appear often in the val set due to PV+PD merge), the macro number wasn't pathological. Only test-set per-source breakdown revealed PV top-1 = 0%.

**Lesson for next time:** track top-1 **per source** (PV vs PD separately) during training, not just global macro F1.

## The corn rust class deduplication

After the calibration fix, we noticed indices 9 and 10 were nearly identical:
- idx 9: `Corn (Maize)___Common Rust` (Plant Village folder)
- idx 10: `Corn (Maize)___Common Rust_` (PlantDoc alias mapped here, with stray underscore)

A typo in `plantdoc_alias.json`: PlantDoc's `Corn_rust_leaf` mapped to `"Corn (Maize)___Common Rust_"` (trailing underscore) instead of `"Corn (Maize)___Common Rust"`. This split corn rust into two classes — Plant Village images labeled idx 9, PlantDoc corn rust images labeled idx 10.

### Fix without retraining: classifier head surgery

[`scripts/merge_class.py`](../scripts/merge_class.py) collapses two output classes in a trained model:

```
new_weight[target] = old_weight[target] + old_weight[src]
new_bias[target]   = log(exp(old_bias[target]) + exp(old_bias[src]))   # logsumexp
                                                                       # exact for softmax merging
drop old row[src]
```

This is a **mathematically valid** operation for softmax ensembling: `softmax(a, b, c) → softmax(a+log(exp(a)+exp(b)), c)` produces the same probability for the merged class as summing `softmax(a) + softmax(b)`. We approximate weights by summation (close enough since the two classes share features).

After head surgery, num_classes = 36, and the calibration step is re-run on the merged model.

## Inference (the Predictor)

[`src/plant_disease/inference.py`](../src/plant_disease/inference.py) is the entry point used by the Streamlit UI:

```python
predictor = Predictor(checkpoint_path, class_map_path, image_size=224)
top_k, processed_image, raw_probs = predictor.predict_topk(image, k=5, use_segmentation=False)
```

- Auto-detects CUDA if available, else CPU.
- Reads `ckpt["model"]` to dispatch to SimpleCNN or EfficientNetV2-S.
- Default `use_segmentation=False` — models were trained on raw resized images. The UI checkbox is preserved as an experiment toggle but defaults off.
- Returns top-k + processed image + full prob tensor (UI uses prob tensor for the comparison bar charts).

## Evaluation

[`evaluate.py`](../evaluate.py) emits a self-contained JSON for one (model, dataset, image-kind) combo. Schema:

```json
{
  "checkpoint": "...best_calibrated.pt",
  "model": "efficientnet",
  "split": "test",
  "image_column": "image_path_resized",
  "source_filter": "plant_village",
  "n_samples": 1358,
  "device": "cuda",
  "num_classes": 36,
  "metrics": {
    "accuracy": 0.9993,
    "macro_f1": 0.9993,
    "weighted_f1": 0.9993,
    "macro_precision": 0.9994,
    "macro_recall": 0.9993,
    "top3_accuracy": 1.0,
    "top5_accuracy": 1.0
  },
  "per_class": [{"idx": 0, "name": "Apple___Apple Scab", "precision": 1.0, "recall": 0.95, "f1": 0.97, "support": 20}, ...],
  "confusion_matrix": [[19, 0, 0, ...], ...],
  "model_info": {
    "parameters": 20223604,
    "checkpoint_size_mb": 232.7,
    "mean_latency_ms_per_image": 1.55
  }
}
```

The Streamlit Compare tab reads these JSON files directly — no recomputation in the UI.

## UI

[`app.py`](../app.py) is a 3-tab Streamlit app:

- **Tab 1 — Test: CNN** — file uploader + camera input, runs CNN top-5 prediction, shows segmentation preview, bar chart of confidences.
- **Tab 2 — Test: EfficientNet** — same UI, EffNet predictor.
- **Tab 3 — Compare** —
  - Live: same image fed to both models, side-by-side top-5 with bar charts.
  - Static: dataset toggle (Plant Village / PlantDoc) × image-kind toggle (raw / segmented). Loads the corresponding `eval_*.json` files and renders:
    - Metrics table (accuracy, macro F1, weighted F1, top-3, top-5, params, checkpoint MB, latency ms/img) with a "winner per metric" column
    - Per-class F1 bar chart (CNN vs EffNet)
    - Confusion matrices (matplotlib, row-normalized)

`@st.cache_resource` caches both predictors in session state so model reloading on rerun is free.

## Why these specific choices

- **EfficientNetV2-S over Tiny ConvNeXt or larger EffNet variants:** balances accuracy and ~1.5ms/img inference latency, fits in 8GB VRAM at batch 64 with AMP. ConvNeXt-Tiny was the secondary candidate in the original plan; we didn't pursue it because EffNet already saturated PV near 100%.
- **224×224 resolution:** standard ImageNet size, matches pretrained weights' assumption, fast enough for real-time webcam inference.
- **No `torch.compile`:** Blackwell + cu128 + Python 3.13 is bleeding-edge and `compile()` had spurious failures during initial smoke tests. Throughput is GPU-bound at 85%+ already without it.
- **No `channels_last`:** caused allocator fragmentation on RTX 5060 with cu128 (the `expandable_segments` allocator is not supported on Windows). Standard NCHW works fine.
- **rembg dropped, grabCut kept:** rembg+U2Net on CPU is too slow for 70k images. grabCut is ~10× faster and runs in OpenCV without extra deps. Quality is lower for natural-photo PlantDoc images, which is reflected in the segmented eval being worse than raw.
