# Plant Leaf Disease Detection — CNN vs EfficientNetV2-S

A leaf disease classifier trained on **Plant Village** + **PlantDoc** (70,040 images, 36 classes) that compares a from-scratch SimpleCNN against a pretrained EfficientNetV2-S, with a Streamlit UI for both per-model testing and side-by-side comparison.

## Headline results

8-eval matrix on the held-out test split (best_calibrated.pt, 36-class):

| model | eval | acc | macro F1 | weighted F1 | top-5 |
|---|---|---:|---:|---:|---:|
| **EfficientNet** | PV raw | **99.93%** | **99.93%** | **99.93%** | **100.00%** |
| EfficientNet | PV segmented | 94.33% | 91.05% | 94.23% | 98.23% |
| **EfficientNet** | PD raw | **58.33%** | 50.76% | 58.35% | 88.10% |
| EfficientNet | PD segmented | 54.29% | 48.68% | 53.34% | 81.22% |
| CNN | PV raw | 81.22% | 75.48% | 80.49% | 98.31% |
| CNN | PV segmented | 25.33% | 23.35% | 26.79% | 66.49% |
| CNN | PD raw | 17.46% | 11.24% | 10.92% | 45.24% |
| CNN | PD segmented | 11.84% | 8.10% | 7.32% | 36.73% |

- **EfficientNetV2-S** (20.2M params, 232.7 MB ckpt, ~3 ms/img on RTX 5060) wins every metric.
- **SimpleCNN** (399K params, 4.6 MB ckpt, ~0.6 ms/img) is the lightweight baseline.

PV = Plant Village test split (1358 imgs, 29 disease classes); PD = PlantDoc test split (245 imgs, mixed in-domain + 7 PlantDoc-only classes).

## What's in this repo

```
.
├── app.py                          # Streamlit UI: 3 tabs (Test CNN / Test EffNet / Compare)
├── train.py                        # Trainer (--model {cnn,efficientnet}, AMP, --resume)
├── evaluate.py                     # Single-eval script that emits comparable JSON metrics
├── dataset_verification.py         # Initial data audit (corruption, class balance)
├── requirements.txt
├── RUNBOOK.md                      # Step-by-step reproduction commands
├── project_plan.md                 # Original design document
├── docs/
│   ├── ARCHITECTURE.md             # Technical design + data flow
│   └── TROUBLESHOOTING.md          # Issues we hit during development + fixes
├── src/plant_disease/
│   ├── data.py                     # PlantDiseaseDataset (CSV-driven)
│   ├── models.py                   # SimpleCNN + build_model() dispatcher
│   ├── inference.py                # Predictor (top-k, segmentation toggle)
│   └── segmentation.py             # rembg / grabCut leaf isolation
├── scripts/
│   ├── build_metadata.py           # Plant Village–only metadata
│   ├── build_unified_metadata.py   # Plant Village + PlantDoc unified metadata
│   ├── cache_resized.py            # Pre-resize 70k images → 224×224 JPEG cache
│   ├── preprocess_segment.py       # rembg/grabCut segmentation pipeline
│   ├── run_eval_matrix.sh          # Run all 8 evals (PV/PD × raw/seg × cnn/effnet)
│   ├── summarize_evals.py          # Pretty-print eval JSON files
│   ├── logit_adjust.py             # Post-hoc bias calibration (fixes sampler/loss imbalance)
│   └── merge_class.py              # Classifier-head surgery for class deduplication
└── artifacts/
    ├── metadata/
    │   ├── plantdoc_alias.json             # PlantDoc → Plant Village class mapping
    │   ├── unified_class_map.json          # 36 unified classes (idx → name)
    │   ├── unified_metadata.csv            # All 70k rows with raw image_path
    │   ├── unified_metadata_resized.csv    # Adds image_path_resized column (224×224 jpeg)
    │   └── unified_test_segmented_ok.csv   # Test split with grabCut segmentation
    ├── processed/
    │   ├── resized/<source>/<split>/<class>/*.jpg   # 224×224 training cache
    │   └── <source>/<split>/<class>/*.png           # Segmented test images (for "segmented" eval)
    └── checkpoints/
        ├── cnn/{best,last,best_merged,best_calibrated}.pt
        └── efficientnet/{best,last,best_merged,best_calibrated}.pt
```

`best_calibrated.pt` is the **production checkpoint** — it has the merged corn-rust class collapse + post-hoc logit-bias calibration baked in.

## Quick start

```powershell
# 1. Install deps (RTX 50-series needs PyTorch ≥2.7 cu128 — see docs/TROUBLESHOOTING.md)
.venv\Scripts\python.exe -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Verify CUDA
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 3. Launch the app (uses pre-trained checkpoints already in artifacts/)
.venv\Scripts\python.exe -m streamlit run app.py
# → http://localhost:8501
```

If you want to retrain from scratch or rebuild any artifact, see [RUNBOOK.md](RUNBOOK.md). For deeper context on architecture, calibration, and why decisions were made, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). For things that broke and how we fixed them, see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## Hardware used

Development hardware:
- NVIDIA RTX 5060 (Blackwell, 8 GB VRAM, sm_120) — needs PyTorch cu128
- Intel Core i5 (13th gen)
- 16 GB DDR5 RAM
- Windows 11

Wall-clock budget on this hardware:
| Phase | Time |
|---|---|
| Pre-resize cache (70k → 224×224 JPEG) | ~10 min |
| Test-set segmentation (1610 imgs) | ~10 min |
| CNN train, 25 epochs | ~60-90 min |
| EfficientNet train, 25 epochs (20 + 5 fine-tune) | ~140 min |
| 8-eval matrix | ~5 min |
| **Total reproduction** | ~3-4 hrs |

## Key design notes

1. **Both datasets, single label space.** PlantDoc class names are mapped via [`plantdoc_alias.json`](artifacts/metadata/plantdoc_alias.json) into Plant Village's namespace where possible; remaining 7 PlantDoc-only classes (Soybean, Blueberry, Raspberry, Squash, Tomato mold/mosaic/spider-mite) get their own indices, prefixed `PlantDoc___`.
2. **Pre-resize cache** is critical on this hardware — without it, GPU sits at <5% utilization. 224×224 JPEG cache lets training run GPU-bound at ~85% util.
3. **Train on raw resized images, evaluate on both raw and segmented test sets.** Original plan was to train on segmented but `rembg` on CPU is ~25 hr for 70k images on this machine. Training on raw + comparing raw/segmented at eval time gives more useful info anyway (does segmentation actually help?).
4. **Post-hoc logit calibration** ([`scripts/logit_adjust.py`](scripts/logit_adjust.py)) is the single most impactful step. WeightedRandomSampler + class-weighted CE caused the model to over-bias toward minority PlantDoc classes — top-1 accuracy on PV dropped to **0%** even though top-5 was 99.99%. Adding `log(p_natural)` to logits at inference recovered top-1 to **99.93%** with no retraining.
5. **`--resume` flag** in `train.py` saves full state (model + optimizer + scheduler + AMP scaler + epoch + best F1) so power outages / sleep don't destroy progress. Use `--fresh-schedule` to restart the LR cosine over a new epoch budget when extending a training run.

## Limitations

- PlantDoc test accuracy (58%) is much lower than Plant Village (99.9%). PlantDoc images are real-world photos with cluttered backgrounds; Plant Village is studio shots. The model genuinely generalizes worse here, which is expected and matches published results.
- The 7 PlantDoc-only classes have ~100-200 train images each — not enough for strong representations. Top-5 accuracy is much better than top-1 on PD, suggesting the model encodes the right features but can't always pick the right minority class.
- This is a classifier, not a detector. It assumes the input is a leaf, mostly centered. Doesn't draw boxes or handle multi-leaf scenes.
