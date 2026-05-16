# Plant Leaf Disease Detection and Classification Project Plan

## Objective
Build a robust, real-time plant leaf disease system using a segmentation-first, background-agnostic pipeline and a lightweight SOTA classifier suitable for desktop/webcam inference.

Current execution priority:
- Build and validate the full v1 pipeline using Plant Village only.
- Defer combined training with PlantDoc to a future expansion phase after v1 baseline is stable.

Step status:
- Step 1 (dataset verification): completed on Plant Village.
- Step 2 (project plan): finalized in this document.

## Scope and Assumptions
- Labels are folder-level class labels (no detection boxes or segmentation masks).
- Dataset layout can be either:
  - `dataset_root/<crop>/<class>/<images>`
  - `dataset_root/<crop>/<split>/<class>/<images>`
- v1 training dataset: Plant Village Dataset (Updated) only.
- PlantDoc dataset is reserved for future domain adaptation and robustness improvements.
- Preprocessing will isolate leaves before classification using pre-trained/background-removal methods.
- Final user experience is a Streamlit app with both upload and live webcam modes.

## End-to-End Architecture
1. Data Verification:
- Run integrity and class-balance checks on Plant Village for v1.
- Identify class gaps across splits and corrupted files.

Step 1 verified results (Plant Village):
- Total images: 67,118
- Layout: crop_split_class
- Corrupted/unreadable files: 0
- Imbalance stats: min class count 43, max class count 2016, ratio 46.88x

2. Segmentation-First Preprocessing:
- Background removal (`rembg`) with fallback (`grabCut`) when needed.
- Preserve foreground leaf pixels and replace background with neutral color (or alpha-aware compositing).

3. Classification:
- EfficientNetV2-S (primary recommendation) with transfer learning.
- Optional ConvNeXt-Tiny comparison baseline.

4. Evaluation and Export:
- Macro F1, weighted F1, per-class recall, confusion matrix, and confidence calibration.
- Export best model (TorchScript or ONNX) for low-latency inference.

5. Future Expansion (Post-v1):
- Integrate PlantDoc as additional training/fine-tuning data for background/noise robustness.
- Keep separate benchmark reporting for Plant Village test and PlantDoc test.

6. Streamlit UI:
- Single-image inference and live camera inference.
- Confidence display, top-k predictions, optional segmentation preview.

---

## Phase 1: Preprocessing and Data Augmentation Pipeline

### 1.1 Data Verification and Metadata Build
- Run the dataset verification script to gather:
  - per-class image counts,
  - split coverage,
  - corrupted image list.
- Generate a metadata index (`metadata.csv`) with fields:
  - `image_path`, `crop`, `class_name`, `split`, `source_dataset`.
- For v1, generate metadata from Plant Village only.
- Remove or quarantine corrupted files.

Execution decision for v1:
- Use Plant Village train/val/test as-is.
- No file removal required from corruption check (0 corrupted files found).

### 1.2 Automated Leaf Isolation (Segmentation-First)
- Primary method: `rembg` (U2Net-based background removal).
- Fallback method: OpenCV `grabCut` for samples where mask quality is poor.
- Quality checks for segmentation output:
  - foreground area threshold,
  - edge continuity,
  - minimum non-background pixel ratio.
- Persist segmented images under a mirrored directory, e.g.:
  - `processed/<split>/<class>/<filename>.png`
- Cache segmentation outputs to avoid re-processing every training run.

### 1.3 Resize, Normalize, Tensor Conversion
- Recommended input size: `224x224` for fast baseline; `288` or `320` for higher accuracy sweeps.
- Convert to RGB tensors.
- Normalize using ImageNet mean/std for transfer learning:
  - mean: `[0.485, 0.456, 0.406]`
  - std: `[0.229, 0.224, 0.225]`
- Keep deterministic validation/test transforms.

### 1.4 Leaf-Specific Data Augmentation
Use only biologically plausible augmentations:
- Random horizontal flip.
- Small random rotations (e.g., ±20 degrees).
- Random resized crop with moderate scale range.
- Mild color jitter (brightness/contrast/saturation/hue) to simulate lighting variability.
- Optional: slight Gaussian blur/noise for camera robustness.

Avoid or limit:
- Extreme perspective distortions.
- Aggressive elastic transforms that alter lesion morphology.
- Strong color shifts that destroy disease color signatures.

### 1.5 Class Imbalance Handling
Final v1 strategy based on observed imbalance ratio (46.88x):
- Loss: CrossEntropyLoss with class weights from inverse class frequency.
- Sampler: WeightedRandomSampler on train split.
- Augmentation: targeted extra augmentation probability for minority classes.
- Fallback: switch to focal loss only if minority-class recall stays low after first tuning cycle.

---

## Phase 2: Model Architecture and Training Strategy

### 2.1 Recommended Model
Primary: EfficientNetV2-S
- Strong accuracy/latency trade-off.
- Efficient for near-real-time webcam inference on mid-range hardware.

Secondary benchmark: ConvNeXt-Tiny
- Useful as an ablation for accuracy vs compute.

### 2.2 Transfer Learning Strategy
1. Initialize with ImageNet pre-trained weights.
2. Replace classifier head with `num_classes` output.
3. Stage A (warmup): freeze backbone, train head only for a few epochs.
4. Stage B (fine-tune): progressively unfreeze deeper blocks with lower LR for backbone.
5. Optional discriminative learning rates:
- lower LR for earlier layers,
- higher LR for classifier head.

### 2.3 Hyperparameter Strategy
- Optimizer: AdamW.
- Base LR:
  - head-only warmup: `1e-3`
  - full fine-tune: `2e-4`
- Weight decay: `1e-4`.
- Batch size: start at 32 (fallback 16 if GPU memory is limited).
- Scheduler: CosineAnnealingLR (primary for v1 baseline).
- Loss: weighted CrossEntropyLoss (v1 default).

### 2.4 Regularization and Stability
- Label smoothing (small, e.g., 0.05).
- Mixed precision training (AMP) for speed.
- Gradient clipping if spikes occur.
- Reproducibility seeds and deterministic validation pipeline.

### 2.5 Callbacks and Training Control
- Early stopping on validation macro F1 (patience 8-12 epochs).
- Model checkpointing:
  - save best by macro F1,
  - save last epoch.
- Logging:
  - train/val loss,
  - macro F1 and per-class recall,
  - LR curve.

### 2.6 Evaluation Protocol
- Report per-split metrics and confusion matrix.
- Focus on macro F1 in addition to accuracy to handle imbalance.
- Calibrate confidence scores (temperature scaling) for safer UI confidence display.

---

## Phase 3: Streamlit User Interface

### 3.1 App Architecture
Modules:
- `preprocess.py`: segmentation and transform pipeline.
- `inference.py`: model loading, prediction, top-k confidence.
- `app.py`: Streamlit UI and interaction flow.

Artifacts:
- `best_model.pth` or exported `best_model.onnx`.
- `class_to_idx.json` and reverse mapping.

### 3.2 Single Image Upload Workflow
1. User uploads image via `st.file_uploader`.
2. App runs segmentation-first preprocessing.
3. App applies model transforms and inference.
4. Display:
- predicted class,
- confidence score,
- top-k alternatives,
- optional segmented preview.

### 3.3 Live Webcam Workflow
Option A (recommended first): `st.camera_input`
1. Capture frame on-demand.
2. Run same preprocessing and inference path.
3. Display class and confidence immediately.

Option B (advanced): OpenCV continuous stream
1. Open webcam frames in loop.
2. Run periodic inference (e.g., every N frames).
3. Show smoothed confidence over sliding window.

For real-time stability:
- Use input resolution around `224`.
- Cache model in Streamlit session state.
- Apply optional confidence threshold and `Unknown` label handling.

### 3.4 UX and Safety Details
- Show warning when confidence < threshold.
- Include disclaimer: model aid, not definitive diagnosis.
- Provide class name normalization (human-readable disease names).

---

## Milestones and Deliverables
1. M1 Data audit complete:
- Plant Village class counts, corruption report, split consistency findings.

2. M2 Preprocessing pipeline complete:
- Plant Village background removal + cached processed dataset.

3. M3 Baseline model complete:
- EfficientNetV2 training on Plant Village, best checkpoint, evaluation report.

4. M4 Streamlit app complete:
- Upload mode + camera mode + confidence visualization.

5. M5 Final optimization:
- Latency tuning, confidence calibration, packaging.

6. M6 Future expansion (deferred):
- Combine Plant Village + PlantDoc via staged fine-tuning.
- Re-run evaluation with per-dataset reporting.

---

## What We Need Before Training Code
Step 1 is complete and integrated.

Ready now for implementation:
- Build preprocessing pipeline with segmentation cache.
- Build training pipeline for EfficientNetV2-S on Plant Village.
- Train baseline using weighted loss + weighted sampler.
