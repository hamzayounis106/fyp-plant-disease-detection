from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.plant_disease.inference import Predictor

st.set_page_config(page_title="Plant Disease Detection — CNN vs EfficientNet", layout="wide")
st.title("Plant Leaf Disease Detection: CNN vs EfficientNetV2-S")


# ---------- sidebar ----------
with st.sidebar:
    st.header("Model Settings")
    # cnn_ckpt = st.text_input("CNN checkpoint", "artifacts/checkpoints/cnn/best_calibrated.pt")
    # eff_ckpt = st.text_input("EfficientNet checkpoint", "artifacts/checkpoints/efficientnet/best_calibrated.pt")
    # class_map_path = st.text_input("Class map", "artifacts/metadata/unified_class_map.json")
    # metrics_dir = st.text_input("Metrics dir", "artifacts/checkpoints")
    cnn_ckpt = "artifacts/checkpoints/cnn/best_calibrated.pt"
    eff_ckpt = "artifacts/checkpoints/efficientnet/best_calibrated.pt"
    class_map_path = "artifacts/metadata/unified_class_map.json"
    metrics_dir = "artifacts/checkpoints"
    use_segmentation = st.checkbox("Segmentation-first preprocessing", value=False,
                                   help="Models were trained on raw resized images, not segmented. Leave OFF unless you specifically want to test segmentation.")
    image_size = st.number_input("Image size", min_value=128, max_value=512, value=224, step=32)


@st.cache_resource(show_spinner="Loading model…")
def load_predictor(ckpt: str, cmap: str, img_sz: int):
    return Predictor(ckpt, cmap, image_size=img_sz)


def safe_load(ckpt_path: str, label: str):
    try:
        return load_predictor(ckpt_path, class_map_path, int(image_size))
    except Exception as exc:
        # Streamlit can keep a previously failed cached resource; clear and retry once.
        load_predictor.clear()
        try:
            return load_predictor(ckpt_path, class_map_path, int(image_size))
        except Exception as retry_exc:
            st.warning(f"{label} model not loaded: {retry_exc}")
            return None


cnn = safe_load(cnn_ckpt, "CNN")
eff = safe_load(eff_ckpt, "EfficientNet")

tab_cnn, tab_eff, tab_cmp = st.tabs(["Test: CNN", "Test: EfficientNet", "Compare"])


# ---------- helpers ----------
def render_test_tab(predictor, label: str):
    if predictor is None:
        st.error(f"{label} predictor unavailable. Set checkpoint path in sidebar.")
        return
    st.caption(
        f"Model={predictor.model_name}  |  Params={predictor.parameters:,}  |  "
        f"Checkpoint={predictor.checkpoint_size_mb:.1f} MB  |  Device={predictor.device}"
    )
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Upload")
        up = st.file_uploader(
            "Choose a leaf image", type=["jpg", "jpeg", "png", "bmp", "webp"], key=f"{label}-up"
        )
        img = Image.open(up).convert("RGB") if up else None
    with col_r:
        st.subheader("Camera")
        shot = st.camera_input("Capture leaf", key=f"{label}-cam")
        if shot is not None:
            img = Image.open(shot).convert("RGB")

    if img is None:
        return
    topk, proc, _ = predictor.predict_topk(img, k=5, use_segmentation=use_segmentation)
    c1, c2 = st.columns(2)
    with c1:
        st.image(proc, caption="Processed image", use_container_width=True)
    with c2:
        st.success(f"Top-1: **{topk[0][0]}** — {topk[0][1]:.2%}")
        st.write("Top-5 confidences:")
        df = pd.DataFrame(topk, columns=["class", "probability"])
        st.bar_chart(df.set_index("class"))


PERF_METRICS = ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall", "top3_accuracy", "top5_accuracy"]
DATASETS = ["plant_village", "plantdoc"]


def load_metrics(model_subdir: str, dataset: str, image_kind: str):
    p = Path(metrics_dir) / model_subdir / f"eval_{dataset}_{image_kind}.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def combine_metrics(m1: dict, m2: dict) -> dict:
    """Average two metric dicts into a single combined result."""
    combined_metrics = {k: (m1["metrics"][k] + m2["metrics"][k]) / 2 for k in PERF_METRICS}
    combined_model_info = m1["model_info"]  # model_info is dataset-independent

    pc1 = pd.DataFrame(m1["per_class"])[["name", "f1"]].set_index("name")
    pc2 = pd.DataFrame(m2["per_class"])[["name", "f1"]].set_index("name")
    pc_combined = pc1.join(pc2, lsuffix="_1", rsuffix="_2", how="outer")
    pc_combined["f1"] = pc_combined[["f1_1", "f1_2"]].mean(axis=1)
    per_class_combined = [{"name": n, "f1": row["f1"]} for n, row in pc_combined.iterrows()]

    return {"metrics": combined_metrics, "model_info": combined_model_info, "per_class": per_class_combined}


def render_compare_tab():
    st.subheader("Live side-by-side prediction")
    up = st.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="cmp-up")
    if up is not None and cnn is not None and eff is not None:
        img = Image.open(up).convert("RGB")
        cnn_top, cnn_proc, _ = cnn.predict_topk(img, k=5, use_segmentation=use_segmentation)
        eff_top, eff_proc, _ = eff.predict_topk(img, k=5, use_segmentation=use_segmentation)
        c1, c2 = st.columns(2)
        with c1:
            st.image(cnn_proc, caption="CNN — input", use_container_width=True)
            st.success(f"CNN: {cnn_top[0][0]} ({cnn_top[0][1]:.2%})")
            st.bar_chart(pd.DataFrame(cnn_top, columns=["class", "p"]).set_index("class"))
        with c2:
            st.image(eff_proc, caption="EfficientNet — input", use_container_width=True)
            st.success(f"EffNet: {eff_top[0][0]} ({eff_top[0][1]:.2%})")
            st.bar_chart(pd.DataFrame(eff_top, columns=["class", "p"]).set_index("class"))

    st.divider()
    st.subheader("Static benchmark comparison")
    col_a, col_b = st.columns(2)
    with col_a:
        ds = st.selectbox("Dataset", ["plant_village", "plantdoc", "Both (combined)"], index=0)
    with col_b:
        kind = st.selectbox("Images", ["segmented", "raw"], index=0)

    if ds == "Both (combined)":
        pairs = {}
        missing = []
        for model_dir, label in [("cnn", "CNN"), ("efficientnet", "EfficientNet")]:
            ms = [load_metrics(model_dir, d, kind) for d in DATASETS]
            if any(m is None for m in ms):
                missing.append(label)
            else:
                pairs[label] = combine_metrics(ms[0], ms[1])
        if missing:
            for label in missing:
                st.info(
                    f"Metric JSONs not found for {label} (both datasets / {kind}). "
                    f"Generate `eval_plant_village_{kind}.json` and `eval_plantdoc_{kind}.json` "
                    f"under `artifacts/checkpoints/<model>/`."
                )
            return
        cnn_m, eff_m = pairs["CNN"], pairs["EfficientNet"]
        st.caption("Metrics are averaged across plant_village and plantdoc datasets.")
    else:
        cnn_m = load_metrics("cnn", ds, kind)
        eff_m = load_metrics("efficientnet", ds, kind)
        if cnn_m is None or eff_m is None:
            st.info(
                f"Metric JSON not found for {ds}/{kind}. Generate via:\n"
                f"`python evaluate.py --metadata-csv ... --checkpoint ... --output-json "
                f"artifacts/checkpoints/<model>/eval_{ds}_{kind}.json --source-filter {ds} "
                f"--image-column {'processed_path' if kind == 'segmented' else 'image_path'}`"
            )
            return

    rows = []
    for key in PERF_METRICS:
        rows.append({"metric": key, "CNN": cnn_m["metrics"][key], "EfficientNet": eff_m["metrics"][key]})
    rows.append({"metric": "parameters", "CNN": cnn_m["model_info"]["parameters"], "EfficientNet": eff_m["model_info"]["parameters"]})
    rows.append({"metric": "checkpoint_MB", "CNN": cnn_m["model_info"]["checkpoint_size_mb"], "EfficientNet": eff_m["model_info"]["checkpoint_size_mb"]})
    rows.append({"metric": "latency_ms_per_img", "CNN": cnn_m["model_info"]["mean_latency_ms_per_image"], "EfficientNet": eff_m["model_info"]["mean_latency_ms_per_image"]})
    metrics_df = pd.DataFrame(rows).set_index("metric")

    def winner(r):
        c, e = r["CNN"], r["EfficientNet"]
        if r.name in {"parameters", "checkpoint_MB", "latency_ms_per_img"}:
            return "CNN" if c < e else "EfficientNet"
        return "CNN" if c > e else "EfficientNet"

    metrics_df["winner"] = metrics_df.apply(winner, axis=1)
    st.dataframe(metrics_df.style.format({"CNN": "{:.4f}", "EfficientNet": "{:.4f}"}), use_container_width=True)

    st.markdown("**Per-class F1**")
    pc_cnn = pd.DataFrame(cnn_m["per_class"])[["name", "f1"]].rename(columns={"f1": "CNN_f1"})
    pc_eff = pd.DataFrame(eff_m["per_class"])[["name", "f1"]].rename(columns={"f1": "EffNet_f1"})
    pc = pc_cnn.merge(pc_eff, on="name").set_index("name")
    st.bar_chart(pc)

    if ds != "Both (combined)":
        st.markdown("**Confusion matrices**")
        cm_c1, cm_c2 = st.columns(2)
        try:
            import matplotlib.pyplot as plt
            for col, m, title in [(cm_c1, cnn_m, "CNN"), (cm_c2, eff_m, "EfficientNet")]:
                with col:
                    cm = np.array(m["confusion_matrix"])
                    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.imshow(cm_norm, cmap="Blues", aspect="auto")
                    ax.set_title(f"{title} — row-normalized")
                    ax.set_xlabel("predicted")
                    ax.set_ylabel("true")
                    st.pyplot(fig, clear_figure=True)
        except ImportError:
            st.warning("matplotlib not installed — skipping confusion-matrix figures.")


with tab_cnn:
    render_test_tab(cnn, "CNN")
with tab_eff:
    render_test_tab(eff, "EfficientNet")
with tab_cmp:
    render_compare_tab()
