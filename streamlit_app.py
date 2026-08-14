"""
streamlit_app.py -- Metamorphic Circuit Breaker -- Interactive Dashboard.

Mode flag
---------
The app accepts a --mode flag via the Streamlit CLI argument separator --:

    # Display results from an existing run folder (default)
    streamlit run streamlit_app.py -- --mode stored

    # Run the full Phase 1 pipeline, then display results
    streamlit run streamlit_app.py -- --mode live --data_dir data/isic2019 --n_samples 100

In stored mode the user selects a timestamped run folder from the sidebar.
In live mode the app exposes configuration controls and a Run Pipeline
button; once the pipeline completes the results are automatically displayed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# Resolve project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils import (
    CLASS_NAMES, CLASS_FULL_NAMES, BEST_MODEL_PATH,
    list_run_dirs, load_json, get_latest_run_dir,
)
from src.metamorphic import MR_REGISTRY, MR_IDS, MR_NAMES

# Page config (must be first Streamlit call)
st.set_page_config(
    page_title="Metamorphic Circuit Breaker",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Parse CLI flags passed after -- separator
_raw_args = sys.argv[1:]
_mode        = "stored"
_data_dir    = "data/isic2019"
_n_samples   = 100
_checkpoint  = str(BEST_MODEL_PATH)

_i = 0
while _i < len(_raw_args):
    if _raw_args[_i] == "--mode" and _i + 1 < len(_raw_args):
        _mode = _raw_args[_i + 1]; _i += 2
    elif _raw_args[_i] == "--data_dir" and _i + 1 < len(_raw_args):
        _data_dir = _raw_args[_i + 1]; _i += 2
    elif _raw_args[_i] == "--n_samples" and _i + 1 < len(_raw_args):
        _n_samples = int(_raw_args[_i + 1]); _i += 2
    elif _raw_args[_i] == "--checkpoint" and _i + 1 < len(_raw_args):
        _checkpoint = _raw_args[_i + 1]; _i += 2
    else:
        _i += 1


# Custom CSS
st.markdown("""
<style>
/* Global dark background */
.stApp { background: #0f0e17; color: #e0e0e0; font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1a1a2e;
    border-right: 1px solid #333355;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #1a1a2e;
    border: 1px solid #333355;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: #a0a0c0 !important; font-size: 0.82rem; }
[data-testid="stMetricValue"] { color: #06d6a0 !important; font-size: 1.7rem; font-weight: 700; }

/* Tab headers */
[data-testid="stTab"] button { color: #a0a0c0; font-weight: 600; }
[data-testid="stTab"] button[aria-selected="true"] {
    color: #7b2d8b; border-bottom: 2px solid #7b2d8b;
}

/* DataFrames */
[data-testid="stDataFrame"] { border-radius: 8px; }

/* Section header helper */
.section-header {
    font-size: 1.05rem; font-weight: 700; color: #7b2d8b;
    border-left: 3px solid #7b2d8b; padding-left: 10px; margin: 16px 0 8px;
}

/* Status pill */
.pill-stable   { background:#06d6a0; color:#0f0e17; padding:2px 9px;
                  border-radius:12px; font-size:0.78rem; font-weight:700; }
.pill-unstable { background:#e94560; color:#fff; padding:2px 9px;
                  border-radius:12px; font-size:0.78rem; font-weight:700; }

/* Info boxes */
.info-box {
    background:#1a1a2e; border:1px solid #333355; border-radius:8px;
    padding:12px 16px; margin:6px 0; font-size:0.9rem;
}
</style>
""", unsafe_allow_html=True)


# Helpers

def _load_run(run_dir: Path):
    """Load predictions CSV and metadata from a run directory."""
    csv  = run_dir / "predictions_table.csv"
    meta = run_dir / "metadata.json"
    df   = pd.read_csv(csv)      if csv.exists()  else None
    md   = load_json(meta)       if meta.exists() else {}
    return df, md


def _image_or_placeholder(path: Path, caption: str = "", width: int = 200):
    """Display an image if it exists, otherwise show a placeholder label."""
    if path.exists():
        st.image(str(path), caption=caption, width=width)
    else:
        st.markdown(
            f"<div class='info-box'>{caption}<br><small>{path.name} not found</small></div>",
            unsafe_allow_html=True,
        )


def _stability_pill(stable: bool) -> str:
    if stable:
        return "<span class='pill-stable'>Stable</span>"
    return "<span class='pill-unstable'>Flipped</span>"


# Sidebar

with st.sidebar:
    st.markdown("## Metamorphic\nCircuit Breaker")
    st.markdown("---")

    # Mode badge
    mode_color = "#06d6a0" if _mode == "stored" else "#ffd166"
    st.markdown(
        f"<div style='background:{mode_color};color:#0f0e17;font-weight:700;"
        f"border-radius:8px;padding:6px 12px;margin-bottom:12px;'>"
        f"Mode: {'Stored' if _mode == 'stored' else 'Live'}</div>",
        unsafe_allow_html=True,
    )

    if _mode == "stored":
        run_dirs = list_run_dirs()
        if run_dirs:
            run_labels = [d.name for d in run_dirs]
            sel_label  = st.selectbox("Select Run", run_labels, index=0)
            selected_run = next(d for d in run_dirs if d.name == sel_label)
        else:
            selected_run = None
            st.warning("No runs found. Run the pipeline first.")
    else:
        selected_run = None   # set after pipeline completes

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **Metamorphic Circuit Breaker**
    Detects transform-brittleness in deployed skin-lesion classifiers.

    **Dataset:** ISIC 2019
    **Backbone:** ConvNeXt-Base
    **MRs:** 8 semantically-preserving transforms
    **Explainability:** GradCAM
    """)

    st.markdown("---")
    st.markdown("### MR Reference")
    for mr in MR_REGISTRY:
        params = ", ".join(f"{k}={v}" for k, v in mr.parameters.items()) or "--"
        st.markdown(
            f"<div class='info-box'><b>{mr.id}</b> {mr.name}<br>"
            f"<small>{mr.description}</small><br>"
            f"<small style='color:#7b2d8b;'>params: {params}</small></div>",
            unsafe_allow_html=True,
        )


# Live Mode: Run Pipeline

if _mode == "live":
    st.title("Live Mode -- Run Metamorphic Testing Pipeline")
    st.markdown(
        "Configure the pipeline below and click **Run Pipeline** to execute "
        "metamorphic testing and generate all results in one step."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        live_data_dir  = st.text_input("Dataset directory", value=_data_dir)
    with col2:
        live_n_samples = st.number_input("Sample count", 10, 500, _n_samples, step=10)
    with col3:
        live_seed      = st.number_input("Random seed", 0, 9999, 42)

    live_checkpoint = st.text_input("Checkpoint path", value=_checkpoint)
    live_skip_gc    = st.checkbox("Skip GradCAM (faster)", value=False)

    st.markdown("---")

    if not Path(live_checkpoint).exists():
        st.error(
            f"Checkpoint not found: `{live_checkpoint}`\n\n"
            "Train the model first:\n```\npython src/train.py --data_dir data/isic2019\n```"
        )

    run_button = st.button("Run Pipeline", type="primary", use_container_width=True)

    if run_button:
        cmd = [
            sys.executable, "run_phase1.py",
            "--data_dir",   live_data_dir,
            "--n_samples",  str(live_n_samples),
            "--seed",       str(live_seed),
            "--checkpoint", live_checkpoint,
        ]
        if live_skip_gc:
            cmd.append("--skip_gradcam")

        with st.status("Running Phase 1 pipeline...", expanded=True) as status:
            st.write("Command:", " ".join(cmd))
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(ROOT),
            )
            out_placeholder = st.empty()
            full_out = ""
            for line in process.stdout:
                full_out += line
                # Show last 4000 chars to avoid overwhelming the UI
                out_placeholder.code(full_out[-4000:], language="text")
            process.wait()

            if process.returncode == 0:
                status.update(label="Pipeline completed!", state="complete")
                latest = get_latest_run_dir()
                if latest:
                    st.session_state["live_run_dir"] = str(latest)
                st.rerun()
            else:
                status.update(label="Pipeline failed -- see output above", state="error")

    # If a live run completed previously in this session, load it
    if "live_run_dir" in st.session_state:
        selected_run = Path(st.session_state["live_run_dir"])
    else:
        st.info("Configure the options above and click **Run Pipeline** to get started.")
        st.stop()


# Guard: need a valid run

if selected_run is None:
    st.title("Metamorphic Circuit Breaker Dashboard")
    st.warning("No run selected. Please select a run from the sidebar, or switch to Live mode.")
    st.stop()

df, meta = _load_run(selected_run)

if df is None:
    st.error(f"predictions_table.csv not found in `{selected_run}`")
    st.stop()

plots_dir   = selected_run / "plots"
gradcam_dir = selected_run / "gradcam"
sample_dir  = selected_run / "sample_images"
trans_dir   = selected_run / "transformed"


# Main Tabs

st.title("Metamorphic Circuit Breaker -- Results Dashboard")
st.caption(
    f"Run: `{selected_run.name}`  |  "
    f"{df['image_id'].nunique()} images  |  "
    f"{len(MR_IDS)} transforms  |  "
    f"{len(df[df['mr_id'] != 'Original'])} total tests"
)

tab_overview, tab_table, tab_gcam, tab_plots, tab_explorer = st.tabs([
    "Overview",
    "Prediction Table",
    "GradCAM Viewer",
    "Charts",
    "Image Explorer",
])


# =======================================
# TAB 1 -- Overview
# =======================================

with tab_overview:
    st.markdown("### Summary Metrics")

    mr_only   = df[df["mr_id"] != "Original"]
    n_total   = len(mr_only)
    n_stable  = int(mr_only["prediction_stable"].sum())
    n_flipped = n_total - n_stable
    n_imgs    = df["image_id"].nunique()
    orig_acc  = mr_only["original_correct"].mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Images Tested",    n_imgs)
    c2.metric("Total Tests",       n_total)
    c3.metric("Stable Preds",      f"{n_stable} ({100*n_stable/n_total:.1f}%)")
    c4.metric("Prediction Flips",  f"{n_flipped} ({100*n_flipped/n_total:.1f}%)")
    c5.metric("Original Accuracy", f"{orig_acc:.1f}%")

    st.markdown("---")
    st.markdown("### Per-Transform Stability")

    per_mr = (
        mr_only.groupby(["mr_id", "mr_name"])["prediction_stable"]
        .agg(["sum", "count"])
        .assign(
            stability_pct=lambda x: 100.0 * x["sum"] / x["count"],
            flip_count   =lambda x: x["count"] - x["sum"],
        )
        .reset_index()
        .rename(columns={"sum": "stable_count"})
    )
    per_mr["mr_id"] = pd.Categorical(per_mr["mr_id"], categories=MR_IDS, ordered=True)
    per_mr = per_mr.sort_values("mr_id")

    for _, row in per_mr.iterrows():
        col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 4])
        with col_a:
            st.markdown(f"**{row['mr_id']}** {row['mr_name']}")
        with col_b:
            st.metric("", f"{row['stability_pct']:.1f}%",
                      delta=f"{int(row['flip_count'])} flipped", delta_color="inverse")
        with col_c:
            label = "High" if row["stability_pct"] >= 80 else "Mid" if row["stability_pct"] >= 60 else "Low"
            st.markdown(label)
        with col_d:
            st.progress(int(row["stability_pct"]))

    st.markdown("---")
    st.markdown("### Run Metadata")
    if meta:
        c1, c2 = st.columns(2)
        with c1:
            st.json({
                "checkpoint": meta.get("checkpoint", "n/a"),
                "n_samples":  meta.get("n_samples",  "n/a"),
                "seed":       meta.get("seed",        "n/a"),
            })
        with c2:
            st.markdown("**ISIC 2019 Class Legend**")
            for abbr, full in CLASS_FULL_NAMES.items():
                st.markdown(f"- **{abbr}** -- {full}")


# =======================================
# TAB 2 -- Prediction Table
# =======================================

with tab_table:
    st.markdown("### Predictions Table")
    st.markdown(
        "Every row is one *(image, transform)* pair. "
        "**Prediction Stable** = True when the transformed image received the same "
        "predicted class as the original (unmodified) image."
    )

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        mr_filter = st.multiselect(
            "Filter by MR", ["All"] + MR_IDS,
            default=["All"],
        )
    with col2:
        stability_filter = st.selectbox(
            "Filter by Stability", ["All", "Stable only", "Flipped only"]
        )
    with col3:
        class_filter = st.multiselect(
            "Filter by Ground-Truth Class", ["All"] + CLASS_NAMES,
            default=["All"],
        )

    display_df = df[df["mr_id"] != "Original"].copy()

    if "All" not in mr_filter and mr_filter:
        display_df = display_df[display_df["mr_id"].isin(mr_filter)]
    if stability_filter == "Stable only":
        display_df = display_df[display_df["prediction_stable"]]
    elif stability_filter == "Flipped only":
        display_df = display_df[~display_df["prediction_stable"]]
    if "All" not in class_filter and class_filter:
        display_df = display_df[display_df["ground_truth_name"].isin(class_filter)]

    st.markdown(f"Showing **{len(display_df)}** rows")

    show_cols = [
        "image_id", "ground_truth_name", "mr_id", "mr_name",
        "original_pred_name", "prediction_name",
        "original_conf", "confidence", "prediction_stable",
        "original_correct", "transformed_correct",
    ]
    st.dataframe(
        display_df[show_cols].reset_index(drop=True),
        use_container_width=True,
        height=500,
    )

    csv_bytes = display_df.to_csv(index=False).encode()
    st.download_button(
        "Download filtered CSV",
        data=csv_bytes,
        file_name="filtered_predictions.csv",
        mime="text/csv",
    )


# =======================================
# TAB 3 -- GradCAM Viewer
# =======================================

with tab_gcam:
    st.markdown("### GradCAM Wrong-Prediction Viewer")
    st.markdown(
        "GradCAM (Gradient-weighted Class Activation Mapping) highlights the "
        "image regions that most influenced the model's prediction. "
        "Comparing the heatmap for the **original** vs the **transformed** image "
        "reveals *why* a transformation caused a prediction flip."
    )

    wrong_dir  = gradcam_dir / "wrong_predictions"
    gcam_files = sorted(wrong_dir.glob("*.png")) if wrong_dir.exists() else []

    if not gcam_files:
        st.info(
            "No GradCAM comparison figures found. Either all predictions were "
            "stable, or the pipeline was run with --skip_gradcam."
        )
    else:
        st.markdown(f"**{len(gcam_files)} wrong-prediction comparisons available**")

        # Extract metadata from filenames: <image_id>_<MR_id>_gradcam.png
        file_labels = {}
        for f in gcam_files:
            parts = f.stem.split("_")
            if len(parts) >= 3:
                mr_id = [p for p in parts if p.startswith("MR")]
                mr_id = mr_id[0] if mr_id else "?"
                file_labels[f] = f"{f.stem}"
            else:
                file_labels[f] = f.name

        # Filter by MR
        mr_gcam_filter = st.multiselect(
            "Filter by transform",
            ["All"] + MR_IDS,
            default=["All"],
        )

        filtered_files = gcam_files
        if "All" not in mr_gcam_filter and mr_gcam_filter:
            filtered_files = [
                f for f in gcam_files
                if any(mr in f.name for mr in mr_gcam_filter)
            ]

        st.markdown(f"Showing {len(filtered_files)} figures")

        # Show 2 per row
        for i in range(0, len(filtered_files), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(filtered_files):
                    fpath = filtered_files[i + j]
                    with col:
                        st.image(str(fpath), caption=file_labels[fpath],
                                 use_container_width=True)


# =======================================
# TAB 4 -- Charts
# =======================================

with tab_plots:
    st.markdown("### Graphical Analysis")

    chart_info = {
        "mr_consistency.png": (
            "Per-MR Prediction Stability",
            "Percentage of images whose prediction **did not change** after each "
            "metamorphic transformation. Green bars (>=80%) indicate robust "
            "transforms; red bars indicate high transform-brittleness."
        ),
        "wrong_counts.png": (
            "Wrong Prediction Counts per Transform",
            "Number of images (out of 100) that received a **different** prediction "
            "after each transform. Longer bars = more brittleness."
        ),
        "overall_pie.png": (
            "Overall Stability Breakdown",
            "All 800 (image x transform) tests categorised as: "
            "Stable & Correct, Stable & Wrong, Unstable to Correct, Unstable & Wrong."
        ),
        "class_brittleness.png": (
            "Per-Class Brittleness Heatmap",
            "Instability rate (%) per ISIC class per transform. Darker red cells "
            "indicate classes that are more susceptible to that transformation."
        ),
    }

    for fname, (title, desc) in chart_info.items():
        fpath = plots_dir / fname
        st.markdown(f"#### {title}")
        st.markdown(desc)
        if fpath.exists():
            st.image(str(fpath), use_container_width=True)
        else:
            st.warning(f"Plot not generated: `{fname}`")
        st.markdown("---")


# =======================================
# TAB 5 -- Image Explorer
# =======================================

with tab_explorer:
    st.markdown("### Image Explorer")
    st.markdown(
        "Select an image from the 100 sampled images and inspect how each of the "
        "8 metamorphic transformations appears alongside the model's prediction."
    )

    image_ids = sorted(df[df["mr_id"] == "Original"]["image_id"].unique().tolist())

    if not image_ids:
        st.warning("No images found in predictions table.")
    else:
        sel_id = st.selectbox("Select image", image_ids)

        # Show ground truth and original prediction
        orig_row = df[(df["image_id"] == sel_id) & (df["mr_id"] == "Original")]
        if not orig_row.empty:
            r = orig_row.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Ground Truth",       r["ground_truth_name"])
            c2.metric("Original Prediction", r["original_pred_name"])
            c3.metric("Confidence",          f"{r['original_conf']:.1%}")

        st.markdown("---")
        st.markdown("#### Transformations & Predictions")

        # Original image
        col_orig, _ = st.columns([1, 3])
        with col_orig:
            orig_img_path = sample_dir / f"{sel_id}.jpg"
            if orig_img_path.exists():
                st.image(str(orig_img_path), caption="Original", width=180)
            else:
                st.caption("Original image not saved")

        st.markdown("---")

        # Grid: 4 columns for 8 transforms
        img_rows = df[(df["image_id"] == sel_id) & (df["mr_id"] != "Original")]

        cols = st.columns(4)
        for idx, (_, row) in enumerate(img_rows.iterrows()):
            col = cols[idx % 4]
            with col:
                t_path = trans_dir / row["mr_id"] / f"{sel_id}.jpg"
                if t_path.exists():
                    img_pil = Image.open(t_path)
                    st.image(img_pil, width=160)
                else:
                    st.markdown(f"*{row['mr_name']} image not saved*")

                stable_label = "[Stable]" if row["prediction_stable"] else "[Flipped]"
                st.markdown(
                    f"**{row['mr_id']}** {row['mr_name']}\n\n"
                    f"Pred: **{row['prediction_name']}** ({row['confidence']:.1%})\n\n"
                    f"{stable_label}"
                )

        # GradCAM for this image (if any wrong predictions)
        wrong_for_img = [
            gradcam_dir / "wrong_predictions" / f"{sel_id}_{mr_id}_gradcam.png"
            for mr_id in MR_IDS
        ]
        existing_gc = [p for p in wrong_for_img if p.exists()]

        if existing_gc:
            st.markdown("---")
            st.markdown("#### GradCAM Comparisons for This Image")
            for gc_path in existing_gc:
                st.image(str(gc_path), use_container_width=True)
        else:
            st.markdown("---")
            st.info("No GradCAM figures for this image (all predictions were stable, or GradCAM was skipped).")
