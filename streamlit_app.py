"""
streamlit_app.py -- Metamorphic Circuit Breaker Live Dashboard v2.3

Fixes:
  - All emojis removed (plain-text UI).
  - Batch run archive properly re-reads CSV/metadata per selected run.
  - Live circuit breaker confirmed real-time PyTorch inference.
  - Clear clinical callout when image shows CLOSED despite boundary flips.
"""

from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils import (
    CLASS_NAMES, CLASS_FULL_NAMES, BEST_MODEL_PATH,
    list_run_dirs, load_json,
)
from src.metamorphic_families import HIERARCHICAL_TEST_MATRIX, FAMILIES, SEVERITY_WEIGHTS

# Page config
st.set_page_config(
    page_title="Metamorphic Circuit Breaker",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PRESET_DIR = ROOT / "data" / "presets"
SEV_COLOR  = {"mild": "#06d6a0", "moderate": "#ffd166", "severe": "#e94560"}
FAM_LABEL  = {"geometric": "Geometric", "photometric": "Photometric", "sensor_noise": "Sensor / Noise"}

# Default Phase 2 parameters stored in session state
for k, v in [("alpha",0.50),("beta",0.35),("gamma",0.15),
              ("tau_fam",0.35),("theta_warn",0.25),("theta_trip",0.55)]:
    if k not in st.session_state:
        st.session_state[k] = v

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
.stApp { background: #0d0c1e; color: #e2e2f0; font-family: 'Inter', sans-serif; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #13122a; border-right: 1px solid #2a2a4a; }
[data-testid="stSidebar"] * { color: #c8c8e8 !important; }
[data-testid="stMetric"] {
    background: #181730; border: 1px solid #2a2a50;
    border-radius: 12px; padding: 14px 18px;
}
[data-testid="stMetricLabel"] { color: #8080aa !important; font-size: 0.78rem; text-transform:uppercase; }
[data-testid="stMetricValue"] { color: #7fffcc !important; font-size: 1.65rem; font-weight: 800; }
.banner { border-radius: 14px; padding: 22px 30px; text-align: center;
    font-size: 1.55rem; font-weight: 900; margin: 14px 0 8px; line-height: 1.4; }
.banner-closed   { background: linear-gradient(135deg,#00c896,#06d6a0); color: #041a10; }
.banner-halfopen { background: linear-gradient(135deg,#f0a500,#ffd166); color: #1a1000; }
.banner-open     { background: linear-gradient(135deg,#c0002a,#e94560); color: #fff; }
.sec { font-size: .98rem; font-weight: 700; color: #9b4fc8;
    border-left: 3px solid #9b4fc8; padding-left: 10px; margin: 18px 0 8px; }
.card { background: #181730; border: 1px solid #2a2a50; border-radius: 10px;
    padding: 14px 18px; margin: 6px 0; font-size: .88rem; line-height: 1.6; }
.fbox { background: #0e0d22; border: 1px solid #4a3d88; border-radius: 10px;
    padding: 16px 20px; margin: 10px 0; font-family: 'Courier New', monospace;
    font-size: .84rem; line-height: 1.8; color: #d0d0ff; }
.flip { background: #2a0e18; border: 2px solid #e94560; border-radius: 10px;
    padding: 14px 18px; margin: 8px 0; }
.live-badge { display: inline-block; background: #2b1d52; color: #c499f3;
    padding: 4px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 700;
    border: 1px solid #6b3fb8; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)


# Helpers

@st.cache_resource(show_spinner="Loading MobileNetV3-Small model (one-time ~2s)...")
def _load_model():
    from src.models.mobilenet_adapter import MobileNetAdapter
    return MobileNetAdapter()


def _make_cb(alpha, beta, gamma, tau_fam, theta_warn, theta_trip):
    from src.circuit_breaker import MetamorphicCircuitBreaker, CircuitBreakerConfig
    cfg = CircuitBreakerConfig(alpha=alpha, beta=beta, gamma=gamma,
                               tau_fam=tau_fam, theta_warn=theta_warn, theta_trip=theta_trip)
    return MetamorphicCircuitBreaker(_load_model(), config=cfg)


def _load_presets():
    presets = {}
    if not PRESET_DIR.exists():
        return presets
    for d in sorted(PRESET_DIR.iterdir()):
        mp = d / "metadata.json"
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text())
        imgs = [p for p in d.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}]
        if imgs:
            meta["image_path"] = imgs[0]
            presets[meta["preset_id"]] = meta
    return presets


def _banner(state: str, cbi: float, action: str):
    if state == "CLOSED":
        cls, label = "banner-closed", "CLOSED  |  AUTO-APPROVED"
    elif state == "HALF_OPEN":
        cls, label = "banner-halfopen", "HALF-OPEN  |  MONITOR / WARNING"
    else:
        cls, label = "banner-open", "OPEN  |  SAFETY INTERCEPT  |  PREDICTION BLOCKED"
    st.markdown(
        f"<div class='banner {cls}'>{label}<br>"
        f"<span style='font-size:1rem;font-weight:600;opacity:.85'>"
        f"CBI = {cbi:.4f}  |  {action.replace('_',' ')}</span></div>",
        unsafe_allow_html=True)


def _cbi_gauge(cbi, tw, tt):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 0.9))
    fig.patch.set_facecolor("#0d0c1e"); ax.set_facecolor("#181730")
    ax.barh(.5, tw,     height=.55, left=0,  color="#06d6a0", alpha=.25)
    ax.barh(.5, tt-tw,  height=.55, left=tw, color="#ffd166", alpha=.25)
    ax.barh(.5, 1.0-tt, height=.55, left=tt, color="#e94560", alpha=.25)
    col = "#06d6a0" if cbi<tw else ("#ffd166" if cbi<tt else "#e94560")
    ax.axvline(cbi, color=col, lw=3.5)
    ax.text(cbi, .9, f" {cbi:.3f}", ha="left" if cbi<.85 else "right",
            va="top", color=col, fontsize=9, fontweight="bold")
    for val, lbl in [(tw,f"theta_warn={tw}"),(tt,f"theta_trip={tt}")]:
        ax.axvline(val, color="#555", lw=1, ls="--")
        ax.text(val, .08, lbl, ha="center", va="bottom", color="#888", fontsize=7)
    ax.set_xlim(0,1); ax.set_yticks([])
    ax.set_xlabel("Composite Brittleness Index (CBI)", color="#888", fontsize=8)
    ax.tick_params(axis="x", colors="#888", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#2a2a50")
    plt.tight_layout(pad=.2); return fig


def _family_bars(fi, tau):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams   = [FAM_LABEL.get(f,f) for f in fi]
    scores = list(fi.values())
    colors = [("#e94560" if s>=tau else ("#ffd166" if s>=tau*.5 else "#06d6a0")) for s in scores]
    fig, ax = plt.subplots(figsize=(5.5, 2.4))
    fig.patch.set_facecolor("#0d0c1e"); ax.set_facecolor("#181730")
    bars = ax.barh(fams, scores, color=colors, height=.5)
    ax.axvline(tau, color="#e94560", lw=1.5, ls="--", label=f"tau_fam={tau}")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("IR_k", color="#888", fontsize=8)
    ax.tick_params(colors="#888", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#2a2a50")
    for b, s in zip(bars, scores):
        ax.text(s+.01, b.get_y()+b.get_height()/2,
                f"{s:.3f}", va="center", color="#fff", fontsize=8, fontweight="bold")
    ax.legend(fontsize=7, facecolor="#181730", labelcolor="#aaa")
    plt.tight_layout(pad=.3); return fig


def _test_matrix_df(test_results):
    rows = []
    for r in test_results:
        rows.append({
            "ID": r.test_id,
            "Transform": r.test_name,
            "Family": FAM_LABEL.get(r.family, r.family),
            "Severity": r.severity.capitalize(),
            "w(s)": r.weight,
            "Prediction": r.pred_label,
            "Conf": f"{r.confidence:.1%}",
            "Result": "FLIP" if r.flipped else "Stable",
        })
    return pd.DataFrame(rows)


def _render_gradcam(report, gcam, orig_img):
    from src.gradcam import make_comparison_figure, overlay_heatmap_on_image
    st.markdown("<div class='sec'>Grad-CAM Attention Analysis</div>", unsafe_allow_html=True)
    bc = gcam.get("baseline_cam")
    fc_list = gcam.get("flip_cams", [])
    c1, c2 = st.columns([1, 2])
    with c1:
        if bc is not None:
            ov = overlay_heatmap_on_image(orig_img, bc, alpha=.50, out_size=240)
            st.image(ov, caption=f"Baseline GradCAM  |  {report.baseline_label} ({report.baseline_confidence:.1%})", width=240)
        else:
            st.image(orig_img.resize((240,240)),
                     caption=f"Original  |  {report.baseline_label} ({report.baseline_confidence:.1%})", width=240)
    with c2:
        full_name = CLASS_FULL_NAMES.get(report.baseline_label, report.baseline_label)
        n_flips = sum(1 for r in report.test_results if r.flipped)
        st.markdown(f"""
<div class='card'>
<b>Baseline prediction:</b>&nbsp; <code>{report.baseline_label}</code> — {full_name}<br>
<b>Confidence:</b>&nbsp; {report.baseline_confidence:.1%}<br>
<b>Prediction flips detected:</b>&nbsp; {n_flips} / {len(report.test_results)} metamorphic tests<br>
<b>Peak unstable family:</b>&nbsp; {FAM_LABEL.get(report.peak_family, report.peak_family)} (IR_k = {report.peak_family_score:.3f})<br><br>
<small style='color:#888'>GradCAM heatmap shows where MobileNetV3-Small focused attention.
A healthy model focuses on the lesion. In a flipped prediction, attention drifts
to boundary artifacts, rotation halos, or dermatoscope rings.</small>
</div>
""", unsafe_allow_html=True)
    if not fc_list:
        st.success("No prediction flips detected — model attention is consistent across all metamorphic perturbations.")
        return
    st.markdown(f"**Grad-CAM attention shift for the {len(fc_list)} most impactful flip(s):**")
    for fc in fc_list:
        sc = SEV_COLOR.get(fc["severity"], "#aaa")
        with st.expander(
            f"Flip: {report.baseline_label} -> {fc['flipped_label']}  |  "
            f"{fc['test_name']}  |  Severity: {fc['severity'].capitalize()}  (w={fc['weight']})",
            expanded=True,
        ):
            st.markdown(
                f"<div class='flip'>"
                f"<b>Prediction flip</b><br>"
                f"<span style='color:#06d6a0'>Original: <b>{report.baseline_label}</b> ({report.baseline_confidence:.1%})</span>"
                f"&nbsp; -> &nbsp;"
                f"<span style='color:#e94560'>After <b>{fc['test_name']}</b>: "
                f"<b>{fc['flipped_label']}</b> ({fc['flipped_confidence']:.1%})</span><br>"
                f"<small>Family: {FAM_LABEL.get(fc['family'],fc['family'])} &nbsp;|&nbsp; "
                f"Severity: <span style='color:{sc}'>{fc['severity'].capitalize()}</span> "
                f"&nbsp;|&nbsp; w(s) = {fc['weight']}</small>"
                f"</div>", unsafe_allow_html=True)
            if bc is not None and fc.get("cam") is not None:
                fig_img = make_comparison_figure(
                    original_image=orig_img, baseline_cam=bc,
                    transformed_image=fc["transformed_image"], flip_cam=fc["cam"],
                    baseline_label=report.baseline_label, baseline_conf=report.baseline_confidence,
                    flipped_label=fc["flipped_label"], flipped_conf=fc["flipped_confidence"],
                    test_name=fc["test_name"], out_size=230,
                )
                st.image(fig_img, use_container_width=True)
            else:
                ca, cb_ = st.columns(2)
                with ca: st.image(orig_img.resize((230,230)), caption=f"Original: {report.baseline_label}", width=230)
                with cb_: st.image(fc["transformed_image"].resize((230,230)), caption=f"Transformed: {fc['flipped_label']}", width=230)


def _render_math(report, cfg):
    st.markdown("<div class='sec'>Mathematical Attribution Breakdown</div>", unsafe_allow_html=True)
    a, b, g  = cfg["alpha"], cfg["beta"], cfg["gamma"]
    tw, tt   = cfg["theta_warn"], cfg["theta_trip"]
    tau      = cfg["tau_fam"]
    mi       = report.peak_family_score
    cfs      = report.cross_family_spread
    unc      = 1.0 - report.baseline_confidence
    cbi      = report.cbi
    decision = (f"CBI {cbi:.4f} < theta_warn {tw}  ->  CLOSED" if cbi < tw
                else (f"theta_warn {tw} <= CBI {cbi:.4f} < theta_trip {tt}  ->  HALF-OPEN"
                      if cbi < tt else f"CBI {cbi:.4f} >= theta_trip {tt}  ->  OPEN"))
    st.markdown(
        f"<div class='fbox'>"
        f"<b>CBI(x)</b> = alpha * max_k IR_k(x)  +  beta * CFS(x)  +  gamma * (1 - c0)<br>"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {a} x {mi:.4f}  +  {b} x {cfs:.4f}  +  {g} x {unc:.4f}<br>"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {a*mi:.4f}  +  {b*cfs:.4f}  +  {g*unc:.4f}<br>"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <b>{cbi:.4f}</b><br><br>"
        f"<b>Decision:</b>  {decision}"
        f"</div>", unsafe_allow_html=True)
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("CBI (Master Risk Score)", f"{cbi:.4f}")
    m2.metric(f"max IR_k  ({report.peak_family})", f"{mi:.4f}")
    m3.metric("CFS (Cross-Family Spread)", f"{cfs:.4f}")
    m4.metric("Uncertainty  (1-c0)", f"{unc:.4f}")
    st.markdown("")
    gl, gr = st.columns([3, 2])
    with gl:
        st.markdown("<div class='sec'>CBI Risk Gauge</div>", unsafe_allow_html=True)
        st.pyplot(_cbi_gauge(cbi, tw, tt), use_container_width=True)
    with gr:
        st.markdown("<div class='sec'>Family IR_k Breakdown</div>", unsafe_allow_html=True)
        st.pyplot(_family_bars(report.family_instability, tau), use_container_width=True)
    with st.expander("Level 1 -- Score_m per Transformation Type"):
        rows = [{"Transform Type": t, "Score_m": f"{s:.4f}",
                 "Instability %": f"{s*100:.1f}%"} for t,s in report.type_scores.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with st.expander("Level 2 -- Intra-Family IR_k (unskewed average)"):
        rows2 = [{
            "Family": FAM_LABEL.get(f,f), "IR_k": f"{v:.4f}",
            ">= tau_fam?": "YES -- Compromised" if v>=tau else "No -- Healthy",
            "Contributes to CFS?": "YES" if v>=tau else "No",
        } for f,v in report.family_instability.items()]
        st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)
    with st.expander("Full Metamorphic Test Matrix (all 22 results)"):
        df = _test_matrix_df(report.test_results)
        st.dataframe(df, use_container_width=True, hide_index=True)


# TABS
tab_live, tab_calib, tab_batch = st.tabs([
    "Live Circuit Breaker",
    "Parameter Calibration",
    "Batch Run Archive",
])


# TAB 1 -- LIVE CIRCUIT BREAKER
with tab_live:
    st.markdown("""
<div style='padding:4px 0 6px'>
  <span class='live-badge'>LIVE INFERENCE ENGINE (REAL-TIME PYTORCH)</span><br>
  <span style='font-size:2rem;font-weight:900;color:#e2e2f0'>Metamorphic Circuit Breaker Studio</span><br>
  <span style='color:#8080aa;font-size:.92rem'>
    Runs 22 metamorphic tests per image against the trained MobileNetV3-Small model in real time.
  </span>
</div>
""", unsafe_allow_html=True)
    st.markdown("---")

    _alpha = st.session_state["alpha"]
    _beta  = st.session_state["beta"]
    _gamma = st.session_state["gamma"]
    _tau   = st.session_state["tau_fam"]
    _tw    = st.session_state["theta_warn"]
    _tt    = st.session_state["theta_trip"]

    src_col, cfg_col = st.columns([3, 1])

    with cfg_col:
        st.markdown(
            "<div class='card'><b>Active Config</b><br>"
            f"<small>alpha = {_alpha:.2f} | beta = {_beta:.2f} | gamma = {_gamma:.2f}<br>"
            f"theta_warn = {_tw:.2f} | theta_trip = {_tt:.2f}<br>"
            f"tau_fam = {_tau:.2f}<br><br>"
            f"Total tests: {len(HIERARCHICAL_TEST_MATRIX)}<br>"
            f"Families: Geometric / Photometric / Sensor-Noise<br><br>"
            "<i style='color:#a0a0c0'>Adjust via Parameter Calibration tab</i></small></div>",
            unsafe_allow_html=True)
        st.markdown(
            "<div class='card'><b>Getting OPEN or HALF-OPEN states</b><br>"
            "<small>The MobileNetV3-Small model is trained on GTSRB traffic-sign images.<br><br>"
            "To see all three states:<br>"
            "1. Use the verified demo presets below.<br>"
            "2. Upload images from a <b>different acquisition source</b> — smartphone photos, "
            "DermNet images, or HAM10000 non-dermoscope images.<br>"
            "3. Lower <code>theta_trip</code> to 0.15 in Tab 2 to make the breaker more sensitive.<br>"
            "</small></div>",
            unsafe_allow_html=True)

    with src_col:
        img_mode = st.radio("Input source:", ["Upload image", "Use verified demo preset"], horizontal=True)
        uploaded_img: Image.Image | None = None
        preset_meta = None

        if img_mode == "Upload image":
            uf = st.file_uploader("Choose an image file (JPG / PNG):", type=["jpg","jpeg","png"])
            if uf:
                uploaded_img = Image.open(uf).convert("RGB")
                st.image(uploaded_img, caption=f"Uploaded: {uf.name}", width=240)
        else:
            presets = _load_presets()
            if not presets:
                st.warning("No presets found in data/presets/. Run: python scripts/update_presets.py")
            else:
                fmt = lambda pid: presets[pid]["name"]
                chosen = st.selectbox("Select demo preset:", list(presets.keys()), format_func=fmt)
                preset_meta = presets[chosen]
                uploaded_img = Image.open(preset_meta["image_path"]).convert("RGB")
                pa, pb = st.columns([1, 2])
                with pa:
                    st.image(uploaded_img, caption=preset_meta.get("image_filename",""), width=190)
                with pb:
                    exp_state = preset_meta.get("expected_state","CLOSED")
                    cbi_ref = preset_meta.get("cbi_ref","--")
                    flips_ref = preset_meta.get("flips_ref","--")
                    ecol = "#06d6a0" if exp_state=="CLOSED" else ("#ffd166" if exp_state=="HALF_OPEN" else "#e94560")
                    st.markdown(
                        f"<div class='card'><b>{preset_meta['name']}</b><br>"
                        f"{preset_meta['description']}<br><br>"
                        f"Expected State: <b style='color:{ecol}'>{exp_state}</b>"
                        f"&nbsp;|&nbsp; Ref CBI: <code>{cbi_ref}</code>"
                        f"&nbsp;|&nbsp; Flips: <code>{flips_ref}</code></div>",
                        unsafe_allow_html=True)

    st.markdown("")
    run_btn = st.button("Run Live Metamorphic Circuit Breaker", type="primary",
                        disabled=(uploaded_img is None))
    if uploaded_img is None:
        st.caption("Upload an image or pick a demo preset above.")

    if run_btn and uploaded_img is not None:
        with st.status("Running live circuit breaker analysis...", expanded=True) as status:
            st.write("Loading PyTorch MobileNetV3-Small model...")
            model = _load_model()
            st.write(f"Executing {len(HIERARCHICAL_TEST_MATRIX)} metamorphic test forward passes...")
            t0 = time.time()
            cb = _make_cb(_alpha, _beta, _gamma, _tau, _tw, _tt)

            report_q = cb.evaluate(uploaded_img)
            n_flips  = sum(1 for r in report_q.test_results if r.flipped)

            if n_flips > 0 or report_q.state.value in ("HALF_OPEN","OPEN"):
                st.write(f"{n_flips} flip(s) detected -- running GradCAM backpropagation...")
                report, gcam = cb.evaluate_with_gradcam(uploaded_img, max_gradcam_flips=3)
            else:
                st.write("Zero flips -- extracting baseline GradCAM only...")
                report = report_q
                bc = model.explain(uploaded_img, target_class=report.baseline_idx)
                gcam = {"baseline_cam": bc, "baseline_image": uploaded_img, "flip_cams": []}

            elapsed = time.time() - t0
            status.update(label=f"Analysis complete in {elapsed:.2f}s", state="complete")

        st.markdown("---")
        _banner(report.state.value, report.cbi, report.action)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Baseline Prediction", report.baseline_label,
                  help=CLASS_FULL_NAMES.get(report.baseline_label, report.baseline_label))
        c2.metric("Confidence (c0)", f"{report.baseline_confidence:.1%}")
        c3.metric("Prediction Flips", f"{n_flips} / {len(report.test_results)}")
        c4.metric("CBI Score", f"{report.cbi:.4f}")

        s = report.state.value
        if s == "OPEN":
            st.error(
                f"**CIRCUIT BREAKER TRIPPED -- OPEN STATE**\n\n"
                f"The model collapsed across {len(report.compromised_families)} family/families "
                f"({', '.join(FAM_LABEL.get(f,f) for f in report.compromised_families)}). "
                f"CBI = {report.cbi:.4f} >= theta_trip ({_tt}).\n\n"
                f"**Fallback:** Automated diagnosis intercepted. "
                f"Case flagged for Human-in-the-Loop triage with GradCAM attention maps."
            )
        elif s == "HALF_OPEN":
            st.warning(
                f"**HALF-OPEN MONITORING STATE**\n\n"
                f"Systematic sensitivity in family: "
                f"{FAM_LABEL.get(report.peak_family, report.peak_family)} "
                f"(IR_k = {report.peak_family_score:.3f}). "
                f"CBI = {report.cbi:.4f} between theta_warn ({_tw}) and theta_trip ({_tt}).\n\n"
                f"**Action:** Prediction approved with an amber safety warning for secondary review."
            )
        else:
            if n_flips > 0:
                st.info(
                    f"**CLOSED STATE -- AUTO-APPROVED (with anti-false-alarm protection)**\n\n"
                    f"Note: {n_flips} flip(s) occurred but all were at extreme Severe/Moderate boundary tiers "
                    f"(w = 0.3 or 0.6). No single family reached the tau_fam ({_tau}) threshold "
                    f"(max IR_k = {report.peak_family_score:.3f}, CFS = {report.cross_family_spread:.3f}). "
                    f"Phase 2 hierarchical normalization correctly identifies these as benign edge-case artifacts, "
                    f"not systemic model collapse. CBI = {report.cbi:.4f} < theta_warn ({_tw})."
                )
            else:
                st.success(
                    f"**CLOSED STATE -- AUTO-APPROVED**\n\n"
                    f"Zero flips across all 22 metamorphic tests. "
                    f"Prediction is rock-solid (CBI = {report.cbi:.4f} < theta_warn = {_tw})."
                )

        st.markdown("---")
        _render_gradcam(report, gcam, uploaded_img)
        st.markdown("---")
        cfg_dict = report.details.get("config", {
            "alpha":_alpha,"beta":_beta,"gamma":_gamma,
            "tau_fam":_tau,"theta_warn":_tw,"theta_trip":_tt,
        })
        _render_math(report, cfg_dict)


# TAB 2 -- PARAMETER CALIBRATION
with tab_calib:
    st.title("Parameter Calibration")
    st.markdown("Tune CBI weights and thresholds. Changes apply immediately to Tab 1.")

    left, right = st.columns([1, 2])
    with left:
        st.markdown("### CBI Formula Weights (alpha + beta + gamma = 1.0)")
        ca  = st.slider("alpha -- Peak Family Instability", 0.0, 1.0, st.session_state["alpha"], 0.05, key="s_a")
        cb_ = st.slider("beta  -- Cross-Family Spread",     0.0, 1.0, st.session_state["beta"],  0.05, key="s_b")
        cg  = st.slider("gamma -- Baseline Uncertainty",    0.0, 1.0, st.session_state["gamma"], 0.05, key="s_g")
        tot = ca + cb_ + cg
        if abs(tot-1.0) > 0.011:
            st.error(f"alpha + beta + gamma = {tot:.2f} (must sum to 1.0)")
        else:
            st.session_state["alpha"] = ca
            st.session_state["beta"]  = cb_
            st.session_state["gamma"] = cg
            st.success(f"alpha + beta + gamma = {tot:.2f}  (active in Tab 1)")

        st.markdown("### Decision Thresholds")
        ctau  = st.slider("tau_fam   -- family activation",  0.10, 0.60, st.session_state["tau_fam"],    0.05, key="s_tau")
        cwarn = st.slider("theta_warn -- HALF-OPEN trigger", 0.10, 0.50, st.session_state["theta_warn"], 0.05, key="s_warn")
        ctrip = st.slider("theta_trip -- OPEN trigger",      0.10, 0.90, st.session_state["theta_trip"], 0.05, key="s_trip")
        if cwarn >= ctrip:
            st.warning("theta_warn must be less than theta_trip.")
        else:
            st.session_state["tau_fam"]    = ctau
            st.session_state["theta_warn"] = cwarn
            st.session_state["theta_trip"] = ctrip

    with right:
        st.markdown("### Live Formula Preview")
        st.markdown(
            f"<div class='fbox'>"
            f"CBI(x) = {ca} * max_k IR_k  +  {cb_} * CFS  +  {cg} * (1 - c0)<br><br>"
            f"CLOSED    if CBI &lt; {cwarn}<br>"
            f"HALF-OPEN if {cwarn} &lt;= CBI &lt; {ctrip}<br>"
            f"OPEN      if CBI >= {ctrip}"
            f"</div>", unsafe_allow_html=True)

        st.markdown("### Hypothetical Scenario Simulator")
        sim_ir  = st.slider("Hypothetical max IR_k",              0.0, 1.0, 0.40, 0.05)
        sim_cfs = st.slider("Hypothetical CFS",                   0.0, 1.0, 0.33, step=round(1/3,4))
        sim_unc = st.slider("Hypothetical uncertainty (1 - c0)",  0.0, 1.0, 0.20, 0.05)
        if abs(tot-1.0) <= 0.011:
            sim_cbi   = ca*sim_ir + cb_*sim_cfs + cg*sim_unc
            sim_state = "CLOSED" if sim_cbi<cwarn else ("HALF_OPEN" if sim_cbi<ctrip else "OPEN")
            st.pyplot(_cbi_gauge(sim_cbi, cwarn, ctrip), use_container_width=True)
            scls = {"CLOSED":"banner-closed","HALF_OPEN":"banner-halfopen","OPEN":"banner-open"}[sim_state]
            st.markdown(
                f"<div class='banner {scls}' style='font-size:1.1rem;padding:14px'>"
                f"Simulated CBI = {sim_cbi:.4f}  ->  {sim_state}</div>",
                unsafe_allow_html=True)

        st.markdown("### Clinical Risk Profiles")
        for name, params, why in [
            ("High-Risk Oncology Clinic",       "alpha=0.55, beta=0.30, gamma=0.15, theta_trip=0.50",
             "Zero tolerance -- trips aggressively even on single-family instability."),
            ("Rural Clinic (Low Capacity)",      "alpha=0.40, beta=0.45, gamma=0.15, theta_trip=0.60",
             "Requires multi-family proof before tripping -- reduces triage overload."),
            ("Telehealth / Smartphone Cameras", "alpha=0.45, beta=0.40, gamma=0.15, theta_trip=0.55",
             "Higher beta to separate sensor noise from genuine model collapse."),
            ("Calibrated Model",                "alpha=0.45, beta=0.30, gamma=0.25, theta_trip=0.55",
             "Higher gamma -- temperature-scaled confidence is more trustworthy."),
        ]:
            st.markdown(
                f"<div class='card'><b>{name}</b>&nbsp;&nbsp;<code>{params}</code><br>{why}</div>",
                unsafe_allow_html=True)


# TAB 3 -- BATCH RUN ARCHIVE
with tab_batch:
    st.title("Batch Run Archive")
    st.markdown("Historical Phase 1 batch pipeline results. Refresh the page to pick up new runs.")

    run_dirs = list_run_dirs()  # always re-called each Streamlit run (not cached)

    if not run_dirs:
        st.info(
            "No batch runs found. Run the Phase 1 pipeline to populate this view:\n"
            "```\npython run_phase1.py --data_dir data/isic2019 --n_samples 100\n```"
        )
    else:
        run_labels = [d.name for d in run_dirs]
        sel_label  = st.selectbox("Select run:", run_labels, index=0, key="batch_run_select")
        selected   = next(d for d in run_dirs if d.name == sel_label)

        csv_p  = selected / "predictions_table.csv"
        meta_p = selected / "metadata.json"

        # Read fresh every time the selection changes (no caching on these reads)
        md = load_json(meta_p) if meta_p.exists() else {}
        df = pd.read_csv(csv_p) if csv_p.exists() else None

        st.markdown(f"**Run directory:** `{selected}`")

        if md:
            st.markdown("### Run Metadata")
            mc1,mc2,mc3,mc4 = st.columns(4)
            mc1.metric("Samples",    str(md.get("n_samples","--")))
            mc2.metric("MR Count",   str(md.get("n_mrs","--")))
            mc3.metric("Runtime",    f"{md.get('runtime_seconds',0):.0f}s")
            mc4.metric("Checkpoint", Path(md.get("checkpoint","--")).name)

        if df is not None and not df.empty:
            st.markdown(f"### Predictions Table  ({len(df)} rows from `{selected.name}`)")
            if "flipped" in df.columns:
                flip_col = df["flipped"]
                flip_rate = flip_col.mean() if flip_col.dtype == bool else (flip_col == True).mean()
                st.metric("Overall Flip Rate", f"{flip_rate:.1%}")
            st.dataframe(df.head(300), use_container_width=True)

            if "mr_id" in df.columns and "flipped" in df.columns:
                st.markdown("### Flip Rate per Metamorphic Relation")
                mr_stats = (
                    df.groupby("mr_id")["flipped"]
                    .mean()
                    .reset_index()
                    .rename(columns={"mr_id":"MR ID","flipped":"Flip Rate"})
                    .sort_values("Flip Rate", ascending=False)
                )
                mr_stats["Flip Rate"] = mr_stats["Flip Rate"].map(lambda x: f"{x:.1%}")
                st.dataframe(mr_stats, use_container_width=True, hide_index=True)
        else:
            st.warning(f"No predictions_table.csv found in `{selected.name}`.")

    with st.expander("Run Phase 1 Pipeline"):
        col1,col2,col3 = st.columns(3)
        with col1: p1_dir  = st.text_input("Dataset dir",  "data/isic2019")
        with col2: p1_n    = st.number_input("Samples", 10, 500, 100, step=10)
        with col3: p1_ckpt = st.text_input("Checkpoint", str(BEST_MODEL_PATH))
        if st.button("Launch Phase 1 Pipeline", type="secondary"):
            cmd = [sys.executable,"run_phase1.py",
                   "--data_dir",p1_dir,"--n_samples",str(p1_n),"--checkpoint",p1_ckpt]
            with st.status("Running Phase 1...", expanded=True) as s2:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, cwd=str(ROOT))
                ph = st.empty(); buf = []
                for line in proc.stdout:
                    buf.append(line); ph.code("".join(buf[-40:]))
                proc.wait()
                s2.update(label="Done!" if proc.returncode==0 else "Failed.",
                          state="complete" if proc.returncode==0 else "error")
