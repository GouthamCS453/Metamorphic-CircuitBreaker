"""
src/visualize.py — Generate graphical outputs from a predictions_table.csv.

Four chart types
----------------
1. Per-MR Prediction Consistency Bar Chart
   Percentage of images whose prediction was unchanged by each transform.

2. Wrong Prediction Count per MR
   Number of images that flipped to a different class, per transform.

3. Overall Stability Pie Chart
   Stable & correct / stable & wrong / unstable → correct / unstable & wrong.

4. Per-Class Brittleness Heatmap
   Rows = ISIC class, Columns = MR, value = instability rate (%).

All figures use a consistent dark colour theme to match the Streamlit dashboard.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import CLASS_NAMES
from src.metamorphic import MR_IDS, MR_NAMES


# ─── Colour Palette ───────────────────────────────────────────────────────────────

DARK_BG  = "#0f0e17"
PANEL_BG = "#1a1a2e"
GREEN    = "#06d6a0"
YELLOW   = "#ffd166"
RED      = "#e94560"
BLUE     = "#0f3460"
TEXT     = "#e0e0e0"
MID_GREY = "#333355"


def _apply_dark_style() -> None:
    """Apply a consistent dark-theme to matplotlib."""
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    PANEL_BG,
        "axes.edgecolor":    MID_GREY,
        "axes.labelcolor":   TEXT,
        "xtick.color":       TEXT,
        "ytick.color":       TEXT,
        "text.color":        TEXT,
        "grid.color":        "#2a2a4a",
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "font.family":       "DejaVu Sans",
        "legend.facecolor":  PANEL_BG,
        "legend.edgecolor":  MID_GREY,
    })


# ─── Plot 1: Per-MR Consistency Bar Chart ───────────────────────────────────

def plot_mr_consistency(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Bar chart showing prediction stability percentage per metamorphic relation.

    Colour coding:
    * Green  (≥80 %)  — robust to this transform.
    * Yellow (60-80 %) — borderline.
    * Red    (<60 %)   — high brittleness.

    Args:
        df:         Full predictions DataFrame (including "Original" rows).
        output_dir: Directory in which to save the PNG.

    Returns:
        Path to the saved figure.
    """
    _apply_dark_style()

    mr_df = df[df["mr_id"] != "Original"].copy()
    agg = (
        mr_df.groupby(["mr_id", "mr_name"])["prediction_stable"]
        .mean().mul(100).reset_index()
        .rename(columns={"prediction_stable": "stability_pct"})
    )
    agg["mr_id"] = pd.Categorical(agg["mr_id"], categories=MR_IDS, ordered=True)
    agg = agg.sort_values("mr_id")

    colors = [
        GREEN  if v >= 80 else
        YELLOW if v >= 60 else
        RED
        for v in agg["stability_pct"]
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        agg["mr_name"], agg["stability_pct"],
        color=colors, edgecolor=MID_GREY, linewidth=0.8, zorder=3,
    )
    ax.axhline(80, color=RED,    linestyle="--", alpha=0.7, label="80 % threshold")
    ax.axhline(60, color=YELLOW, linestyle=":",  alpha=0.5, label="60 % threshold")
    ax.set_ylim(0, 108)
    ax.set_xlabel("Metamorphic Relation",     fontsize=12, labelpad=8)
    ax.set_ylabel("Prediction Stability (%)", fontsize=12, labelpad=8)
    ax.set_title(
        "Prediction Stability per Metamorphic Transformation",
        fontsize=14, pad=12, fontweight="bold",
    )
    ax.grid(axis="y", zorder=0)
    ax.legend(fontsize=10)
    plt.xticks(rotation=20, ha="right")

    for bar, val in zip(bars, agg["stability_pct"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center", va="bottom", color=TEXT, fontsize=9,
        )

    path = output_dir / "mr_consistency.png"
    plt.savefig(path, dpi=140, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Plot] Saved: {path.name}")
    return path


# ─── Plot 2: Wrong Prediction Count per MR ───────────────────────────────

def plot_wrong_counts(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Horizontal bar chart of prediction flip counts per MR.

    Args:
        df:         Full predictions DataFrame.
        output_dir: Save directory.

    Returns:
        Path to the saved figure.
    """
    _apply_dark_style()

    mr_df = df[df["mr_id"] != "Original"].copy()
    wrong = (
        mr_df[~mr_df["prediction_stable"]]
        .groupby(["mr_id", "mr_name"]).size().reset_index(name="wrong_count")
    )
    mr_template = pd.DataFrame({"mr_id": MR_IDS, "mr_name": MR_NAMES})
    wrong = mr_template.merge(wrong, on=["mr_id", "mr_name"], how="left").fillna(0)
    wrong["wrong_count"] = wrong["wrong_count"].astype(int)
    wrong = wrong.sort_values("wrong_count", ascending=True)

    cmap = plt.colormaps["Reds"]
    max_v = max(wrong["wrong_count"].max(), 1)
    colors = [cmap(0.3 + 0.65 * v / max_v) for v in wrong["wrong_count"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        wrong["mr_name"], wrong["wrong_count"],
        color=colors, edgecolor=MID_GREY, linewidth=0.8, zorder=3,
    )
    ax.set_xlabel("Number of Prediction Flips (out of 100 images)",
                  fontsize=11, labelpad=8)
    ax.set_title("Wrong Predictions Caused by Each Transform",
                 fontsize=13, pad=10, fontweight="bold")
    ax.grid(axis="x", zorder=0)

    for bar, val in zip(bars, wrong["wrong_count"]):
        ax.text(
            val + 0.3, bar.get_y() + bar.get_height() / 2,
            str(int(val)), va="center", color=TEXT, fontsize=10,
        )

    path = output_dir / "wrong_counts.png"
    plt.savefig(path, dpi=140, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Plot] Saved: {path.name}")
    return path


# ─── Plot 3: Overall Stability Pie ──────────────────────────────────────────

def plot_overall_pie(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Pie chart breaking down all (image × MR) test results into four categories.

    Categories:
    * Stable & Correct    — prediction unchanged AND original was correct.
    * Stable & Wrong      — prediction unchanged but original was incorrect.
    * Unstable → Correct  — transform caused a flip that landed on the right class.
    * Unstable & Wrong    — transform caused a flip to the wrong class.

    Args:
        df:         Full predictions DataFrame.
        output_dir: Save directory.

    Returns:
        Path to the saved figure.
    """
    _apply_dark_style()

    mr_df = df[df["mr_id"] != "Original"].copy()

    def _categorise(row):
        if row["prediction_stable"] and row["original_correct"]:
            return "Stable & Correct"
        if row["prediction_stable"] and not row["original_correct"]:
            return "Stable & Wrong"
        if not row["prediction_stable"] and row["transformed_correct"]:
            return "Unstable → Correct"
        return "Unstable & Wrong"

    mr_df["category"] = mr_df.apply(_categorise, axis=1)
    counts = mr_df["category"].value_counts()

    ordered = ["Stable & Correct", "Stable & Wrong",
               "Unstable → Correct", "Unstable & Wrong"]
    labels   = [l for l in ordered if l in counts.index]
    sizes    = [counts[l] for l in labels]
    colors   = [GREEN, BLUE, YELLOW, RED][:len(labels)]

    fig, ax = plt.subplots(figsize=(8, 7))
    wedges, _, autotexts = ax.pie(
        sizes, colors=colors, autopct="%1.1f%%",
        startangle=140, pctdistance=0.80,
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 2},
    )
    for t in autotexts:
        t.set(color="white", fontsize=11, fontweight="bold")

    ax.legend(
        wedges,
        [f"{l}  ({c})".replace("→", "to") for l, c in zip(labels, sizes)],
        loc="upper right", bbox_to_anchor=(1.40, 1.05),
        fontsize=10,
    )
    total = len(mr_df)
    ax.set_title(
        f"Overall Prediction Stability\n"
        f"({df['image_id'].nunique()} images × {len(MR_IDS)} transforms = {total} tests)",
        fontsize=13, pad=12, fontweight="bold",
    )

    path = output_dir / "overall_pie.png"
    plt.savefig(path, dpi=140, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Plot] Saved: {path.name}")
    return path


# ─── Plot 4: Per-Class Brittleness Heatmap ────────────────────────────────

def plot_class_brittleness(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Heatmap of instability rate (%) per (class × transform) combination.

    Rows are ISIC 2019 skin-lesion classes; columns are the 8 MRs.
    Cell values are the percentage of images of that class whose prediction
    flipped under that transform.

    Args:
        df:         Full predictions DataFrame.
        output_dir: Save directory.

    Returns:
        Path to the saved figure.
    """
    _apply_dark_style()

    mr_df = df[df["mr_id"] != "Original"].copy()
    pivot = mr_df.pivot_table(
        index="ground_truth_name",
        columns="mr_name",
        values="prediction_stable",
        aggfunc=lambda x: (1.0 - x.mean()) * 100,
    ).fillna(0)

    # Ensure all class rows present
    for cls in CLASS_NAMES:
        if cls not in pivot.index:
            pivot.loc[cls] = 0.0
    pivot = pivot.reindex(CLASS_NAMES)

    # Ensure all MR columns present
    for name in MR_NAMES:
        if name not in pivot.columns:
            pivot[name] = 0.0
    pivot = pivot[MR_NAMES]

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.heatmap(
        pivot, ax=ax, cmap="Reds",
        annot=True, fmt=".0f",
        annot_kws={"size": 9, "color": "white"},
        linewidths=0.5, linecolor=DARK_BG,
        cbar_kws={"label": "Instability (%)", "shrink": 0.8},
        vmin=0, vmax=100,
    )
    ax.set_title("Per-Class Brittleness per Transform (%)",
                 fontsize=13, pad=10, fontweight="bold")
    ax.set_xlabel("Metamorphic Relation", fontsize=11, labelpad=8)
    ax.set_ylabel("Skin Lesion Class",    fontsize=11, labelpad=8)
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)

    path = output_dir / "class_brittleness.png"
    plt.savefig(path, dpi=140, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Plot] Saved: {path.name}")
    return path


# ─── Convenience Wrapper ────────────────────────────────────────────────────────

def generate_all_plots(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Generate all four plots and return a dict of ``{name: path}``.

    Args:
        df:         Full predictions DataFrame.
        output_dir: Run-level output directory. A ``plots/`` subdirectory
                    is created automatically.

    Returns:
        Dict mapping plot name strings to :class:`pathlib.Path` objects.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    return {
        "mr_consistency":    plot_mr_consistency(df, plots_dir),
        "wrong_counts":      plot_wrong_counts(df, plots_dir),
        "overall_pie":       plot_overall_pie(df, plots_dir),
        "class_brittleness": plot_class_brittleness(df, plots_dir),
    }
