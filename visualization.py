"""
Visualization Module
Reproduces key figures from the paper:
  - Fig 4: Difficulty vs Discrimination scatter plot
  - Fig 5: Point-Biserial distribution histogram
  - Fig 6: Metric correlation matrix heatmap
  - Bloom distribution pie chart
  - Pipeline efficiency bar chart
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from typing import List, Dict, Optional


# ─────────────────────────────────────────────
# Style Setup
# ─────────────────────────────────────────────

def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.grid": True,
        "grid.alpha": 0.4,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })


# ─────────────────────────────────────────────
# Fig 4: Difficulty vs Discrimination (scatter)
# ─────────────────────────────────────────────

def plot_difficulty_discrimination(mcqs: List[Dict], save_path: Optional[str] = None):
    """Reproduces Fig 4 from the paper."""
    set_style()
    p_vals = [m["p_value"] for m in mcqs]
    d_vals = [m["d_value"] for m in mcqs]
    q_vals = [m.get("quality_score", 0.45) for m in mcqs]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Shade ideal zones
    ax.axvspan(0.30, 0.70, alpha=0.08, color="green", label="Ideal difficulty (0.30–0.70)")
    ax.axhline(0.40, color="red", linestyle="--", alpha=0.6, linewidth=1, label="Good threshold (D=0.40)")

    sc = ax.scatter(p_vals, d_vals, c=q_vals, cmap="RdYlGn",
                    s=60, alpha=0.8, vmin=0.35, vmax=0.55, edgecolors="k", linewidths=0.3)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Quality Score (Q)", fontsize=9)

    ax.set_xlabel("Item Difficulty (P-value)")
    ax.set_ylabel("Discrimination Index (D-value)")
    ax.set_title("Difficulty vs. Discrimination Map (Fig 4)")
    ax.set_xlim(0.25, 0.75)
    ax.set_ylim(0.15, 0.90)
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────
# Fig 5: Point-Biserial Distribution
# ─────────────────────────────────────────────

def plot_pointbiserial_distribution(mcqs: List[Dict], save_path: Optional[str] = None):
    """Reproduces Fig 5 from the paper."""
    set_style()
    r_pbs = [m["r_pb"] for m in mcqs]

    fig, ax = plt.subplots(figsize=(7, 5))
    n_bins = min(15, len(r_pbs))
    ax.hist(r_pbs, bins=n_bins, color="#4C72B0", edgecolor="white",
            alpha=0.85, rwidth=0.9)

    ax.axvline(0.20, color="red", linestyle="--", linewidth=1.5,
               label="Acceptable threshold (r_pb = 0.20)")
    ax.axvline(0.30, color="orange", linestyle=":", linewidth=1.2,
               label="Good threshold (r_pb = 0.30)")

    pct_above_20 = np.mean(np.array(r_pbs) >= 0.20) * 100
    pct_above_30 = np.mean(np.array(r_pbs) >= 0.30) * 100

    ax.set_xlabel("Point-Biserial Correlation (r_pb)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Point-Biserial Distribution (Fig 5)\n"
                 f"≥0.20: {pct_above_20:.1f}%  |  ≥0.30: {pct_above_30:.1f}%")
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────
# Fig 6: Metric Correlation Matrix
# ─────────────────────────────────────────────

def plot_correlation_matrix(mcqs: List[Dict], save_path: Optional[str] = None):
    """Reproduces Fig 6 from the paper."""
    set_style()

    df = pd.DataFrame({
        "Pseudo_F1": [m.get("pseudo_f1", 0) for m in mcqs],
        "Novelty": [m.get("novelty_index", 0) for m in mcqs],
        "Distraction_Index": [m.get("distraction_index", 0) for m in mcqs],
        "P_Value": [m.get("p_value", 0) for m in mcqs],
        "D_Value": [m.get("d_value", 0) for m in mcqs],
    })

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.zeros_like(corr, dtype=bool)

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        center=0,
        square=True,
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 10, "weight": "bold"}
    )

    ax.set_title("Metric Correlation Matrix (Fig 6)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────
# Bloom Distribution Chart
# ─────────────────────────────────────────────

def plot_bloom_distribution(mcqs: List[Dict], save_path: Optional[str] = None):
    """Show Bloom-level distribution of generated MCQs."""
    set_style()
    from collections import Counter

    bloom_counts = Counter(m.get("bloom_level", "Unknown") for m in mcqs)
    labels = list(bloom_counts.keys())
    sizes = list(bloom_counts.values())
    colors = ["#4C72B0", "#DD8452", "#55A868"][:len(labels)]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=140,
        pctdistance=0.75, labeldistance=1.1
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")

    ax.set_title(f"Bloom Taxonomy Distribution\n(n={len(mcqs)} MCQs)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────
# Pipeline Efficiency Bar Chart
# ─────────────────────────────────────────────

def plot_pipeline_efficiency(pipeline_stats: Dict, save_path: Optional[str] = None):
    """Visualize the funnel of items through the pipeline (Table 7 equivalent)."""
    set_style()

    stages = list(pipeline_stats.keys())
    counts = list(pipeline_stats.values())
    pass_rates = [c / max(counts[0], 1) * 100 for c in counts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar: absolute counts
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(stages)))
    bars = ax1.barh(stages[::-1], counts[::-1], color=colors[::-1], edgecolor="white")
    ax1.set_xlabel("Number of Items")
    ax1.set_title("Pipeline Item Counts (Table 7)")
    for bar, count in zip(bars, counts[::-1]):
        ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                 str(count), va="center", fontsize=9)

    # Bar: pass rates
    ax2.bar(stages, pass_rates, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax2.set_ylabel("Cumulative Pass Rate (%)")
    ax2.set_title("Cumulative Pass Rate per Stage")
    ax2.set_xticklabels(stages, rotation=20, ha="right")
    ax2.axhline(20.1, color="red", linestyle="--", alpha=0.5, label="Final retention 20.1%")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────
# Quality Metrics Distribution
# ─────────────────────────────────────────────

def plot_quality_metrics(mcqs: List[Dict], save_path: Optional[str] = None):
    """Show distribution of all 4 quality metrics."""
    set_style()

    metrics = {
        "Pseudo-F1": [m.get("pseudo_f1", 0) for m in mcqs],
        "Novelty Index": [m.get("novelty_index", 0) for m in mcqs],
        "Distraction Index": [m.get("distraction_index", 0) for m in mcqs],
        "Quality Score (Q)": [m.get("quality_score", 0) for m in mcqs],
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    thresholds = {"Pseudo-F1": 0.45, "Novelty Index": 0.55,
                  "Distraction Index": 0.20, "Quality Score (Q)": 0.42}

    for ax, (name, vals) in zip(axes, metrics.items()):
        ax.hist(vals, bins=15, color="#55A868", edgecolor="white", alpha=0.85)
        if name in thresholds:
            ax.axvline(thresholds[name], color="red", linestyle="--",
                       linewidth=1.5, label=f"Threshold={thresholds[name]}")
        ax.set_title(name)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        pct_pass = np.mean(np.array(vals) >= thresholds.get(name, 0)) * 100
        ax.text(0.98, 0.95, f"{pct_pass:.1f}% pass",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="darkgreen", fontweight="bold")

    plt.suptitle("Quality Metric Distributions (Phase 4)", fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────
# All-in-one plotting
# ─────────────────────────────────────────────

def generate_all_plots(mcqs: List[Dict], output_dir: str = "."):
    """Generate and save all figures."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n  Generating visualizations...")
    plot_difficulty_discrimination(mcqs, f"{output_dir}/fig4_difficulty_discrimination.png")
    plot_pointbiserial_distribution(mcqs, f"{output_dir}/fig5_pointbiserial.png")
    plot_correlation_matrix(mcqs, f"{output_dir}/fig6_correlation_matrix.png")
    plot_bloom_distribution(mcqs, f"{output_dir}/bloom_distribution.png")
    plot_quality_metrics(mcqs, f"{output_dir}/quality_metrics.png")
    print("  ✓ All figures saved.\n")
