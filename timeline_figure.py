#!/usr/bin/env python3
"""
Activity Timeline Chart
- Reads consolidated JSON files (one per video)
- 4 rows: Base, Zero-Shot, Hybrid, SLAM
- Background = ground truth (action_label)
- Dot = each model's prediction field
- Light theme
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ==========================
# CONFIG
# ==========================
INPUT_FOLDER  = Path("predicted_vs_actual_consolidated")
OUTPUT_FOLDER = Path("timeline_charts_final")
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Row label → prediction field in the consolidated JSON
ROW_DEFINITIONS = [
    ("Base",       "model_prediction"),
    ("Zero-Shot",  "zero_shot_prediction"),
    ("SLAM",       "SLAM_prediction"),
    ("Hybrid",     "hybrid_prediction"),
]

# ==========================
# COLOR MAP
# ==========================
ACTION_COLORS = {
    "go upstair":   "#1b9e77",
    "walking":      "#d95f02",
    "standing":     "#7570b3",
    "go downstair": "#666666",
    "sitting":      "#e6ab02",
}
DEFAULT_COLOR = "#aaaaaa"

def get_color(label):
    if label is None:
        return DEFAULT_COLOR
    return ACTION_COLORS.get(str(label).strip().lower(), DEFAULT_COLOR)

# ==========================
# PLOT FUNCTION
# ==========================
def plot_video(json_path: Path):
    with open(json_path) as f:
        data = json.load(f)

    data = sorted(data, key=lambda x: x["clip_start_utc_ns"])

    t0     = data[0]["clip_start_utc_ns"]
    starts = np.array([(d["clip_start_utc_ns"] - t0) / 1e9 for d in data])
    ends   = np.array([(d["clip_end_utc_ns"]   - t0) / 1e9 for d in data])
    widths = ends - starts
    true_labels    = [d["action_label"] for d in data]
    total_duration = ends[-1]

    # Only include rows where the field exists
    active_rows = [
        (label, field)
        for label, field in ROW_DEFINITIONS
        if any(field in d for d in data)
    ]
    n_rows = len(active_rows)

    # Layout constants
    ROW_H   = 1.0
    ROW_GAP = 0.18
    LABEL_W = 3.8

    total_h = n_rows * ROW_H + (n_rows - 1) * ROW_GAP
    fig_w   = max(20, (total_duration + LABEL_W) * 0.15)
    fig_h   = max(4.5, total_h * 1.5 + 2.8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(-LABEL_W, total_duration)
    ax.set_ylim(-ROW_GAP * 1.5, total_h + ROW_GAP * 1.5)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#333333")

    import matplotlib.ticker as ticker
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x) // 60}:{int(x) % 60:02d}" if int(x) >= 60 else f"{int(x)}"
    ))
    ax.tick_params(axis="x", labelsize=16, colors="#111111", width=1.5, length=5)
    ax.set_xlabel("Time (seconds)", fontsize=16, fontweight="bold",
                  color="#111111", labelpad=10)

    # Dot size scales with segment width
    seg_pts  = (widths.mean() / (total_duration + LABEL_W)) * fig_w * 72
    dot_size = max(60, seg_pts ** 1.6)

    for row_idx, (row_label, field) in enumerate(active_rows):
        y_center = (n_rows - 1 - row_idx) * (ROW_H + ROW_GAP) + ROW_H / 2

        # Row label on the left
        ax.text(
            -LABEL_W * 0.5, y_center,
            row_label,
            ha="center", va="center",
            fontsize=14, fontweight="bold", color="#111111",
            clip_on=False
        )

        # Divider line between rows
        if row_idx < n_rows - 1:
            div_y = y_center - ROW_H / 2 - ROW_GAP / 2
            ax.axhline(div_y, color="#cccccc", linewidth=1.0, zorder=1)

        for d, start, width, true_label in zip(data, starts, widths, true_labels):
            # Background: ground truth color
            ax.barh(
                y_center, width, left=start,
                height=ROW_H,
                color=get_color(true_label),
                align="center", linewidth=0, alpha=0.9, zorder=2
            )

            # Dot: this row's model prediction
            pred = d.get(field)
            if pred is not None:
                ax.scatter(
                    start + width / 2, y_center,
                    s=dot_size,
                    color=get_color(pred),
                    edgecolors="#111111",
                    linewidths=1.5,
                    zorder=4
                )

    # Title
    title = json_path.stem.replace("_", " ").title()
    ax.set_title(
        title,
        fontsize=18, fontweight="bold", color="#111111",
        pad=14, loc="center"
    )

    # Legend
    unique_actions = sorted(ACTION_COLORS.keys())
    bg_patches = [
        mpatches.Patch(color=get_color(l), label=l)
        for l in unique_actions
    ]
    dot_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=get_color(l),
                   markeredgecolor="#111111", markeredgewidth=1.2,
                   markersize=18, label=l, linestyle="None")
        for l in unique_actions
    ]

    leg1 = fig.legend(
        handles=bg_patches,
        title="Ground Truth (background)",
        loc="lower left", ncol=1,
        fontsize=18, framealpha=0.95, title_fontsize=19,
        bbox_to_anchor=(0.01, -0.15),
        labelcolor="#111111"
    )
    leg1.get_title().set_color("#111111")
    leg1.get_title().set_fontweight("bold")

    leg2 = fig.legend(
        handles=dot_handles,
        title="Predicted (dot)",
        loc="lower right", ncol=1,
        fontsize=18, framealpha=0.95, title_fontsize=19,
        bbox_to_anchor=(0.99, -0.15),
        labelcolor="#111111"
    )
    leg2.get_title().set_color("#111111")
    leg2.get_title().set_fontweight("bold")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out_path = OUTPUT_FOLDER / f"{json_path.stem}_timeline.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ Saved: {out_path}")

# ==========================
# MAIN
# ==========================
json_files = sorted(INPUT_FOLDER.glob("*.json"))
if not json_files:
    print(f"No JSON files found in {INPUT_FOLDER}")
else:
    for jf in json_files:
        print(f"Processing: {jf.name}")
        plot_video(jf)
    print(f"\n📁 All charts saved to: {OUTPUT_FOLDER}/")