#!/usr/bin/env python3
"""Fig 5: System selection funnel (13 internal + 16 external -> 29 systems).

Numbers are taken verbatim from main.tex Section III.A:
  Internal: 40+ industrial PoC systems (2024-2025) -> stratified sampling
            along 4 dimensions -> 13 systems across 7 industries.
  External: open-source candidates -> inclusion criteria (i)-(iv) -> 16 systems
            across 16 paradigms (>=5 non-LLM after selection).
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

INT_BLUE = "#1f77b4"
EXT_ORANGE = "#ff7f0e"
MERGE_GREY = "#455A64"
FILL_INT = "#E3F2FD"
FILL_EXT = "#FFF3E0"
FILL_MERGE = "#ECEFF1"

fig, ax = plt.subplots(figsize=(7.25, 4.4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")


def box(x, y, w, h, text, edge, fill, fs=8.5, bold_first=True):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.08",
        edgecolor=edge, facecolor=fill, linewidth=1.4, zorder=2))
    lines = text.split("\n")
    if bold_first:
        ax.text(x + w / 2, y + h - 0.18, lines[0], ha="center", va="top",
                fontsize=fs, fontweight="bold", color="#212121", zorder=3)
        if len(lines) > 1:
            ax.text(x + w / 2, y + h - 0.18 - 0.42 * fs / 8.5, "\n".join(lines[1:]),
                    ha="center", va="top", fontsize=fs - 0.8, color="#37474F",
                    zorder=3, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color="#212121", zorder=3, linespacing=1.3)


def arrow(x0, y0, x1, y1, color, label=None, label_dx=0.18):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
        linewidth=1.5, color=color, zorder=1))
    if label:
        ax.text(x1 + label_dx if x0 == x1 else (x0 + x1) / 2, (y0 + y1) / 2,
                label, ha="left" if x0 == x1 else "center", va="center",
                fontsize=7.3, color="#37474F", linespacing=1.25)


# ---- Internal track (left) ----
ax.text(2.45, 9.7, "Internal (industrial)", ha="center", fontsize=10,
        fontweight="bold", color=INT_BLUE)
box(0.7, 7.9, 3.5, 1.3,
    "40+ industrial PoC systems\ndeveloped 2024–2025,\nseven+ client domains",
    INT_BLUE, FILL_INT)
arrow(2.45, 7.75, 2.45, 6.05, INT_BLUE,
      "stratified sampling on 4 dimensions:\ndomain / AI engine / data modality /\npipeline complexity (5–16 stages)")
box(0.7, 4.7, 3.5, 1.2,
    "13 internal systems\nseven industries,\nLLM · RL · VLM · ML engines",
    INT_BLUE, FILL_INT)

# ---- External track (right) ----
ax.text(7.55, 9.7, "External (open-source)", ha="center", fontsize=10,
        fontweight="bold", color=EXT_ORANGE)
box(5.8, 7.9, 3.5, 1.3,
    "Open-source AI systems\nconsidered (candidates listed in\nsupplementary material)",
    EXT_ORANGE, FILL_EXT)
arrow(7.55, 7.75, 7.55, 6.05, EXT_ORANGE,
      "(i) peer-reviewed or ≥1,000 stars\n(ii) source sufficient for mining\n(iii) one system per paradigm\n(iv) ≥4 non-LLM systems")
box(5.8, 4.7, 3.5, 1.2,
    "16 external systems\nsixteen paradigms,\nfive non-LLM",
    EXT_ORANGE, FILL_EXT)

# ---- Merge ----
arrow(2.45, 4.55, 4.6, 3.15, INT_BLUE)
arrow(7.55, 4.55, 5.4, 3.15, EXT_ORANGE)
box(3.0, 1.55, 4.0, 1.5,
    "29 analyzed systems\n16 AI paradigms · 13 paradigm-\nmatched internal–external pairs",
    MERGE_GREY, FILL_MERGE, fs=9.5)
ax.text(5.0, 1.05,
        "Architecture Mining Protocol (Section III-B) applied uniformly to all 29",
        ha="center", fontsize=7.5, color="#546E7A", style="italic")

fig.savefig("/home/babelai/DKAP/experiments/fig5_selection_funnel.png")
print("saved fig5_selection_funnel.png")
