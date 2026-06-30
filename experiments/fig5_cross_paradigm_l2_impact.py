"""
Fig. 5: Cross-Paradigm L2 Impact — B1 vs B2 vs DKAP across 4 domains
IEEE Access format
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ── Data ──
# For JSSP, we invert: "performance" = 100 - gap%, so higher is better
# Random gap=102.5% → perf=-2.5 (clamp to 0)
# MWKR gap=30.9% → perf=69.1
# DKAP estimated gap=10% → perf=90

domains = ["Finance\n(Text-to-SQL)", "Healthcare\n(Clinical QA)", "Public\n(Procurement)", "JSSP\n(RL Scheduling)"]
engines = ["LLM", "LLM", "LLM", "RL+GNN"]

b1   = [0.0,  2.5,  0.0,  0.0]    # EX% or (100 - gap%)
b2   = [54.0, 57.5, 64.0, 69.1]   # MWKR as B2 proxy
dkap = [80.0, 70.0, 64.0, None]   # DKAP; JSSP TBD

x = np.arange(len(domains))
width = 0.22

fig, ax = plt.subplots(figsize=(6, 3.8))

# Bars
bars_b1   = ax.bar(x - width, b1, width, label="B1 (No L2)", color="#EF5350", edgecolor="white", linewidth=0.5)
bars_b2   = ax.bar(x,         b2, width, label="B2 (Partial L2)", color="#FFA726", edgecolor="white", linewidth=0.5)

# DKAP bars (handle None for JSSP)
dkap_vals = [v if v is not None else 0 for v in dkap]
bars_dkap = ax.bar(x + width, dkap_vals, width, label="DKAP (Full L2)", color="#42A5F5", edgecolor="white", linewidth=0.5)

# Mark JSSP DKAP as pending
ax.bar(x[3] + width, 0, width, color="none")
# Hatched placeholder for pending result
ax.bar(x[3] + width, 85, width, color="none", edgecolor="#42A5F5", linewidth=1.0, linestyle="--", hatch="//")
ax.text(x[3] + width, 87, "TBD", ha="center", va="bottom", fontsize=7, color="#1565C0", fontstyle="italic")

# Value labels
for bars, vals in [(bars_b1, b1), (bars_b2, b2)]:
    for bar, val in zip(bars, vals):
        if val > 3:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=7)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=7)

for i, val in enumerate(dkap):
    if val is not None:
        ax.text(bars_dkap[i].get_x() + bars_dkap[i].get_width()/2,
                bars_dkap[i].get_height() + 1,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=7)

# Improvement annotations (B2→DKAP delta)
deltas = ["+26.0pp", "+12.5pp", "+0.0pp", "~+20pp?"]
for i, d in enumerate(deltas):
    y_pos = (dkap[i] if dkap[i] is not None else 85) + 6
    color = "#1565C0" if i < 3 else "#90CAF9"
    ax.text(x[i] + width, y_pos, d, ha="center", va="bottom", fontsize=7,
            color=color, fontweight="bold")

# Engine labels at bottom
for i, eng in enumerate(engines):
    ax.text(x[i], -6, eng, ha="center", va="top", fontsize=7, color="#666666")

ax.set_ylabel("Performance (%)")
ax.set_xticks(x)
ax.set_xticklabels(domains)
ax.set_ylim(-2, 105)
ax.legend(loc="upper left", framealpha=0.9)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.axhline(y=0, color="black", linewidth=0.5)

fig.savefig("fig5_cross_paradigm_l2_impact.png")
fig.savefig("fig5_cross_paradigm_l2_impact.pdf")
print("Saved: fig5_cross_paradigm_l2_impact.png/pdf")
plt.close(fig)

# ── Print summary ──
print("\nCross-Paradigm L2 Impact Summary:")
print(f"  Finance:    B1={b1[0]:.1f}% → B2={b2[0]:.1f}% → DKAP={dkap[0]:.1f}%  (B2-B1: +{b2[0]-b1[0]:.1f}pp, DKAP-B2: +{dkap[0]-b2[0]:.1f}pp)")
print(f"  Healthcare: B1={b1[1]:.1f}% → B2={b2[1]:.1f}% → DKAP={dkap[1]:.1f}%  (B2-B1: +{b2[1]-b1[1]:.1f}pp, DKAP-B2: +{dkap[1]-b2[1]:.1f}pp)")
print(f"  Public:     B1={b1[2]:.1f}% → B2={b2[2]:.1f}% → DKAP={dkap[2]:.1f}%  (B2-B1: +{b2[2]-b1[2]:.1f}pp, DKAP-B2: +{dkap[2]-b2[2]:.1f}pp)")
print(f"  JSSP:       B1=0.0%  → B2={b2[3]:.1f}% → DKAP=TBD       (B2-B1: +{b2[3]-b1[3]:.1f}pp)")
