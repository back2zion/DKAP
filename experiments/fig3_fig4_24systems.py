#!/usr/bin/env python3
"""
Fig. 3: DKS+RSS vs Pipeline Stage Count (n=26, 13 External + 13 Internal)
Fig. 4: Paired Cross-Validation: External vs Internal by Paradigm (13 pairs)

Updated 2026-05-06: 13-pair configuration (was 12-pair).
  Changes from prior version:
  - REMOVED: DIRE (self-reference risk), Logic-LLM (no internal pair after DIRE removed)
  - REMOVED: aistream (non-functional per author), LLaVA (paradigm mismatch after an internal system was reclassified)
  - REPLACED: aistream → Indexis-AI (KG/Graph); LLaVA/Saeyon → MetaGPT/AutoInspect (Role-Based Multi-Agent)
  - ADDED: iyagi (Multi-Agent Creative), EduRAG (Educational AI), bidwatch re-paired under Info Monitoring
  - ADDED externals: MetaGPT (67K★, ICLR 2024), Pulse AI, DeepTutor
  - Paradigm renames: ASR/TTS→Audio Toolkit; VLM→Role-Based Multi-Agent;
    Agentic AI split→Multi-Agent Creative + Info Monitoring; KG/Ontology→Knowledge Graph
  - Internal scores updated to 2026-04-26 conservative re-score
  - ex-GPT outlier (stages=22→7) removed; outlier annotation deleted

IEEE Access format: 300 DPI, serif fonts
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.lines import Line2D

# ── IEEE Access style ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ════════════════════════════════════════════════════════════════
# DATA — 26 Systems (13 External + 13 Internal)
# All internal scores: 2026-04-26 conservative re-score (strict bin-threshold)
# New externals: MetaGPT (source-verified 2026-05-06), Pulse AI, DeepTutor
# ════════════════════════════════════════════════════════════════

PARADIGMS = {
    "Healthcare RAG":         {"marker": "o",  "color": "#2196F3"},
    "Document RAG":           {"marker": "o",  "color": "#64B5F6"},
    "Text-to-SQL":            {"marker": "s",  "color": "#4CAF50"},
    "Domain Fine-Tuning":     {"marker": "s",  "color": "#81C784"},
    "RL+Scheduling":          {"marker": "D",  "color": "#FF9800"},
    "Audio Toolkit":          {"marker": "^",  "color": "#E91E63"},
    "Music Gen":              {"marker": "P",  "color": "#FF5722"},
    "Knowledge Graph":        {"marker": "p",  "color": "#9C27B0"},
    "Role-Based Multi-Agent": {"marker": "H",  "color": "#00BCD4"},
    "Emotion/Voice":          {"marker": ">",  "color": "#795548"},
    "Multi-Agent Creative":   {"marker": "<",  "color": "#3F51B5"},
    "Info Monitoring":        {"marker": "*",  "color": "#607D8B"},
    "Educational AI":         {"marker": "X",  "color": "#009688"},
    "Conversational AI":      {"marker": "v",  "color": "#F06292"},
    "Autonomous Agent":       {"marker": "8",  "color": "#26C6DA"},
    "Enterprise RAG":         {"marker": "d",  "color": "#66BB6A"},
}

# ── External Systems (13) ──
external = {
    # Unchanged from prior verified scoring
    "MedRAG":      {"dks_rss": 10, "stages":  4, "paradigm": "Healthcare RAG"},
    "LlamaIndex":  {"dks_rss": 27, "stages":  6, "paradigm": "Document RAG"},
    "BIRD-SQL":    {"dks_rss": 29, "stages":  7, "paradigm": "Text-to-SQL"},
    "FinSQL":      {"dks_rss": 41, "stages":  7, "paradigm": "Domain Fine-Tuning"},
    "Wheatley":    {"dks_rss": 35, "stages":  8, "paradigm": "RL+Scheduling"},
    "SpeechBrain": {"dks_rss": 41, "stages":  8, "paradigm": "Audio Toolkit"},
    "AudioCraft":  {"dks_rss": 46, "stages":  7, "paradigm": "Music Gen"},
    "GraphRAG":    {"dks_rss": 65, "stages": 10, "paradigm": "Knowledge Graph"},
    "EmotiVoice":  {"dks_rss": 50, "stages":  8, "paradigm": "Emotion/Voice"},
    "CrewAI":      {"dks_rss": 24, "stages":  7, "paradigm": "Multi-Agent Creative"},
    # New externals — source-code-verified 2026-05-06
    "MetaGPT":     {"dks_rss": 41, "stages":  6, "paradigm": "Role-Based Multi-Agent"},
    "Pulse AI":    {"dks_rss": 15, "stages":  6, "paradigm": "Info Monitoring"},
    "DeepTutor":   {"dks_rss": 48, "stages":  8, "paradigm": "Educational AI"},
    # New externals — source-code-verified 2026-05-25 (3 new paradigms)
    "Rasa":        {"dks_rss": 44, "stages":  8, "paradigm": "Conversational AI"},
    "AutoGPT":     {"dks_rss": 56, "stages": 10, "paradigm": "Autonomous Agent"},
    "Haystack":    {"dks_rss": 36, "stages":  7, "paradigm": "Enterprise RAG"},
}

# ── Internal Systems (13) — 2026-04-26 conservative re-score ──
internal = {
    "ClinicalRAG": {"dks_rss": 94, "stages": 16, "paradigm": "Healthcare RAG"},
    "ex-GPT":      {"dks_rss": 45, "stages":  7, "paradigm": "Document RAG"},
    "Lineagis-AI": {"dks_rss": 90, "stages": 12, "paradigm": "Text-to-SQL"},
    "nama":        {"dks_rss": 46, "stages":  8, "paradigm": "Domain Fine-Tuning"},
    "ShipSched":   {"dks_rss": 59, "stages":  9, "paradigm": "RL+Scheduling"},
    "babel-audio": {"dks_rss": 72, "stages":  8, "paradigm": "Audio Toolkit"},
    "Babel-Beats": {"dks_rss": 64, "stages":  8, "paradigm": "Music Gen"},
    "Indexis-AI":  {"dks_rss": 88, "stages": 10, "paradigm": "Knowledge Graph"},
    "AutoInspect": {"dks_rss": 80, "stages": 12, "paradigm": "Role-Based Multi-Agent"},
    "Emotive-Dub": {"dks_rss": 40, "stages":  7, "paradigm": "Emotion/Voice"},
    "iyagi":       {"dks_rss": 65, "stages":  8, "paradigm": "Multi-Agent Creative"},
    "bidwatch":    {"dks_rss": 35, "stages":  5, "paradigm": "Info Monitoring"},
    "EduRAG":       {"dks_rss": 87, "stages": 12, "paradigm": "Educational AI"},
}

# Merge for combined analysis
all_systems = {}
for name, d in external.items():
    all_systems[name] = {**d, "origin": "External"}
for name, d in internal.items():
    all_systems[name] = {**d, "origin": "Internal"}

# ════════════════════════════════════════════════════════════════
# Fig. 3: Scatter — DKS+RSS vs Pipeline Stages (n=26)
# ════════════════════════════════════════════════════════════════

dks_all = [s["dks_rss"] for s in all_systems.values()]
stg_all = [s["stages"] for s in all_systems.values()]

slope, intercept, r_val, p_val, _ = stats.linregress(dks_all, stg_all)
rho, p_rho = stats.spearmanr(dks_all, stg_all)

print("=" * 60)
print("Fig. 3: All Systems (n=29)")
print(f"  Pearson r={r_val:.3f}  p={p_val:.4f}  R²={r_val**2:.3f}")
print(f"  Spearman ρ={rho:.3f}  p={p_rho:.4f}")
print(f"  Regression: y = {slope:.4f}x + {intercept:.2f}")
print("=" * 60)

fig, ax = plt.subplots(figsize=(7.25, 5))

x_line = np.linspace(0, 100, 200)
ax.plot(x_line, slope * x_line + intercept, "-",
        color="#EF9A9A", linewidth=1.2,
        label=f"All (n=26): R²={r_val**2:.3f}, r={r_val:.3f}", zorder=1)

# Data points
for name, sys in all_systems.items():
    para = sys["paradigm"]
    is_ext = sys["origin"] == "External"
    mkr = PARADIGMS[para]["marker"]
    fc = "#2196F3" if is_ext else "#FF9800"
    sz = 70 if is_ext else 80
    ax.scatter(sys["dks_rss"], sys["stages"], marker=mkr, c=fc, s=sz,
               edgecolors="white", linewidths=0.8, zorder=3)

# Labels
label_offsets = {
    "MedRAG": (-1, -0.7), "LlamaIndex": (1.5, -0.5), "BIRD-SQL": (1.5, 0.2),
    "Wheatley": (-9, 0.2), "FinSQL": (-0.5, -0.55), "SpeechBrain": (-12, -0.3),
    "CrewAI": (-7, 0.3), "MetaGPT": (1.5, -0.45),
    "AudioCraft": (1.0, -0.55), "EmotiVoice": (2.5, 0.4), "GraphRAG": (-10, -0.5),
    "Pulse AI": (1.5, 0.3), "DeepTutor": (1.2, 0.95),
    "bidwatch": (1.5, -0.6), "babel-audio": (1.5, 0.2), "ex-GPT": (0.2, 0.45),
    "ShipSched": (1.5, 0.3), "Lineagis-AI": (-12, -0.5),
    "Emotive-Dub": (-12, 0.2), "Babel-Beats": (1.5, -0.5),
    "AutoInspect": (1.5, 0.3), "ClinicalRAG": (-13, 0.3),
    "nama": (0.2, 0.45), "Indexis-AI": (-11, 0.3),
    "iyagi": (1.5, 0.3), "EduRAG": (1.5, -0.5),
    "Rasa": (-2.5, 0.5), "AutoGPT": (1.5, 0.3), "Haystack": (1.5, -0.9),
}

for name, sys in all_systems.items():
    ox, oy = label_offsets.get(name, (1.5, 0.2))
    ax.annotate(name, (sys["dks_rss"], sys["stages"]),
                xytext=(sys["dks_rss"] + ox, sys["stages"] + oy),
                fontsize=6.5, ha="left" if ox > 0 else "right", color="#333333")

# Stats box
textstr = (f"n=29 (29 systems, 13 AI paradigm pairs + 3 unpaired):\n"
           f"  Pearson r = {r_val:.4f}\n"
           f"  p = {p_val:.4f}\n"
           f"  Spearman ρ = {rho:.4f}\n"
           f"  R² = {r_val**2:.4f}")
ax.text(0.03, 0.81, textstr, transform=ax.transAxes, fontsize=7.5,
        verticalalignment="top", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF9C4", edgecolor="#FFEB3B", alpha=0.9))

# Legends
origin_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3", markersize=7, label="External (n=16)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF9800", markersize=7, label="Internal (n=13)"),
]
paradigm_handles = []
for pname, pstyle in PARADIGMS.items():
    paradigm_handles.append(
        Line2D([0], [0], marker=pstyle["marker"], color="w",
               markerfacecolor="#888888", markersize=6, label=pname)
    )

leg1 = ax.legend(handles=origin_handles, loc="upper left", framealpha=0.9, fontsize=7, title="Origin", title_fontsize=7.5)
ax.add_artist(leg1)
ax.legend(handles=paradigm_handles, loc="upper right", framealpha=0.9,
          fontsize=6.0, ncol=4, title="Paradigm", title_fontsize=7.0,
          labelspacing=0.25, handletextpad=0.3, columnspacing=0.6)

ax.set_xlabel("Domain Knowledge Score + Retrieval Sophistication Score (DKS+RSS)")
ax.set_ylabel("Pipeline Stages")
ax.set_xlim(-2, 107)
ax.set_ylim(2, 20)
ax.set_title("DKS+RSS vs Pipeline Stage Count (n=29, 16 AI Paradigms)", fontsize=11, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.3)

fig.savefig("fig3_24systems_scatter.png")
fig.savefig("fig3_24systems_scatter.pdf")
print("Saved: fig3_24systems_scatter.png/pdf")
plt.close(fig)


# ════════════════════════════════════════════════════════════════
# Fig. 4: Paired Cross-Validation (13 pairs)
# ════════════════════════════════════════════════════════════════

pairs = [
    # (paradigm_label, ext_name, ext_dks_rss, ext_stages, int_name, int_dks_rss, int_stages)
    ("Healthcare\nRAG",          "MedRAG",      10,  4,  "ClinicalRAG",  94, 16),
    ("Document\nRAG",            "LlamaIndex",  27,  6,  "ex-GPT",       45,  7),
    ("Text-to-SQL",              "BIRD-SQL",    29,  7,  "Lineagis-AI",  90, 12),
    ("Domain\nFine-Tuning",      "FinSQL",      41,  7,  "nama",         46,  8),
    ("RL+\nScheduling",          "Wheatley",    35,  8,  "ShipSched",    59,  9),
    ("Audio\nToolkit",           "SpeechBrain", 41,  8,  "babel-audio",  72,  8),
    ("Music Gen",                "AudioCraft",  46,  7,  "Babel-Beats",  64,  8),
    ("Knowledge\nGraph",         "GraphRAG",    65, 10,  "Indexis-AI",   88, 10),
    ("Role-Based\nMulti-Agent",  "MetaGPT",     41,  6,  "AutoInspect",  80, 12),
    ("Emotion/\nVoice",          "EmotiVoice",  50,  8,  "Emotive-Dub",  40,  7),
    ("Multi-Agent\nCreative",    "CrewAI",      24,  7,  "iyagi",        65,  8),
    ("Info\nMonitoring",         "Pulse AI",    15,  6,  "bidwatch",     35,  5),
    ("Educational\nAI",          "DeepTutor",   48,  8,  "EduRAG",        87, 12),
]

fig2, ax2 = plt.subplots(figsize=(7.5, 6.0))

x = np.arange(len(pairs))
width = 0.35

ext_vals   = [p[2] for p in pairs]
int_vals   = [p[5] for p in pairs]
ext_stages = [p[3] for p in pairs]
int_stages = [p[6] for p in pairs]
ext_names  = [p[1] for p in pairs]
int_names  = [p[4] for p in pairs]
categories = [p[0] for p in pairs]

bars_ext = ax2.bar(x - width/2, ext_vals, width, label="External DKS+RSS",
                   color="#64B5F6", edgecolor="white", linewidth=0.5)
bars_int = ax2.bar(x + width/2, int_vals, width, label="Internal DKS+RSS",
                   color="#FFB74D", edgecolor="white", linewidth=0.5)

# Stage count labels above bars
for i in range(len(pairs)):
    ax2.text(bars_ext[i].get_x() + bars_ext[i].get_width()/2,
             bars_ext[i].get_height() + 1,
             f"{ext_stages[i]}st", ha="center", va="bottom",
             fontsize=8, color="#1565C0", fontweight="bold")
    ax2.text(bars_int[i].get_x() + bars_int[i].get_width()/2,
             bars_int[i].get_height() + 1,
             f"{int_stages[i]}st", ha="center", va="bottom",
             fontsize=8, color="#E65100", fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(categories, fontsize=8, rotation=45, ha="right")
ax2.set_ylabel("DKS+RSS Score")
ax2.set_ylim(0, 105)
ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)
ax2.grid(axis="y", linestyle="--", alpha=0.3)
ax2.set_title("Paired Cross-Validation: External vs Internal by Paradigm (13 pairs, n=26)",
              fontsize=11, fontweight="bold")

ax2.text(0.5, -0.22,
         "Numbers above bars = Pipeline Stages (st). Blue = External open-source, Orange = Internal proprietary.",
         transform=ax2.transAxes, ha="center", fontsize=8, color="#666666")

fig2.tight_layout()
fig2.savefig("fig4_paired_crossvalidation.png")
fig2.savefig("fig4_paired_crossvalidation.pdf")
print("Saved: fig4_paired_crossvalidation.png/pdf")
plt.close(fig2)


# ════════════════════════════════════════════════════════════════
# Summary Statistics
# ════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("DATASET SUMMARY (n=29, 13 pairs + 3 unpaired external)")
print("=" * 60)

ext_dks = [s["dks_rss"] for s in external.values()]
ext_stg = [s["stages"] for s in external.values()]
r_ext, p_ext = stats.pearsonr(ext_dks, ext_stg)
rho_ext, p_rho_ext = stats.spearmanr(ext_dks, ext_stg)
n_ext = len(external)
print(f"External-only (n={n_ext}):")
print(f"  Pearson  r = {r_ext:.4f}, p = {p_ext:.4f}, R² = {r_ext**2:.4f}")
print(f"  Spearman ρ = {rho_ext:.4f}, p = {p_rho_ext:.4f}")

int_dks = [s["dks_rss"] for s in internal.values()]
int_stg = [s["stages"] for s in internal.values()]
r_int, p_int = stats.pearsonr(int_dks, int_stg)
rho_int, p_rho_int = stats.spearmanr(int_dks, int_stg)
print(f"\nInternal-only (n=13):")
print(f"  Pearson  r = {r_int:.4f}, p = {p_int:.4f}, R² = {r_int**2:.4f}")
print(f"  Spearman ρ = {rho_int:.4f}, p = {p_rho_int:.4f}")

print(f"\nPaired Cross-Validation (13 pairs):")
int_higher = sum(1 for p in pairs if p[5] > p[2])
ext_higher = sum(1 for p in pairs if p[2] > p[5])
print(f"  Internal DKS+RSS > External: {int_higher}/13 pairs")
print(f"  External DKS+RSS > Internal: {ext_higher}/13 pairs")

diffs = [p[5] - p[2] for p in pairs]
from scipy.stats import wilcoxon
w_stat, w_p = wilcoxon(diffs)
print(f"  Wilcoxon signed-rank: W={w_stat:.1f}, p={w_p:.4f}")
print(f"  Mean difference (Internal - External): {np.mean(diffs):.1f}")

print(f"\nPer-pair results:")
for p in pairs:
    direction = "Int>" if p[5] > p[2] else "Ext>"
    print(f"  {p[0].replace(chr(10),' '):22s} {direction}  {p[1]:12s}({p[2]:3d}) vs {p[4]:12s}({p[5]:3d})  Δ={p[5]-p[2]:+4d}")
print("=" * 60)
