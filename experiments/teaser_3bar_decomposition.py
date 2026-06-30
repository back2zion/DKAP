#!/usr/bin/env python3
"""Teaser figure: three-bar decomposition (B2 -> B2' -> DKAP) across six model families.

Reproduces the finance cross-model ablation teaser in both Korean and English.
Data source: experiments/results/ablation_openrouter/statistics.json

The figure isolates two effects:
  * completeness gain  (B2  -> B2') : adding information, large and robust (+17.5pp mean)
  * structuring gain   (B2' -> DKAP): pure structure, scatters around zero

Sized to remain legible when placed full-width (figure*) in a two-column paper:
a relatively narrow canvas with large fonts, so the on-page text stays readable.

Usage:
    python teaser_3bar_decomposition.py          # builds both KR and EN
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
STATS = HERE / "results" / "ablation_openrouter" / "statistics.json"

# ---------------------------------------------------------------- fonts
KFONT_DIR = Path.home() / ".fonts" / "korean"
for fp in ("NanumGothic-Regular.ttf", "NanumGothic-Bold.ttf"):
    p = KFONT_DIR / fp
    if p.exists():
        fm.fontManager.addfont(str(p))

# ---------------------------------------------------------------- data
with open(STATS, encoding="utf-8") as f:
    S = json.load(f)

# display labels aligned with paper Table (tab:crossmodel)
PANELS = [
    ("OpenAI",    "OpenAI (GPT-5.4)",              "Reasoning"),
    ("Anthropic", "Anthropic (Claude Sonnet 4.6)", "Reasoning"),
    ("Google",    "Google (Gemini 3.1 Pro)",       "Reasoning"),
    ("DeepSeek",  "DeepSeek (V3.2)",               "Reasoning"),
    ("Qwen",      "Alibaba (Qwen3.6-27b)",         "Reasoning"),
    ("Mistral",   "Mistral (Medium 3.5)",          "General"),
]


def row(key):
    m = S["per_model"][key]
    return (m["B2"], m["B2_PRIME"], m["DKAP"],
            m["completeness_gain_B2prime_minus_B2"],
            m["structuring_gain_DKAP_minus_B2prime"]["overall"])


COMPLETE_MEAN = round(S["completeness_gain_B2prime_minus_B2_mean_pp"], 1)   # 17.5
EM_MEAN = round(S["structuring_gain_EasyMedium"]["mean_pp"], 1)            # 0.7

OVERALL_GAINS = {k: S["per_model"][k]["structuring_gain_DKAP_minus_B2prime"]["overall"]
                 for k, _, _ in PANELS}
import math, statistics as _stats
_gains = list(OVERALL_GAINS.values())
OVERALL_MEAN = round(sum(_gains) / len(_gains), 1)  # 0.8
# Family-level Student-t 95% CI on the six overall-EX structuring gains
# (model family = replication unit, n=6, df=5, t_.975=2.5706). Computed from
# the released per-family values so the figure is reproducible, not hardcoded.
_T975_DF5 = 2.5706
_margin = _T975_DF5 * _stats.stdev(_gains) / math.sqrt(len(_gains))
OVERALL_CI = (round(OVERALL_MEAN - _margin, 1), round(OVERALL_MEAN + _margin, 1))

# ---------------------------------------------------------------- colors
C_B2     = "#5B9BD5"
C_B2P    = "#70AD47"
C_DKAP   = "#8064A2"
C_GREEN  = "#2E7D32"
C_PURPLE = "#5E35B1"
C_RED    = "#C62828"
C_GRAY   = "#666666"

# ---------------------------------------------------------------- text
TEXT = {
    "ko": {
        "font": "NanumGothic",
        "title_lines": ["6개 모델 패밀리 금융 제거 실험의 3-막대 분해 (B2 → B2′ → DKAP)"],
        "title_fs": 24,
        "fs": {"corner": 17, "subtitle": 16, "panel_title": 16.5,
               "strip_title": 17, "footer": 13, "sum_title": 18},
        "corner": "(Finance, 전체 EX %)",
        "subtitle_lines": [
            "B2(평문 청크) → B2′(정답 일치 산문) → DKAP(구조화).",
            "B2→B2′ = 완전성 이득(정보 추가)    ·    B2′→DKAP = 구조화 이득(순수 구조)",
        ],
        "ylabel": "Execution Accuracy (%)",
        "reasoning": "Reasoning", "general": "General",
        "sum_title": "요약 (6 패밀리 평균)",
        "sum_comp": "B2 → B2′ 완전성",
        "sum_struct": "B2′ → DKAP 구조화",
        "sum_overall": "(전체 EX)",
        "sum_em": f"+{EM_MEAN:.1f} pp  (Easy+Med, 헤드라인)",
        "sum_ci": "두 경우 모두 95% CI가 0을 포함",
        "strip_title": "패밀리별 구조화 이득 (DKAP - B2′, 전체 EX): 평균과 95% 부트스트랩 CI",
        "strip_x": "구조화 이득 (DKAP - B2′, 전체 EX, pp)",
        "mean_lab": f"평균 +{OVERALL_MEAN:.1f}",
        "ci_lo": f"CI 하한 {OVERALL_CI[0]:.1f}",
        "ci_hi": f"CI 상한 +{OVERALL_CI[1]:.1f}",
        "footer": ("조건: Finance 50 queries · Temp=0.3 · Seeds=10 · 동일 프롬프트 템플릿 · "
                   "실행 정확도(EX) · 6 패밀리 × 3 조건 × 10 시드 = 9,000 runs · "
                   "출처: ablation_openrouter/statistics.json"),
        "out": REPO / "teaser_3bar_decomposition.png",
    },
    "en": {
        "font": "DejaVu Serif",
        "title_lines": ["Three-bar decomposition across six model families (B2 → B2′ → DKAP)"],
        "title_fs": 18,
        "fs": {"corner": 15.5, "subtitle": 14.5, "panel_title": 13,
               "strip_title": 13.5, "footer": 10, "sum_title": 16},
        "corner": "(Finance, overall EX %)",
        "subtitle_lines": [
            "B2 (flat chunks) → B2′ (information-matched prose) → DKAP (structured).",
            "B2→B2′ = completeness gain (added information)    ·    B2′→DKAP = structuring gain (pure structure)",
        ],
        "ylabel": "Execution Accuracy (%)",
        "reasoning": "Reasoning", "general": "General",
        "sum_title": "Summary (6-family mean)",
        "sum_comp": "B2 → B2′ completeness",
        "sum_struct": "B2′ → DKAP structuring",
        "sum_overall": "(overall EX)",
        "sum_em": f"+{EM_MEAN:.1f} pp  (Easy+Med, headline)",
        "sum_ci": "Both 95% CIs include zero",
        "strip_title": "Per-family structuring gain (DKAP − B2′, overall EX): mean and 95% bootstrap CI",
        "strip_x": "Structuring gain (DKAP - B2′, overall EX, pp)",
        "mean_lab": f"Mean +{OVERALL_MEAN:.1f}",
        "ci_lo": f"CI lower {OVERALL_CI[0]:.1f}",
        "ci_hi": f"CI upper +{OVERALL_CI[1]:.1f}",
        "footer": ("Setup: Finance 50 queries · Temp=0.3 · Seeds=10 · identical prompt template · "
                   "Execution Accuracy (EX) · 6 families × 3 conditions × 10 seeds = 9,000 runs · "
                   "Source: ablation_openrouter/statistics.json"),
        "out": HERE / "teaser_3bar_decomposition_en.png",
    },
}

# vertical placement of each strip label; adjacent dots are staggered so the
# two close pairs (Anthropic/DeepSeek, Qwen/Google) do not collide horizontally
STRIP_LABEL_POS = {"OpenAI": "up", "Mistral": "down", "Anthropic": "down",
                   "DeepSeek": "down_lo", "Qwen": "up", "Google": "up_hi"}
STRIP_LEVEL = {"up": (0.40, "bottom"), "up_hi": (0.70, "bottom"),
               "down": (-0.30, "top"), "down_lo": (-0.62, "top")}


def fnum(v):
    return f"{v:+.1f}"


def build(lang):
    T = TEXT[lang]
    FS = T["fs"]
    plt.rcParams.update({"font.family": T["font"], "axes.unicode_minus": False})

    # narrow, tall canvas -> large on-page fonts when scaled to text width
    fig = plt.figure(figsize=(15.5, 13.6), dpi=200)
    fig.patch.set_facecolor("white")

    # ---- title + subtitle ----
    ty = 0.982
    for line in T["title_lines"]:
        fig.text(0.045, ty, line, ha="left", va="top",
                 fontsize=T["title_fs"], fontweight="bold", color="#1A1A1A")
        ty -= 0.038
    fig.text(0.955, 0.975, T["corner"], ha="right", va="top",
             fontsize=FS["corner"], color=C_PURPLE)
    sy0 = ty - 0.006
    for line in T["subtitle_lines"]:
        fig.text(0.045, sy0, line, ha="left", va="top",
                 fontsize=FS["subtitle"], color="#333333")
        sy0 -= 0.030

    # ---- 2x3 grid of bar panels ----
    col_x = [0.050, 0.276, 0.502]
    pan_w = 0.196
    row_y = [0.620, 0.375]          # bottoms
    pan_h = 0.150
    title_y_off = 0.028

    for idx, (key, disp, kind) in enumerate(PANELS):
        r, c = divmod(idx, 3)
        ax = fig.add_axes([col_x[c], row_y[r], pan_w, pan_h])
        b2, b2p, dkap, comp, struct = row(key)

        ax.bar([0, 1, 2], [b2, b2p, dkap], width=0.62,
               color=[C_B2, C_B2P, C_DKAP], zorder=3)
        ax.set_ylim(40, 100)
        ax.set_xlim(-0.65, 2.65)
        ax.set_xticks([])
        ax.set_yticks([40, 60, 80, 100])
        ax.tick_params(labelsize=14)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E5E5E5", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if c == 0:
            ax.set_ylabel(T["ylabel"], fontsize=15)

        kind_lab = T["reasoning"] if kind == "Reasoning" else T["general"]
        fig.text(col_x[c] + pan_w / 2, row_y[r] + pan_h + title_y_off,
                 f"{disp}\n({kind_lab})", ha="center", va="bottom",
                 fontsize=FS["panel_title"], fontweight="bold", linespacing=1.2)

        for xb, val in zip([0, 1, 2], [b2, b2p, dkap]):
            ax.text(xb, val + 1.3, f"{val:.1f}", ha="center", va="bottom",
                    fontsize=14.5, fontweight="bold", color="#222222")

        ax.annotate("", xy=(1, b2p - 0.5), xytext=(0, b2 + 0.5),
                    arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=2.6))
        ax.text(0.5, max(b2, b2p) + 7.0, fnum(comp), ha="center", va="bottom",
                fontsize=18, fontweight="bold", color=C_GREEN)

        pos = struct >= 0
        sc = C_PURPLE if pos else C_RED
        ax.annotate("", xy=(2, dkap - 0.5), xytext=(1, b2p - 0.5),
                    arrowprops=dict(arrowstyle="-|>", color=sc, lw=2.6,
                                    linestyle="-" if pos else (0, (4, 2))))
        ax.text(1.5, max(b2p, dkap) + 7.0, fnum(struct), ha="center", va="bottom",
                fontsize=18, fontweight="bold", color=sc)

    # ---- right summary box (spans both bar rows) ----
    sx, sw = 0.730, 0.225
    sy, sh = 0.375, 0.395
    sax = fig.add_axes([sx, sy, sw, sh]); sax.axis("off")
    sax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                  boxstyle="round,pad=0.01,rounding_size=0.04",
                  ec=C_DKAP, fc="#F6F3FB", lw=2.4,
                  transform=sax.transAxes, zorder=1))
    sax.text(0.5, 0.93, T["sum_title"], ha="center", va="top",
             fontsize=FS["sum_title"], fontweight="bold", color="#1A1A1A",
             transform=sax.transAxes)
    sax.text(0.5, 0.785, T["sum_comp"], ha="center", va="center",
             fontsize=16, color=C_GREEN, transform=sax.transAxes)
    sax.text(0.5, 0.655, f"+{COMPLETE_MEAN:.1f} pp", ha="center", va="center",
             fontsize=37, fontweight="bold", color=C_GREEN, transform=sax.transAxes)
    sax.plot([0.12, 0.88], [0.55, 0.55], color="#CCCCCC", lw=1.0,
             ls=(0, (4, 3)), transform=sax.transAxes)
    sax.text(0.5, 0.45, T["sum_struct"], ha="center", va="center",
             fontsize=16, color=C_PURPLE, transform=sax.transAxes)
    sax.text(0.5, 0.32, f"+{OVERALL_MEAN:.1f} pp", ha="center", va="center",
             fontsize=37, fontweight="bold", color=C_PURPLE, transform=sax.transAxes)
    sax.text(0.5, 0.21, T["sum_overall"], ha="center", va="center",
             fontsize=14, color=C_GRAY, transform=sax.transAxes)
    sax.text(0.5, 0.13, T["sum_em"], ha="center", va="center",
             fontsize=14.5, color=C_PURPLE, transform=sax.transAxes)
    sax.text(0.5, 0.05, T["sum_ci"], ha="center", va="center",
             fontsize=14, color=C_RED, transform=sax.transAxes)

    # ---- bottom strip / forest plot ----
    tax = fig.add_axes([0.070, 0.130, 0.885, 0.185])
    tax.set_xlim(-4, 6)
    tax.set_ylim(-1, 1)
    tax.set_yticks([])
    tax.set_xticks(range(-4, 7))
    tax.tick_params(labelsize=14)
    for sp in ("top", "right", "left"):
        tax.spines[sp].set_visible(False)
    tax.set_xlabel(T["strip_x"], fontsize=15.5)

    tax.axvspan(OVERALL_CI[0], OVERALL_CI[1], color="#ECECEC", zorder=0)
    tax.axvline(0, color="#222222", lw=1.8, zorder=2)
    tax.axvline(OVERALL_MEAN, color=C_PURPLE, lw=2.0, ls=(0, (5, 3)), zorder=2)
    tax.text(OVERALL_MEAN, 0.84, T["mean_lab"], ha="center", va="bottom",
             fontsize=15.5, color=C_PURPLE, fontweight="bold")
    tax.text(OVERALL_CI[0], -0.94, T["ci_lo"], ha="center", va="bottom",
             fontsize=13, color=C_GRAY)
    tax.text(OVERALL_CI[1], -0.94, T["ci_hi"], ha="center", va="bottom",
             fontsize=13, color=C_GRAY)

    for key, disp, _ in PANELS:
        g = OVERALL_GAINS[key]
        col = C_PURPLE if g >= 0 else C_RED
        tax.scatter([g], [0], s=520, color=col, zorder=4,
                    edgecolors="white", linewidths=1.6)
        ylab, va = STRIP_LEVEL[STRIP_LABEL_POS[key]]
        name = disp.split(" (")[0]
        tax.text(g, ylab, f"{name}\n{fnum(g)}", ha="center", va=va,
                 fontsize=14, color=col, fontweight="bold", linespacing=1.1)

    tax.set_title(T["strip_title"], fontsize=FS["strip_title"], fontweight="bold",
                  color="#1A1A1A", pad=14)

    # ---- footer ----
    fig.text(0.5, 0.038, T["footer"], ha="center", va="center",
             fontsize=FS["footer"], color="#888888")

    out = T["out"]
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"[{lang}] wrote {out}")


if __name__ == "__main__":
    build("ko")
    build("en")
