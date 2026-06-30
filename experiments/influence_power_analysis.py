#!/usr/bin/env python3
"""
Cook's distance influence analysis + power analysis for n=16 external systems.
Results to be cited in main.tex.
"""

import numpy as np
from scipy import stats
from scipy.stats import t as t_dist
import warnings
warnings.filterwarnings("ignore")

# ── External Systems (n=16) ──
external = {
    "MedRAG":      {"dks_rss": 10, "stages":  4},
    "LlamaIndex":  {"dks_rss": 27, "stages":  6},
    "BIRD-SQL":    {"dks_rss": 29, "stages":  7},
    "FinSQL":      {"dks_rss": 41, "stages":  7},
    "Wheatley":    {"dks_rss": 35, "stages":  8},
    "SpeechBrain": {"dks_rss": 41, "stages":  8},
    "AudioCraft":  {"dks_rss": 46, "stages":  7},
    "GraphRAG":    {"dks_rss": 65, "stages": 10},
    "EmotiVoice":  {"dks_rss": 50, "stages":  8},
    "CrewAI":      {"dks_rss": 24, "stages":  7},
    "MetaGPT":     {"dks_rss": 41, "stages":  6},
    "Pulse AI":    {"dks_rss": 15, "stages":  6},
    "DeepTutor":   {"dks_rss": 48, "stages":  8},
    "Rasa":        {"dks_rss": 44, "stages":  8},
    "AutoGPT":     {"dks_rss": 56, "stages": 10},
    "Haystack":    {"dks_rss": 36, "stages":  7},
}

names = list(external.keys())
x = np.array([external[n]["dks_rss"] for n in names], dtype=float)
y = np.array([external[n]["stages"]  for n in names], dtype=float)
n = len(x)

# ── 1. OLS regression ──
X = np.column_stack([np.ones(n), x])
beta = np.linalg.lstsq(X, y, rcond=None)[0]
y_hat = X @ beta
resid = y - y_hat
SSE = np.sum(resid**2)
s2 = SSE / (n - 2)
p = 2  # intercept + slope

# ── 2. Cook's distance ──
H = X @ np.linalg.inv(X.T @ X) @ X.T   # hat matrix
h = np.diag(H)                           # leverage
cook_d = (resid**2 * h) / (p * s2 * (1 - h)**2)

# Threshold: 4/n  (common convention)
threshold_4n = 4 / n
# Threshold: F(0.5; p, n-p) = 0.5 percentile of F distribution (classic)
from scipy.stats import f as f_dist
threshold_f = f_dist.ppf(0.5, p, n - p)

print("=" * 60)
print(f"Cook's Distance Analysis  (n={n})")
print(f"Threshold 4/n = {threshold_4n:.4f}")
print(f"Threshold F(0.5;{p},{n-p}) = {threshold_f:.4f}")
print("=" * 60)
print(f"{'System':<14} {'x':>5} {'y':>4} {'h':>6} {'resid':>7} {'Cook_D':>8}  flag")
print("-" * 60)

influential = []
for i, nm in enumerate(names):
    flag = ""
    if cook_d[i] > threshold_4n:
        flag = "* (>4/n)"
        influential.append(nm)
    if cook_d[i] > threshold_f:
        flag += " ** (>F0.5)"
    print(f"{nm:<14} {x[i]:>5.0f} {y[i]:>4.0f} {h[i]:>6.3f} {resid[i]:>7.3f} {cook_d[i]:>8.4f}  {flag}")

print()
print(f"Max Cook's D: {cook_d.max():.4f} ({names[np.argmax(cook_d)]})")
print(f"Influential (>4/n={threshold_4n:.3f}): {influential if influential else 'None'}")

# ── 3. Leave-one-out sensitivity ──
print()
print("=" * 60)
print("Leave-one-out sensitivity (Pearson r)")
print("=" * 60)
r_full, _ = stats.pearsonr(x, y)
print(f"Full n=16: r = {r_full:.4f}")
print()
max_drop = 0
max_drop_sys = ""
for i, nm in enumerate(names):
    xi = np.delete(x, i)
    yi = np.delete(y, i)
    ri, _ = stats.pearsonr(xi, yi)
    drop = r_full - ri
    flag = " <-- largest drop" if abs(drop) == max(abs(r_full - stats.pearsonr(np.delete(x,j), np.delete(y,j))[0]) for j in range(n)) else ""
    print(f"  drop {nm:<14}: r = {ri:.4f}  (Δ = {drop:+.4f}){flag}")
    if abs(drop) > abs(max_drop):
        max_drop = drop
        max_drop_sys = nm

print()
print(f"Largest single-system impact: {max_drop_sys}  Δr = {max_drop:+.4f}")
print(f"r range across all LOO: [{min(stats.pearsonr(np.delete(x,i), np.delete(y,i))[0] for i in range(n)):.4f}, "
      f"{max(stats.pearsonr(np.delete(x,i), np.delete(y,i))[0] for i in range(n)):.4f}]")

# ── 4. Power analysis ──
print()
print("=" * 60)
print(f"Power Analysis  (α=0.05, two-tailed)")
print("=" * 60)

from scipy.stats import norm

def pearson_power(r, n, alpha=0.05):
    """Achieved power via Fisher z-transform approximation."""
    z_r = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_alpha = norm.ppf(1 - alpha / 2)
    # Non-centrality
    ncp = z_r / se
    power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
    return power

def min_detectable_r(n, alpha=0.05, power=0.80):
    """Minimum detectable r at given power via binary search."""
    lo, hi = 0.0, 0.999
    for _ in range(100):
        mid = (lo + hi) / 2
        if pearson_power(mid, n, alpha) < power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

r_obs = r_full
achieved_power = pearson_power(r_obs, n)
mde_80 = min_detectable_r(n, power=0.80)
mde_90 = min_detectable_r(n, power=0.90)

print(f"Observed r = {r_obs:.4f},  n = {n}")
print(f"Achieved power (α=0.05, two-tailed): {achieved_power:.4f}  ({achieved_power*100:.1f}%)")
print(f"Min detectable r at 80% power: {mde_80:.4f}")
print(f"Min detectable r at 90% power: {mde_90:.4f}")
print()

# ── 5. Spearman LOO ──
print("=" * 60)
print("Leave-one-out sensitivity (Spearman ρ)")
print("=" * 60)
rho_full, _ = stats.spearmanr(x, y)
print(f"Full n=16: ρ = {rho_full:.4f}")
rho_vals = []
for i, nm in enumerate(names):
    xi = np.delete(x, i)
    yi = np.delete(y, i)
    ri, _ = stats.spearmanr(xi, yi)
    rho_vals.append(ri)
    print(f"  drop {nm:<14}: ρ = {ri:.4f}  (Δ = {ri-rho_full:+.4f})")
print(f"ρ range across all LOO: [{min(rho_vals):.4f}, {max(rho_vals):.4f}]")
