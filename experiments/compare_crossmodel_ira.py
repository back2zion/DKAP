#!/usr/bin/env python3
"""
Compare Qwen3-32B-AWQ (original) vs Qwen3.6-27B-AWQ (cross-model) DKS/RSS scoring.
Computes Spearman rho between per-system agent means across the two models.
"""
import json
import itertools
from pathlib import Path

try:
    import numpy as np
    from scipy.stats import spearmanr, pearsonr
except ImportError:
    import os
    os.system("pip install numpy scipy --break-system-packages -q")
    import numpy as np
    from scipy.stats import spearmanr, pearsonr

ORIG_DIR = Path("results/interrater")
CROSS_DIR = Path("results/interrater_crossmodel")

def load_per_system_means(raw_path: Path) -> dict:
    """Load raw results and compute per-system agent-mean DKS+RSS total."""
    raw = json.loads(raw_path.read_text())
    means = {}
    for system_id, system_data in raw.items():
        system_name = system_data.get("system_name", system_id)
        scores = []
        for agent_key, agent_data in system_data.items():
            if not agent_key.startswith("agent_"):
                continue
            dks = agent_data.get("dks_total", 0)
            rss = agent_data.get("rss_total", 0)
            scores.append(dks + rss)
        if scores:
            means[system_name] = np.mean(scores)
    return means

def main():
    orig_raw = ORIG_DIR / "interrater_raw_results.json"
    cross_raw = CROSS_DIR / "interrater_raw_results.json"

    if not orig_raw.exists():
        print(f"ERROR: {orig_raw} not found")
        return
    if not cross_raw.exists():
        print(f"ERROR: {cross_raw} not found")
        return

    orig_means = load_per_system_means(orig_raw)
    cross_means = load_per_system_means(cross_raw)

    # Find common systems
    common = sorted(set(orig_means) & set(cross_means))
    print(f"Common systems: {len(common)}")

    orig_vals = [orig_means[s] for s in common]
    cross_vals = [cross_means[s] for s in common]

    rho, rho_p = spearmanr(orig_vals, cross_vals)
    r, r_p = pearsonr(orig_vals, cross_vals)
    diffs = [abs(o - c) for o, c in zip(orig_vals, cross_vals)]
    mean_diff = np.mean(diffs)

    print(f"\nCross-model consistency (Qwen3-32B vs Qwen3.6-27B):")
    print(f"  Spearman rho = {rho:.3f} (p={rho_p:.3f})")
    print(f"  Pearson r    = {r:.3f} (p={r_p:.3f})")
    print(f"  Mean |delta| = {mean_diff:.1f} points")

    print(f"\nPer-system comparison (DKS+RSS agent mean):")
    print(f"{'System':<20} {'32B':>8} {'27B':>8} {'Delta':>8}")
    print("-" * 48)
    for s in common:
        print(f"{s:<20} {orig_means[s]:>8.1f} {cross_means[s]:>8.1f} {cross_means[s]-orig_means[s]:>+8.1f}")

    # Also load statistics JSONs
    orig_stats = json.loads((ORIG_DIR / "interrater_statistics.json").read_text())
    cross_stats = json.loads((CROSS_DIR / "interrater_statistics.json").read_text())

    print(f"\nInter-agent agreement comparison:")
    print(f"  Qwen3-32B:   Fleiss' kappa = {orig_stats.get('fleiss_kappa', 'N/A')}")
    print(f"  Qwen3.6-27B: Fleiss' kappa = {cross_stats.get('fleiss_kappa', 'N/A')}")

    # Save comparison
    result = {
        "n_common_systems": len(common),
        "spearman_rho": round(rho, 3),
        "spearman_p": round(rho_p, 3),
        "pearson_r": round(r, 3),
        "mean_abs_delta": round(mean_diff, 1),
        "orig_fleiss_kappa": orig_stats.get("fleiss_kappa"),
        "cross_fleiss_kappa": cross_stats.get("fleiss_kappa"),
        "per_system": {s: {"orig": round(orig_means[s], 1), "cross": round(cross_means[s], 1)} for s in common},
    }
    out = Path("results/crossmodel_comparison.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
