"""DKS weighting-sensitivity analysis (Section 3.4.1 / Supplement S5).

Recomputes the DKS-Stages Spearman correlation for the n=13 paradigm-matched
external systems under the original weighting and three alternative schemes
(uniform, inverted, rank-only), to verify the values reported in the paper.

Input data: dks_d1_d8_component_scores.json (see the accompanying README
for provenance). Every system's D1-D8 sum is checked against the DKS total
already published elsewhere in the paper before any correlation is run.
"""

import json
import os

from scipy.stats import spearmanr, rankdata

DATA_PATH = os.path.join(os.path.dirname(__file__), "dks_d1_d8_component_scores.json")

COMPONENTS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]


def load_data():
    with open(DATA_PATH) as f:
        raw = json.load(f)
    orig_max = raw["_component_max"]
    systems = raw["systems"]
    return orig_max, systems


def verify_totals(orig_max, systems):
    for name, rec in systems.items():
        computed = sum(rec[c] for c in COMPONENTS)
        expected = rec["dks_total"]
        assert computed == expected, f"{name}: sum(D1-D8)={computed} != published DKS={expected}"
    print(f"Verified: all {len(systems)} systems' D1-D8 sums match published DKS totals.\n")


def weighted_score(scores, orig_max, weight_map):
    return sum((scores[c] / orig_max[c]) * weight_map[c] for c in COMPONENTS)


def rank_only_composite(systems):
    names = list(systems.keys())
    composite = {n: 0.0 for n in names}
    for c in COMPONENTS:
        col = [systems[n][c] for n in names]
        ranks = rankdata(col)
        for n, r in zip(names, ranks):
            composite[n] += r
    return composite


def run():
    orig_max, systems = load_data()
    verify_totals(orig_max, systems)

    uniform_max = {c: 12.5 for c in COMPONENTS}
    inverted_max = {c: (10 if orig_max[c] == 15 else 15) for c in COMPONENTS}

    names = list(systems.keys())
    stages = [systems[n]["stages"] for n in names]

    original = [weighted_score(systems[n], orig_max, orig_max) for n in names]
    uniform = [weighted_score(systems[n], orig_max, uniform_max) for n in names]
    inverted = [weighted_score(systems[n], orig_max, inverted_max) for n in names]
    rank_comp = rank_only_composite(systems)
    rank_only = [rank_comp[n] for n in names]

    schemes = {
        "original": original,
        "uniform": uniform,
        "inverted": inverted,
        "rank_only": rank_only,
    }

    print(f"{'Scheme':<12}{'rho':>10}{'p-value':>12}")
    results = {}
    for key, vals in schemes.items():
        rho, p = spearmanr(vals, stages)
        results[key] = (rho, p)
        print(f"{key:<12}{rho:>10.3f}{p:>12.4f}")

    return results


if __name__ == "__main__":
    run()
