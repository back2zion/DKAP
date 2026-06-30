#!/usr/bin/env python3
"""
Cross-family LLM-as-Judge inter-rater reliability for DKS/RSS scoring.

Six independent judges from different model families (OpenAI, Anthropic,
Google, DeepSeek, Alibaba/Qwen, Mistral) accessed via OpenRouter, each scoring
the same 12 systems on the DKS/RSS rubric. Diversity comes from the model
family (one neutral expert persona per model), unlike the single-base-model
five-persona protocol in multi_agent_interrater.py.

Reports:
  - per-rater (per-family) score summaries
  - mean pairwise Cohen's kappa (quadratic-weighted, 4-bin)
  - Fleiss' kappa (4-bin categorization)         [comparable to the Qwen run]
  - Krippendorff's alpha (interval, raw totals)  [requested by reviewer]
  - inter-rater Spearman rho
  - author-vs-ensemble agreement (reference scores r1_dks+r1_rss)

Rubric / system evidence / prompt / parsing / kappa helpers are reused from
multi_agent_interrater.py so the only thing that changes is the rater axis.

Usage:
  # put OPENROUTER_API_KEY in ../.env  (repo root) or experiments/.env
  python3.10 openrouter_crossfamily_ira.py --list       # resolve models, no scoring
  python3.10 openrouter_crossfamily_ira.py --limit 2    # smoke test: 2 systems
  python3.10 openrouter_crossfamily_ira.py              # full 6 x 12 run
"""
import os
import re
import sys
import json
import time
import argparse
import itertools
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

# --- reuse rubric / systems / prompt / parsing / kappa from the sibling module
import multi_agent_interrater as mai
from multi_agent_interrater import (
    REFINED_RUBRIC,
    SYSTEM_EVIDENCE,
    SCORING_PROMPT_TEMPLATE,
    parse_scoring_response,
    fleiss_kappa,
    cohens_kappa_weighted,
)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
BASE = "https://openrouter.ai/api/v1"
TEMPERATURE = 0.0
MAX_TOKENS = 8192  # headroom for models that emit reasoning before the JSON
OUT_DIR = Path(os.environ.get("OR_RESULTS_DIR", "results/interrater_openrouter"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
# Systems excluded from the cross-family cohort (e.g. removed from the final
# analysis). Set OR_EXCLUDE="A,B" to override.
EXCLUDE_SYSTEMS = {s for s in os.environ.get("OR_EXCLUDE", "").split(",") if s}

# Seven lineage-distinct families (different labs/architectures, not just
# different vendors). 2026-era general models; first candidate OpenRouter
# actually serves is used. Kimi excluded (K2 reuses DeepSeek-V3 architecture);
# Llama excluded (latest on OpenRouter is 2025-04, below the early-2026 bar).
MODEL_CANDIDATES = {
    "OpenAI":    ["openai/gpt-5.4", "openai/gpt-5.5", "openai/gpt-4.1",
                  "openai/gpt-4o"],
    "Anthropic": ["anthropic/claude-sonnet-4.6", "anthropic/claude-opus-4.6",
                  "anthropic/claude-sonnet-4"],
    "Google":    ["google/gemini-3.1-pro-preview", "google/gemini-3.5-flash",
                  "google/gemini-2.5-pro"],
    "Mistral":   ["mistralai/mistral-medium-3-5", "mistralai/mistral-large-2512",
                  "mistralai/mistral-large"],
    "DeepSeek":  ["deepseek/deepseek-v3.2", "deepseek/deepseek-v4-flash",
                  "deepseek/deepseek-chat-v3.1", "deepseek/deepseek-chat"],
    "Qwen":      ["qwen/qwen3.6-27b", "qwen/qwen3.7-plus",
                  "qwen/qwen-2.5-72b-instruct"],
    "GLM":       ["z-ai/glm-5.2", "z-ai/glm-5.1", "z-ai/glm-5"],
}

# Per-component score caps from the rubric (DKS max 100, RSS max 12).
# Used to clamp out-of-range hallucinated scores (e.g. a negative, or a value
# above the component max) to the valid interval before aggregation.
COMPONENT_MAX = {
    "D1": 15, "D2": 15, "D3": 10, "D4": 15, "D5": 10, "D6": 10, "D7": 10, "D8": 15,
    "R1": 3, "R2": 3, "R3": 2, "R4": 2, "R5": 2,
}


def robust_parse(raw: str):
    """Tolerant parse: strip reasoning blocks / prose around the JSON object."""
    if not raw:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    p = parse_scoring_response(cleaned)
    if p:
        return p
    i, j = cleaned.find("{"), cleaned.rfind("}")
    if 0 <= i < j:
        return parse_scoring_response(cleaned[i:j + 1])
    return None


def clamp_scores(parsed: dict) -> int:
    """Clamp each component to [0, max] and recompute totals. Returns #clamped."""
    n_clamped = 0
    for block in ("dks", "rss"):
        for comp, cell in parsed.get(block, {}).items():
            cap = COMPONENT_MAX.get(comp)
            if cap is None:
                continue
            try:
                s = int(cell["score"])
            except (KeyError, TypeError, ValueError):
                continue
            cs = max(0, min(cap, s))
            if cs != s:
                cell["score"] = cs
                n_clamped += 1
    parsed["dks_total"] = sum(parsed["dks"][k]["score"] for k in
                              ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"])
    parsed["rss_total"] = sum(parsed["rss"][k]["score"] for k in
                              ["R1", "R2", "R3", "R4", "R5"])
    parsed["total"] = parsed["dks_total"] + parsed["rss_total"]
    return n_clamped


# Single neutral expert persona; the rater identity is the model family.
NEUTRAL_PERSONA = {
    "name": "an expert evaluator of AI system architectures",
    "background": (
        "You are a senior researcher in AI systems engineering and software "
        "architecture. You assess how explicitly a system encodes structured "
        "domain knowledge (glossaries, schemas, ontologies, rule sets) and how "
        "sophisticated its retrieval is. You count only explicit, source-visible "
        "artifacts; you do not credit implicit knowledge held in pretrained "
        "weights. You apply the rubric literally and consistently."
    ),
}


# ------------------------------------------------------------------
# .env loading (no external dependency)
# ------------------------------------------------------------------
def load_env():
    here = Path(__file__).resolve().parent
    for p in (here / ".env", here.parent / ".env"):
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


# ------------------------------------------------------------------
# HTTP helpers (urllib only — avoids the broken `requests`/idna install)
# ------------------------------------------------------------------
def _headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        # OpenRouter recommends these; harmless if ignored.
        "HTTP-Referer": "https://github.com/back2zion/DKAP",
        "X-Title": "DKAP cross-family IRA",
    }


def http_get(path: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(BASE + path, headers=_headers(), method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def http_post(path: str, body: dict, timeout: int = 240) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers=_headers(), method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def resolve_models() -> dict:
    """Return {family: slug} using the first candidate OpenRouter actually serves."""
    available = set()
    try:
        catalog = http_get("/models")
        available = {m.get("id") for m in catalog.get("data", [])}
        print(f"[resolve] OpenRouter catalog: {len(available)} models")
    except Exception as e:
        print(f"[resolve] WARNING: could not fetch /models ({e}); using first candidates")

    resolved = {}
    for family, cands in MODEL_CANDIDATES.items():
        pick = None
        if available:
            for c in cands:
                if c in available:
                    pick = c
                    break
        if pick is None:
            pick = cands[0]  # optimistic fallback; per-call errors will surface
            tag = "fallback"
        else:
            tag = "ok"
        resolved[family] = pick
        print(f"  {family:<10} -> {pick}  ({tag})")
    return resolved


def call_judge(slug: str, prompt: str, max_retries: int = 2,
               timeout: int = 150) -> str:
    body = {
        "model": slug,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    last_err = ""
    for attempt in range(max_retries):
        try:
            resp = http_post("/chat/completions", body, timeout=timeout)
            return resp["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:300] if hasattr(e, "read") else ""
            last_err = f"HTTP {e.code}: {detail}"
        except Exception as e:
            last_err = str(e)
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    print(f"    ! call failed ({slug}): {last_err}")
    return ""


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------
def run_all_scorings(models: dict, limit: int | None = None) -> list:
    systems = [(k, v) for k, v in SYSTEM_EVIDENCE.items() if k not in EXCLUDE_SYSTEMS]
    if limit:
        systems = systems[:limit]
    results = []
    total = len(models) * len(systems)
    done = 0
    for family, slug in models.items():
        for system_name, system in systems:
            done += 1
            print(f"[{done}/{total}] {family} ({slug}) scoring {system_name} ...")
            prompt = SCORING_PROMPT_TEMPLATE.format(
                persona_name=NEUTRAL_PERSONA["name"],
                persona_background=NEUTRAL_PERSONA["background"],
                persona_id=family,
                rubric=REFINED_RUBRIC,
                system_name=system_name,
                system_repo=system["repo"],
                system_paper=system["paper"],
                system_domain=system["domain"],
                system_evidence=system["evidence"],
            )
            raw = call_judge(slug, prompt)
            parsed = robust_parse(raw)
            if not parsed:                         # one retry on parse failure
                raw = call_judge(slug, prompt)
                parsed = robust_parse(raw)
            if parsed:
                n_clamped = clamp_scores(parsed)   # fix out-of-range hallucinations
                parsed["persona_id"] = family      # rater identity = family
                parsed["family"] = family
                parsed["model_slug"] = slug
                parsed["n_clamped"] = n_clamped
                results.append(parsed)
                flag = f" [clamped {n_clamped}]" if n_clamped else ""
                print(f"    -> DKS={parsed['dks_total']} RSS={parsed['rss_total']} "
                      f"Total={parsed['total']}{flag}")
            else:
                (OUT_DIR / f"failed_{family}_{system_name}.txt").write_text(raw or "")
                print(f"    -> FAILED to parse ({family}/{system_name})")
            time.sleep(0.3)
    return results


# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------
def krippendorff_alpha_interval(matrix_units_by_raters: np.ndarray) -> float:
    """Krippendorff's alpha, interval metric. Input: units x raters."""
    try:
        import krippendorff
        return float(krippendorff.alpha(
            reliability_data=matrix_units_by_raters.T,  # raters x units
            level_of_measurement="interval",
        ))
    except Exception as e:
        print(f"[alpha] krippendorff pkg unavailable ({e}); using manual interval formula")
        # manual interval alpha
        vals_by_unit = [row[~np.isnan(row)] for row in matrix_units_by_raters]
        all_vals = np.concatenate(vals_by_unit)
        N = len(all_vals)
        if N < 2:
            return float("nan")
        De = np.mean([(a - b) ** 2 for a, b in itertools.permutations(all_vals, 2)])
        num, den = 0.0, 0.0
        for row in vals_by_unit:
            m = len(row)
            if m < 2:
                continue
            num += sum((a - b) ** 2 for a, b in itertools.permutations(row, 2)) / (m - 1)
            den += m
        Do = num / den if den else float("nan")
        return 1 - Do / De if De else float("nan")


def compute_stats(results: list) -> dict:
    systems = [k for k in SYSTEM_EVIDENCE.keys() if k not in EXCLUDE_SYSTEMS]
    families = list(MODEL_CANDIDATES.keys())
    # keep only families that actually produced results
    present = [f for f in families if any(r["persona_id"] == f for r in results)]
    n_sys, n_rat = len(systems), len(present)

    total_m = np.full((n_sys, n_rat), np.nan)
    dks_m = np.full((n_sys, n_rat), np.nan)
    for r in results:
        if r["system"] in systems and r["persona_id"] in present:
            i = systems.index(r["system"])
            j = present.index(r["persona_id"])
            total_m[i, j] = r["total"]
            dks_m[i, j] = r["dks_total"]

    # complete cases only for matrix stats
    mask = ~np.isnan(total_m).any(axis=1)
    total_v = total_m[mask]
    sys_v = [s for s, m in zip(systems, mask) if m]

    stats = {
        "n_systems_scored": int(mask.sum()),
        "n_raters": n_rat,
        "raters": present,
        "rater_means": {
            present[j]: round(float(np.nanmean(total_m[:, j])), 2) for j in range(n_rat)
        },
    }

    if n_rat >= 2 and mask.sum() >= 2:
        # pairwise Cohen's kappa (quadratic-weighted, 4-bin) + Spearman
        kappas, rhos = [], []
        pair_detail = {}
        for a, b in itertools.combinations(range(n_rat), 2):
            k = cohens_kappa_weighted(total_v[:, a], total_v[:, b], n_bins=4)
            rho, _ = spearmanr(total_v[:, a], total_v[:, b])
            kappas.append(k)
            rhos.append(rho)
            pair_detail[f"{present[a]}_vs_{present[b]}"] = {
                "cohen_kappa": round(float(k), 3),
                "spearman_rho": round(float(rho), 3),
            }
        stats["mean_pairwise_cohen_kappa"] = round(float(np.mean(kappas)), 3)
        stats["mean_pairwise_spearman"] = round(float(np.mean(rhos)), 3)
        stats["pairwise"] = pair_detail

        # Fleiss' kappa, 4-bin (same recipe as the Qwen run)
        max_score = float(np.nanmax(total_v))
        bins = np.linspace(0, max_score + 1e-9, 5)
        binned = np.clip(np.digitize(total_v, bins) - 1, 0, 3)
        stats["fleiss_kappa_4bin"] = round(float(fleiss_kappa(binned, 4)), 3)

        # Krippendorff alpha (interval, raw totals)
        stats["krippendorff_alpha_interval"] = round(
            krippendorff_alpha_interval(total_m), 3)

    # author-vs-ensemble (reference scores r1_dks + r1_rss)
    ens, ref, used = [], [], []
    for i, s in enumerate(systems):
        ev = SYSTEM_EVIDENCE[s]
        if "r1_dks" in ev and "r1_rss" in ev and not np.isnan(total_m[i]).all():
            ens.append(float(np.nanmean(total_m[i])))
            ref.append(ev["r1_dks"] + ev["r1_rss"])
            used.append(s)
    if len(ens) >= 3:
        rho, p_rho = spearmanr(ref, ens)
        r, p_r = pearsonr(ref, ens)
        stats["author_vs_ensemble"] = {
            "n": len(ens),
            "spearman_rho": round(float(rho), 3),
            "spearman_p": round(float(p_rho), 4),
            "pearson_r": round(float(r), 3),
            "pearson_p": round(float(p_r), 4),
            "systems": used,
        }

    # per-system ensemble means (for the paper table / stability illustration)
    stats["per_system"] = {}
    for i, s in enumerate(systems):
        col = total_m[i]
        if not np.isnan(col).all():
            stats["per_system"][s] = {
                "ensemble_mean_total": round(float(np.nanmean(col)), 1),
                "min": round(float(np.nanmin(col)), 1),
                "max": round(float(np.nanmax(col)), 1),
                "std": round(float(np.nanstd(col)), 1),
                "by_family": {present[j]: (None if np.isnan(total_m[i, j])
                                           else round(float(total_m[i, j]), 1))
                              for j in range(n_rat)},
            }
    return stats


def write_report(stats: dict, models: dict) -> str:
    L = []
    L.append("# Cross-Family LLM-as-Judge — Inter-Rater Reliability (DKS/RSS)\n")
    L.append(f"- Systems scored (complete cases): **{stats['n_systems_scored']}**")
    L.append(f"- Judges (families): **{stats['n_raters']}** — {', '.join(stats['raters'])}")
    L.append("- Models:")
    for f in stats["raters"]:
        L.append(f"  - {f}: `{models.get(f, '?')}`")
    L.append("")
    if "fleiss_kappa_4bin" in stats:
        L.append("## Agreement")
        L.append(f"- Fleiss' kappa (4-bin): **{stats['fleiss_kappa_4bin']}**")
        L.append(f"- Krippendorff's alpha (interval): **{stats['krippendorff_alpha_interval']}**")
        L.append(f"- Mean pairwise Cohen's kappa: **{stats['mean_pairwise_cohen_kappa']}**")
        L.append(f"- Mean pairwise Spearman rho: **{stats['mean_pairwise_spearman']}**")
        L.append("")
    if "author_vs_ensemble" in stats:
        a = stats["author_vs_ensemble"]
        L.append("## Author vs. cross-family ensemble")
        L.append(f"- n = {a['n']} systems")
        L.append(f"- Spearman rho = **{a['spearman_rho']}** (p = {a['spearman_p']})")
        L.append(f"- Pearson r = **{a['pearson_r']}** (p = {a['pearson_p']})")
        L.append("")
    L.append("## Per-system ensemble totals")
    L.append("| System | mean | min | max | std |")
    L.append("|---|---|---|---|---|")
    for s, d in stats.get("per_system", {}).items():
        L.append(f"| {s} | {d['ensemble_mean_total']} | {d['min']} | {d['max']} | {d['std']} |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="resolve models and exit")
    ap.add_argument("--limit", type=int, default=None, help="score only first N systems")
    args = ap.parse_args()

    if not API_KEY:
        sys.exit("ERROR: OPENROUTER_API_KEY not set (put it in ../.env or export it).")

    print("Resolving model families on OpenRouter ...")
    models = resolve_models()
    if args.list:
        return

    results = run_all_scorings(models, limit=args.limit)
    (OUT_DIR / "raw_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved raw -> {OUT_DIR/'raw_results.json'} ({len(results)} scorings)")

    stats = compute_stats(results)
    (OUT_DIR / "statistics.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    report = write_report(stats, models)
    (OUT_DIR / "REPORT.md").write_text(report)
    print(f"Saved stats -> {OUT_DIR/'statistics.json'}")
    print(f"Saved report -> {OUT_DIR/'REPORT.md'}\n")
    print(report)


if __name__ == "__main__":
    main()
