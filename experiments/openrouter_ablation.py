#!/usr/bin/env python3
"""
Cross-model, multi-seed Finance ablation (B2 / B2' / DKAP) via OpenRouter.

Addresses two reviewer concerns about the structuring effect:
  (1) it was measured on a single deterministic run / underpowered n=5;
  (2) it came from a single LLM (Qwen3-32B).

Here the same B2 / B2' / DKAP ablation is rerun across several independent
model families x multiple stochastic seeds (temperature=0.3). Schema, seed
data, the 50-question benchmark, the four prompt builders, and the SQLite
execution-accuracy (EX) evaluator are reused verbatim from
dkap_experiment.py and dkap_b2prime_stochastic.py; only the model call changes.

Phase 1 (concurrent): generate SQL for every (model, condition, seed, question)
  via OpenRouter -> JSONL (resumable).
Phase 2 (sequential): evaluate EX against gold SQL on SQLite (thread-unsafe, so
  kept out of the concurrent phase).
Phase 3: per-model / condition / difficulty means, the DKAP-B2' structuring gain
  with bootstrap CIs and a cross-model sign test, and the B2'-B2 completeness gain.

Usage:
  python3.10 openrouter_ablation.py --list            # resolve models, exit
  python3.10 openrouter_ablation.py --smoke           # 1 seed, 6 Q/difficulty
  python3.10 openrouter_ablation.py                   # full run
  python3.10 openrouter_ablation.py --analyze-only    # recompute stats from JSONL
"""
import os
import re
import sys
import json
import time
import argparse
import threading
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sqlite3
from dkap_experiment import BENCHMARK, SCHEMA_DDL, SEED_DATA_SQL, evaluate_ex
from dkap_b2prime_stochastic import get_prompt  # dispatches B1/B2/B2_PRIME/DKAP

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
BASE = "https://openrouter.ai/api/v1"
TEMPERATURE = 0.3
SEEDS = int(os.environ.get("ABL_SEEDS", "10"))
CONDITIONS = ["B2", "B2_PRIME", "DKAP"]
MAX_TOKENS = int(os.environ.get("ABL_MAX_TOKENS", "2048"))  # raise for reasoning models
MAX_WORKERS = int(os.environ.get("ABL_WORKERS", "12"))
OUT_DIR = Path(os.environ.get("ABL_OUT", "results/ablation_openrouter"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_PATH = OUT_DIR / "generations.jsonl"

# Six lineage-distinct families (user-specified set for the ablation).
MODEL_CANDIDATES = {
    "OpenAI":    ["openai/gpt-5.4", "openai/gpt-5.5", "openai/gpt-4.1"],
    "Anthropic": ["anthropic/claude-sonnet-4.6", "anthropic/claude-opus-4.6",
                  "anthropic/claude-sonnet-4"],
    "Google":    ["google/gemini-3.1-pro-preview", "google/gemini-3.5-flash",
                  "google/gemini-2.5-pro"],
    "DeepSeek":  ["deepseek/deepseek-v3.2", "deepseek/deepseek-v4-flash",
                  "deepseek/deepseek-chat"],
    "Qwen":      ["qwen/qwen3.6-27b", "qwen/qwen3.7-plus",
                  "qwen/qwen-2.5-72b-instruct"],
    "Mistral":   ["mistralai/mistral-medium-3-5", "mistralai/mistral-large-2512",
                  "mistralai/mistral-large"],
}


# ------------------------------------------------------------------
# .env + HTTP (urllib; the env's requests/idna is broken)
# ------------------------------------------------------------------
def load_env():
    here = Path(__file__).resolve().parent
    for p in (here / ".env", here.parent / ".env"):
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


def _headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/back2zion/DKAP",
        "X-Title": "DKAP cross-model ablation",
    }


def http_get(path, timeout=60):
    req = urllib.request.Request(BASE + path, headers=_headers(), method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def http_post(path, body, timeout=120):
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers=_headers(), method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def resolve_models():
    available = set()
    try:
        available = {m.get("id") for m in http_get("/models").get("data", [])}
        print(f"[resolve] catalog: {len(available)} models")
    except Exception as e:
        print(f"[resolve] WARNING: /models failed ({e}); using first candidates")
    out = {}
    for fam, cands in MODEL_CANDIDATES.items():
        pick = next((c for c in cands if c in available), cands[0]) if available else cands[0]
        out[fam] = pick
        print(f"  {fam:<10} -> {pick}")
    return out


def call_model(slug, prompt, seed, max_retries=3, timeout=int(os.environ.get("ABL_TIMEOUT", "240"))):
    body = {
        "model": slug,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "seed": seed,
    }
    last = ""
    for attempt in range(max_retries):
        try:
            resp = http_post("/chat/completions", body, timeout=timeout)
            content = (resp.get("choices") or [{}])[0].get("message", {}).get("content")
            if content:
                return content
            last = "empty content"
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode()[:200] if hasattr(e,'read') else ''}"
        except Exception as e:
            last = str(e)
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    return f"__ERROR__ {last}"


def clean_sql(text):
    """Extract a single SQL statement from a model response."""
    if not text:
        return ""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:sql)?\s*(.+?)```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        t = m.group(1)
    t = re.sub(r"(?im)^\s*sql\s*:\s*", "", t.strip())
    m = re.search(r"(?is)\b(SELECT|WITH)\b.*", t)
    if m:
        t = m.group(0)
    t = t.split(";")[0].strip()
    return t


# ------------------------------------------------------------------
# Phase 1: concurrent generation (resumable)
# ------------------------------------------------------------------
def load_done(path):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                if r.get("error") or not r.get("pred_sql"):
                    continue  # retry failed / empty generations on resume
                done.add((r["family"], r["condition"], r["seed"], r["qid"]))
            except Exception:
                pass
    return done


def generate(models, bench, seeds):
    done = load_done(RAW_PATH)
    tasks = []
    for fam, slug in models.items():
        for cond in CONDITIONS:
            for seed in range(1, seeds + 1):
                for b in bench:
                    key = (fam, cond, seed, b["id"])
                    if key not in done:
                        tasks.append((fam, slug, cond, seed, b))
    print(f"[generate] {len(tasks)} new calls ({len(done)} already done)")
    if not tasks:
        return
    lock = threading.Lock()
    fh = RAW_PATH.open("a")
    counter = {"n": 0}

    def work(t):
        fam, slug, cond, seed, b = t
        prompt = get_prompt(cond, b["nl"])
        raw = call_model(slug, prompt, seed)
        rec = {
            "family": fam, "model": slug, "condition": cond, "seed": seed,
            "qid": b["id"], "difficulty": b["difficulty"],
            "gold_sql": b["gold_sql"], "pred_sql": clean_sql(raw),
            "error": raw.startswith("__ERROR__"),
        }
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            counter["n"] += 1
            if counter["n"] % 50 == 0:
                print(f"  [{counter['n']}/{len(tasks)}] ...")
        return rec["error"]

    errs = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for e in ex.map(work, tasks):
            errs += int(bool(e))
    fh.close()
    print(f"[generate] done; {errs} call errors")


# ------------------------------------------------------------------
# Phase 2: sequential EX evaluation
# ------------------------------------------------------------------
def evaluate_all():
    conn = sqlite3.connect(":memory:")  # avoid /tmp disk I/O flakiness
    conn.executescript(SCHEMA_DDL)
    conn.executescript(SEED_DATA_SQL)
    raw = [json.loads(l) for l in RAW_PATH.read_text().splitlines() if l.strip()]
    # dedup by key, preferring a successful (non-error, non-empty) record
    best = {}
    for r in raw:
        k = (r["family"], r["condition"], r["seed"], r["qid"])
        cur = best.get(k)
        good = (not r.get("error")) and bool(r.get("pred_sql"))
        if cur is None or good:
            best[k] = r
    rows = list(best.values())
    for r in rows:
        if r.get("error") or not r["pred_sql"]:
            r["ex"] = 0
        else:
            try:
                r["ex"] = int(evaluate_ex(conn, r["pred_sql"], r["gold_sql"]))
            except Exception:
                r["ex"] = 0
    conn.close()
    return rows


# ------------------------------------------------------------------
# Phase 3: analysis
# ------------------------------------------------------------------
def _boot_ci(diffs, n_boot=20000):
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) < 2:
        return (float("nan"), float("nan"))
    idx = np.random.default_rng(12345).integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return (round(float(np.percentile(means, 2.5)), 2),
            round(float(np.percentile(means, 97.5)), 2))


def analyze(rows):
    fams = list(MODEL_CANDIDATES.keys())
    diffs_by = {"E": [], "M": [], "H": []}
    # ex[fam][cond][qid] = mean over seeds
    ex = {}
    for r in rows:
        if r.get("error") or not r.get("pred_sql"):
            continue  # API failures are missing data, not a 0 score
        ex.setdefault(r["family"], {}).setdefault(r["condition"], {}).setdefault(
            r["qid"], []).append(r["ex"])
    qdiff = {b["id"]: b["difficulty"] for b in BENCHMARK}

    def cond_mean(fam, cond, diffs=None):
        d = ex.get(fam, {}).get(cond, {})
        vals = [np.mean(v) for q, v in d.items() if diffs is None or qdiff[q] in diffs]
        return float(np.mean(vals)) * 100 if vals else float("nan")

    # only analyze families with (near-)complete coverage
    expected = len(CONDITIONS) * SEEDS * len(BENCHMARK)

    def fam_cells(f):
        return sum(len(v) for cond in ex.get(f, {}) for v in ex[f][cond].values())

    complete = [f for f in fams if fam_cells(f) >= 0.9 * expected]
    excluded = {f: fam_cells(f) for f in fams if f not in complete and fam_cells(f) > 0}

    per_model = {}
    em_gain, em_gain_signs = [], []
    for fam in complete:
        if fam not in ex:
            continue
        row = {
            "model": None,
            "B2": round(cond_mean(fam, "B2"), 1),
            "B2_PRIME": round(cond_mean(fam, "B2_PRIME"), 1),
            "DKAP": round(cond_mean(fam, "DKAP"), 1),
            "completeness_gain_B2prime_minus_B2": round(
                cond_mean(fam, "B2_PRIME") - cond_mean(fam, "B2"), 1),
            "structuring_gain_DKAP_minus_B2prime": {
                "overall": round(cond_mean(fam, "DKAP") - cond_mean(fam, "B2_PRIME"), 1),
                "Easy": round(cond_mean(fam, "DKAP", ["E"]) - cond_mean(fam, "B2_PRIME", ["E"]), 1),
                "Medium": round(cond_mean(fam, "DKAP", ["M"]) - cond_mean(fam, "B2_PRIME", ["M"]), 1),
                "Hard": round(cond_mean(fam, "DKAP", ["H"]) - cond_mean(fam, "B2_PRIME", ["H"]), 1),
                "EasyMedium": round(cond_mean(fam, "DKAP", ["E", "M"]) - cond_mean(fam, "B2_PRIME", ["E", "M"]), 1),
            },
        }
        per_model[fam] = row
        em_gain.append(row["structuring_gain_DKAP_minus_B2prime"]["EasyMedium"])
        em_gain_signs.append(1 if row["structuring_gain_DKAP_minus_B2prime"]["EasyMedium"] > 0 else 0)
        # paired per-question diffs (Easy+Medium) for pooled bootstrap
        d_dkap = ex[fam].get("DKAP", {})
        d_bp = ex[fam].get("B2_PRIME", {})
        for q in set(d_dkap) & set(d_bp):
            if qdiff[q] in ("E", "M"):
                diffs_by[qdiff[q]].append(np.mean(d_dkap[q]) - np.mean(d_bp[q]))

    em_pool = diffs_by["E"] + diffs_by["M"]
    n_models = len(per_model)
    summary = {
        "n_models": n_models,
        "n_seeds": SEEDS,
        "temperature": TEMPERATURE,
        "models_analyzed": list(per_model.keys()),
        "models_excluded_incomplete": excluded,
        "per_model": per_model,
        "structuring_gain_EasyMedium": {
            "per_model_pp": em_gain,
            "mean_pp": round(float(np.mean(em_gain)), 2) if em_gain else None,
            "min_pp": round(float(np.min(em_gain)), 2) if em_gain else None,
            "max_pp": round(float(np.max(em_gain)), 2) if em_gain else None,
            "models_positive": f"{sum(em_gain_signs)}/{n_models}",
            "pooled_mean_pp": round(float(np.mean(em_pool)) * 100, 2) if em_pool else None,
            "pooled_bootstrap_95CI_pp": tuple(x * 100 if x == x else x for x in _boot_ci(em_pool)),
        },
        "completeness_gain_B2prime_minus_B2_mean_pp": round(
            float(np.mean([per_model[f]["completeness_gain_B2prime_minus_B2"] for f in per_model])), 2),
    }
    return summary


def write_report(s, models):
    L = ["# Cross-Model Multi-Seed Finance Ablation (B2 / B2' / DKAP)\n"]
    L.append(f"- Models analyzed: **{s['n_models']}** ({', '.join(s['models_analyzed'])}) | "
             f"seeds: **{s['n_seeds']}** @ temp={s['temperature']}")
    if s.get("models_excluded_incomplete"):
        L.append(f"- Excluded (incomplete, key-limit): {s['models_excluded_incomplete']}")
    for f in s["models_analyzed"]:
        L.append(f"  - {f}: `{models.get(f,'?')}`")
    L.append("\n## Per-model execution accuracy (%) and gains")
    L.append("| Family | B2 | B2' | DKAP | Compl. (B2'-B2) | Struct. E+M (DKAP-B2') |")
    L.append("|---|---|---|---|---|---|")
    for f, r in s["per_model"].items():
        L.append(f"| {f} | {r['B2']} | {r['B2_PRIME']} | {r['DKAP']} | "
                 f"+{r['completeness_gain_B2prime_minus_B2']} | "
                 f"{r['structuring_gain_DKAP_minus_B2prime']['EasyMedium']:+} |")
    g = s["structuring_gain_EasyMedium"]
    L.append("\n## Structuring gain (DKAP - B2'), Easy+Medium")
    L.append(f"- Per-model (pp): {g['per_model_pp']}")
    L.append(f"- Mean **{g['mean_pp']}pp** (range {g['min_pp']}..{g['max_pp']}); "
             f"models with positive gain: **{g['models_positive']}**")
    L.append(f"- Pooled mean **{g['pooled_mean_pp']}pp**, bootstrap 95% CI {g['pooled_bootstrap_95CI_pp']}")
    L.append(f"\n## Completeness gain (B2' - B2): mean **+{s['completeness_gain_B2prime_minus_B2_mean_pp']}pp**")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if not API_KEY:
        sys.exit("ERROR: OPENROUTER_API_KEY not set (put it in ../.env).")

    if args.analyze_only:
        rows = evaluate_all()
        s = analyze(rows)
        (OUT_DIR / "statistics.json").write_text(json.dumps(s, indent=2, ensure_ascii=False))
        models = {f: rows and next((r["model"] for r in rows if r["family"] == f), "?") for f in MODEL_CANDIDATES}
        (OUT_DIR / "REPORT.md").write_text(write_report(s, models))
        print(write_report(s, models))
        return

    print("Resolving models ...")
    models = resolve_models()
    if args.list:
        return

    bench = BENCHMARK
    seeds = SEEDS
    if args.smoke:
        seeds = 1
        by = {"E": [], "M": [], "H": []}
        for b in BENCHMARK:
            if len(by[b["difficulty"]]) < 2:
                by[b["difficulty"]].append(b)
        bench = by["E"] + by["M"] + by["H"]
        print(f"[smoke] {len(bench)} questions x {len(CONDITIONS)} cond x {len(models)} models")

    t0 = time.time()
    generate(models, bench, seeds)
    print(f"[generate] wall {round(time.time()-t0)}s")
    rows = evaluate_all()
    s = analyze(rows)
    (OUT_DIR / "statistics.json").write_text(json.dumps(s, indent=2, ensure_ascii=False))
    (OUT_DIR / "REPORT.md").write_text(write_report(s, models))
    print("\n" + write_report(s, models))


if __name__ == "__main__":
    main()
