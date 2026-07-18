# DKAP: Domain Knowledge Adaptation Pattern

**Completeness over Structuring: A Controlled Text-to-SQL Ablation and Source-Code Analysis of Twenty-Nine AI Systems**

Dooil Kwak · Department of AI Techno Convergence, Graduate School, Soongsil University, Seoul, Republic of Korea

Code, data, and manuscript source for the PeerJ Computer Science submission (AI Application article). This repository reproduces every quantitative result, table, and figure in the paper.

---

## 1. Description

DKAP (Domain Knowledge Adaptation Pattern) is a five-layer architectural pattern empirically observed across **twenty-nine AI systems** (13 internal industrial systems, 16 independently developed external open-source systems) spanning **sixteen AI paradigms**, including five non-LLM systems (RL, ASR/TTS, music generation):

| Layer | Role | Example Artifacts |
|-------|------|-------------------|
| L1 | Input Processing | Normalize and route the incoming request |
| L2 | Knowledge Structuring | Glossaries, ontologies, schemas, rule sets compiled offline into static artifacts |
| L3 | Retrieval / Routing | Consume L2 artifacts at query time for domain-conditioned retrieval |
| L4 | AI Inference | Engine-agnostic core (LLM, RL, VLM, or ML) |
| L5 | Output Validation | Verify and audit the output |

The central question is not whether providing domain knowledge helps (it clearly does), but whether the value comes from *structuring* that knowledge or simply from providing *more* of it. A controlled Text-to-SQL ablation — including an information-matched control (B2') that supplies the same domain knowledge as unstructured prose — isolates these two effects. **Main finding:** information completeness is the primary, robust driver of accuracy; explicit structuring adds a smaller, difficulty-dependent gain that does not reliably generalize across model families. A lightweight descriptive rubric (Domain Knowledge Score, DKS; Retrieval Sophistication Score, RSS) characterizes domain-knowledge complexity and tracks how the combined L2 benefit scales with it.

**What's in this repo:** the ablation experiment runners (three domains, six model families, information-matched control), the cross-family / within-family inter-rater reliability studies, the DKS/RSS scoring data and weight-sensitivity analysis, the automated DKS proxy scorer and 265-repo population study, and the LaTeX source of the manuscript and Supplemental Material.

## 2. Dataset Information

All datasets needed to reproduce the numbers are in this repository; the released archive is also on Zenodo (concept **DOI: 10.5281/zenodo.21366415**).

**Text-to-SQL ablation benchmarks.** Each domain is *self-contained and synthetic*: the experiment script builds an in-memory SQLite database from an inline schema and inline synthetic seed records, and evaluates a fixed list of author-written natural-language → gold-SQL pairs. No external raw dataset is ingested, filtered, or de-identified at run time.

| Domain | Schema (synthetic) | Questions | Difficulty split | Source file |
|--------|--------------------|-----------|------------------|-------------|
| Finance | 5-table Korean financial-card data warehouse (member / transaction / credit); grounded in the L2 knowledge base (schema DDL + glossary + transform rules) | 50 | 13 E / 19 M / 18 H | `experiments/dkap_experiment.py` |
| Healthcare | Clinical schema following the OMOP Common Data Model, modeled on the **AIhub Clinical QA** domain | 40 | E / M / H | `experiments/dkap_healthcare_experiment.py` |
| Public procurement | Schema modeled on the **KONEPS** e-procurement system | 25 | E / M / H | `experiments/dkap_public_experiment.py` |

**Licensing / redistribution note.** The benchmarks shipped here are *synthetic stand-ins* modeled on the AIhub Clinical QA and KONEPS domains — they do **not** contain records from those sources, so no third-party data is redistributed. If you use the **real** AIhub or KONEPS datasets, obtain them from their official providers under their own licenses. The 13 **internal industrial systems** are proprietary and are **not** redistributed; only their derived, non-identifying architecture scores (DKS, RSS, stage counts) appear in the paper and in `docs/`.

**Source-code analysis data.**
- `tools/dks_d1_d8_component_scores.json` — per-component (D1–D8) author DKS scores for the 13 source-mined external systems (input to the sensitivity analysis).
- `docs/Supplementary_External_System_Analysis.md` — per-system, source-code-level scoring evidence for all 16 external systems (DKS+RSS, stages, Python LOC, file counts).
- `tools/dks_265_results.csv` / `.json`, `tools/repos_all_265.txt` — automated DKS proxy scores + GitHub metadata for the 265-repo (≥10k-star) Python-AI population study.

## 3. Code Information

**Ablation runners** (`experiments/`)
- `dkap_experiment.py` — finance Text-to-SQL ablation; defines the schema/seed data, the 50-question benchmark, and the B1 / B2 / DKAP prompt builders; runs on the local vLLM `Qwen3-32B-AWQ` endpoint and writes raw + summary JSON to `results/`.
- `dkap_healthcare_experiment.py`, `dkap_public_experiment.py` — the same pipeline for the healthcare (40Q) and public-procurement (25Q) domains.
- `dkap_b1_ddl_experiment.py` — the B1_DDL schema-only diagnostic condition.
- `dkap_b2prime_stochastic.py` — the B2′ information-matched control (same information as DKAP, delivered as prose) plus multi-seed stochastic-robustness runs; also exposes `get_prompt()` used by the cross-model runner.
- `openrouter_ablation.py` — reruns the finance B2 / B2′ / DKAP ablation across six independent model families via the OpenRouter API (temperature 0.3, 10 seeds).
- `openrouter_crossfamily_ira.py` — cross-family inter-rater reliability: 7 judge lineages × 13 external systems score the DKS/RSS rubric.
- `multi_agent_interrater.py` — within-family reliability: 5 persona-differentiated raters from a single base model (upper bound on reproducibility).
- `wheatley_l2_ablation.py` (+ `run_wheatley_ablation.sh`) — non-LLM (RL job-shop scheduling) cross-paradigm L2 ablation.
- `fig3_fig4_24systems.py`, `fig5_*.py`, `teaser_3bar_decomposition.py` — figure generation from the released result files.

**Scoring & analysis tools** (`tools/`)
- `sensitivity_analysis.py` — recomputes the DKS–Stages correlation under uniform / inverted / rank-only weighting schemes to show the association is not an artifact of the weight assignment (reproduces §3.4.1). Reads `dks_d1_d8_component_scores.json`.
- `dks_analyzer.py` — automated DKS proxy scorer for an arbitrary GitHub repo (single-repo or `--batch`); scans for the eight D1–D8 proxy signals.
- `collect_265_data.py` / `analyze_265_full.py` — GitHub-API metadata collection and batch shallow-clone + automated DKS analysis for the 265-repo population study.

**Docs** (`docs/`): full DKS/RSS rubric with scoring criteria, the second-rater inter-rater pilot, and the per-system source-code analysis.

## 4. Usage Instructions

```bash
# 0. Clone + install (see Requirements below)
git clone https://github.com/back2zion/DKAP.git && cd DKAP
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt

# 1. Reproduce the DKS weight-sensitivity analysis (§3.4.1) — no GPU or API key needed
python tools/sensitivity_analysis.py

# 2. Regenerate the DKS+RSS vs. pipeline-complexity figures from released results
python experiments/fig3_fig4_24systems.py

# 3. Analyze any GitHub repo with the automated DKS proxy scorer
python tools/dks_analyzer.py https://github.com/org/repo
python tools/dks_analyzer.py --batch tools/repos_all_265.txt

# 4. Text-to-SQL ablations — need an LLM backend (see Requirements)
cp experiments/.env.example experiments/.env      # add OPENROUTER_API_KEY for the cross-model runs
python experiments/dkap_experiment.py             # finance   (local vLLM Qwen3-32B-AWQ)
python experiments/dkap_healthcare_experiment.py  # healthcare
python experiments/dkap_public_experiment.py      # public procurement
python experiments/dkap_b2prime_stochastic.py     # B2′ information-matched control
python experiments/openrouter_ablation.py         # 6 model families via OpenRouter
```

The statistics and figures (steps 1–3) reproduce from the committed result files with **no GPU and no API key**. Only the ablation runners (step 4) call an LLM.

## 5. Requirements

- **Python** 3.11 for analysis, figures, and the OpenRouter experiments; **Python 3.10** for the non-LLM RL experiment.
- **Packages** (`requirements.txt`): `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm` (core); `openai`, `requests` (OpenRouter clients). `sqlite3` is in the Python standard library. GPU-only serving deps (`vllm==0.18.0`, `transformers`, `torch>=2.3`) are commented in `requirements.txt` and installed separately on a CUDA 12.x host.
- **API key (optional):** cross-model / cross-family runs call [OpenRouter](https://openrouter.ai) and need `OPENROUTER_API_KEY` (see `experiments/.env.example`).
- **Hardware (optional):** the primary-model runs need a local vLLM-served `Qwen3-32B-AWQ` endpoint. Paper environment: Ubuntu Linux (exact release not recorded), dual NVIDIA H100 NVL GPUs (96 GB each, tensor-parallel = 2), vLLM 0.18.0, CUDA 12.8. No GPU is required to reproduce the statistics and figures.

## 6. Methodology

Full details are in the manuscript (`paper/PeerJ_CS/main.tex`, *Materials & Methods*). In brief:

1. **Architecture Mining Protocol** (§3.2) — a four-step protocol decomposes each system's source into pipeline stages and assigns them to the five DKAP layers.
2. **Scoring** (§3.4) — each system receives DKS (domain-knowledge density, D1–D8) and RSS (retrieval sophistication, R1–R5) from source-code evidence; robustness is checked by the weight-sensitivity analysis (§3.4.1).
3. **Controlled ablation** (§3.6–3.8) — Text-to-SQL under B1 / B1_DDL / B2 / DKAP plus the B2′ information-matched control, across three domains, six model families, and ten seeds (~9,000 runs). Computing infrastructure and preprocessing are documented in §3.8.
4. **Reliability & robustness** (§5.7) — cross-family (7 lineages) and within-family (5 personas) inter-rater reliability, leave-one-out, and influence analysis.

## 7. Citations

Please cite the paper (and this repository) as:

```bibtex
@article{kwak2026dkap,
  title   = {Completeness over Structuring: A Controlled Text-to-SQL Ablation and
             Source-Code Analysis of Twenty-Nine AI Systems},
  author  = {Kwak, Dooil},
  year    = {2026},
  journal = {PeerJ Computer Science (under review)},
  note    = {Code and data: https://github.com/back2zion/DKAP;
             Zenodo DOI: 10.5281/zenodo.21366415}
}
```

## 8. License & Contribution Guidelines

Released under the **MIT License** — see [`LICENSE`](LICENSE). This license covers the code and the synthetic benchmark data only; it does **not** cover the proprietary internal industrial systems (not redistributed) or the real AIhub / KONEPS datasets (obtain from their official providers under their own terms).

**Contributions are accepted.** Please open an issue to discuss a change before submitting a pull request, and report reproducibility problems via the [issue tracker](https://github.com/back2zion/DKAP/issues) with your OS, Python version, and the exact command you ran.
