# DKAP: Domain Knowledge Adaptation Pattern

**Completeness over Structuring: A Controlled Text-to-SQL Ablation and Source-Code Analysis of Twenty-Nine AI Systems**

Dooil Kwak
Department of AI Techno Convergence, Graduate School, Soongsil University, Seoul, Republic of Korea

---

## Overview

DKAP (Domain Knowledge Adaptation Pattern) is a five-layer architectural pattern empirically observed across **twenty-nine AI systems** (13 internal industrial systems, 16 independently developed external open-source systems) spanning **sixteen AI paradigms**, including five non-LLM systems (RL, ASR/TTS, music generation):

| Layer | Role | Example Artifacts |
|-------|------|-------------------|
| L1 | Input Processing | Normalize and route the incoming request |
| L2 | Knowledge Structuring | Glossaries, ontologies, schemas, rule sets compiled offline into static artifacts |
| L3 | Retrieval / Routing | Consume L2 artifacts at query time for domain-conditioned retrieval |
| L4 | AI Inference | Engine-agnostic core (LLM, RL, VLM, or ML) |
| L5 | Output Validation | Verify and audit the output |

The central question this study asks is not whether providing domain knowledge helps (it clearly does), but whether the value comes from *structuring* that knowledge or simply from providing *more* of it. A controlled ablation — including an information-matched control (B2') that supplies the same domain knowledge as unstructured prose — isolates these two effects. The main finding: **information completeness is the primary, robust driver of accuracy; explicit structuring adds a smaller, difficulty-dependent gain that does not reliably generalize across model families.**

A lightweight descriptive rubric (Domain Knowledge Score, DKS; Retrieval Sophistication Score, RSS) characterizes domain knowledge complexity and tracks how the combined L2 benefit scales with it.

## Repository Structure

```
DKAP/
├── paper/PeerJ_CS/
│   ├── main.tex                       # Manuscript (PeerJ Computer Science submission format)
│   ├── supplementary.tex              # Supplemental Material
│   ├── refs.bib                       # Bibliography
│   ├── main.pdf / supplementary.pdf   # Compiled PDFs
│   ├── wlpeerj.cls                    # PeerJ LaTeX class (vendored template)
│   └── *.pdf / *.png                  # Figures
├── experiments/
│   ├── dkap_experiment.py             # Finance Text-to-SQL ablation (B1/B2/DKAP)
│   ├── dkap_healthcare_experiment.py  # Healthcare Text-to-SQL ablation
│   ├── dkap_public_experiment.py      # Public procurement ablation
│   ├── dkap_b2prime_stochastic.py     # B2' information-matched control + stochastic robustness
│   ├── openrouter_ablation.py         # Cross-model generalization (6 model families)
│   ├── openrouter_crossfamily_ira.py  # Cross-family inter-rater reliability (7 lineages x 13 systems)
│   ├── multi_agent_interrater.py      # Within-family inter-rater reliability (5 personas)
│   ├── wheatley_l2_ablation.py        # Non-LLM (RL scheduling) cross-paradigm ablation
│   ├── fig3_fig4_24systems.py         # DKS+RSS vs. pipeline-stage-count scatter + cross-validation figures
│   └── results/                       # Raw experimental outputs
├── tools/
│   ├── dks_analyzer.py                       # Automated DKS proxy scorer
│   ├── sensitivity_analysis.py               # DKS weight-sensitivity reproduction script
│   ├── dks_d1_d8_component_scores.json       # Per-component author D1-D8 scores (n=13), input to the above
│   ├── dks_d1_d8_component_scores_README.md  # Provenance of the per-component data
│   ├── collect_265_data.py / analyze_265_full.py  # 265-repo GitHub population study
│   └── dks_265_results.csv / .json           # Population-study results
├── docs/
│   ├── DKS_RSS_Scoring_Sheet_for_Second_Rater.md   # Full DKS/RSS rubric with criteria
│   ├── DKS_RSS_Second_Rater_Results.md             # Second-rater inter-rater pilot
│   └── Supplementary_External_System_Analysis.md   # Per-system source-code-level scoring evidence
└── .gitignore
```

## Key Results

### Ablation (Text-to-SQL, execution accuracy)
| Condition | Finance | Healthcare | Public Procurement |
|-----------|---------|------------|---------------------|
| B1 (zero-shot, no L2) | 0.0% | 2.5% | 0.0% |
| B2 (flat-chunk RAG, partial L2) | 54.0% | 57.5% | 64.0% |
| DKAP (full L1-L5, structured L2) | 80.0% | 70.0% | 64.0% |
| **DKAP - B2** | **+26.0pp** | **+12.5pp** | **+0.0pp** |

The combined L2 benefit scales with domain knowledge complexity (DKS): largest in the highest-complexity domain (finance), essentially absent in the lowest-complexity one (public procurement).

### Information Completeness vs. Structuring (B2' control, finance)
Separating *how much* domain knowledge is provided from *how it's organized*: B2' supplies the same information as DKAP but as unstructured prose.
- Completeness (B2 -> B2'): **+26.0pp**, replicating to a **+17.5pp mean** across six independent model families (positive for every family)
- Structuring (B2' -> DKAP): **+5.3 to +7.7pp** on the primary model, but a **+0.7pp mean** across six families with a 95% CI that includes zero — this does not reliably generalize

### DKS+RSS vs. Pipeline Complexity
Across the sixteen independently developed external systems: Pearson r = 0.857 (95% CI [0.63, 0.95], p < 0.001), Spearman rho = 0.789 (p < 0.001). Robust to leave-one-out and alternative DKS weighting schemes (see `tools/sensitivity_analysis.py`).

## Scoring Rubrics

**DKS (Domain Knowledge Score)** [0-100]: 8 components (D1-D8) measuring domain knowledge density and structural depth in source code.
**RSS (Retrieval Sophistication Score)** [0-12]: 5 components (R1-R5) measuring retrieval-side sophistication.

Both are researcher-defined *descriptors*, not validated instruments for evaluating system quality.

- Full rubric with scoring criteria: [`docs/DKS_RSS_Scoring_Sheet_for_Second_Rater.md`](docs/DKS_RSS_Scoring_Sheet_for_Second_Rater.md)
- Per-system, source-code-level scoring evidence: [`docs/Supplementary_External_System_Analysis.md`](docs/Supplementary_External_System_Analysis.md)

## Reproducing Results

```bash
# Finance / Healthcare / Public procurement Text-to-SQL ablations
python experiments/dkap_experiment.py
python experiments/dkap_healthcare_experiment.py
python experiments/dkap_public_experiment.py

# B2' information-matched control + stochastic robustness
python experiments/dkap_b2prime_stochastic.py

# Cross-model generalization (6 independent model families via OpenRouter)
python experiments/openrouter_ablation.py

# DKS weight-sensitivity analysis (verifies robustness to alternative weighting schemes)
python tools/sensitivity_analysis.py

# Analyze a single GitHub repo with the automated DKS proxy scorer
python tools/dks_analyzer.py https://github.com/org/repo

# Batch analyze repos (265-repo population study)
python tools/dks_analyzer.py --batch tools/repos_all_265.txt
```

Most LLM experiments require an OpenRouter API key (`OPENROUTER_API_KEY`, see `experiments/.env.example`) or a local vLLM-served Qwen3-32B-AWQ endpoint (dual H100 NVL GPUs, tensor-parallel=2). See `paper/PeerJ_CS/supplementary.tex`, "Experiment Scripts and Reproducibility," for exact hardware/software versions.

## Citation

```bibtex
@article{kwak2026dkap,
  title={Completeness over Structuring: A Controlled Text-to-SQL Ablation and
         Source-Code Analysis of Twenty-Nine AI Systems},
  author={Kwak, Dooil},
  year={2026},
  note={Manuscript. See paper/PeerJ_CS/main.tex for the current version.}
}
```

## License

This repository contains research code and data for academic purposes.
