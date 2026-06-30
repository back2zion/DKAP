# DKAP: Domain Knowledge Adaptation Pattern

**An Empirical Study of Architectural Patterns in Domain-Specific AI Systems**

Dooil Kwak and Kunwoo Park  
Department of AI Techno Convergence, Soongsil University, Seoul, South Korea

---

## Overview

DKAP (Domain Knowledge Adaptation Pattern) is a five-layer architectural pattern empirically observed across 24 AI systems (12 internal, 12 external open-source) spanning 10 AI paradigms:

| Layer | Role | Example Artifacts |
|-------|------|-------------------|
| L1 | Input Normalization | Tokenizers, schema validators, format converters |
| L2 | Domain Knowledge Structuring | Glossaries, ontologies, cross-standard mappings, SQL schemas |
| L3 | Retrieval & Routing | Vector search, intent classification, adaptive routing |
| L4 | Inference | LLM, RL agent, VLM, ML ensemble |
| L5 | Output Validation | PII masking, regulatory checks, format enforcement |

The key finding is that **L2 (Domain Knowledge Structuring) consistently appears as a distinct, measurable architectural layer** whose complexity correlates with overall system depth (Pearson r = 0.81, p = 0.001 for 12 external systems).

## Repository Structure

```
DKAP/
├── paper/
│   ├── main.tex              # IEEE Access paper (English)
│   ├── main_KO.tex           # Korean version
│   ├── supplementary.tex     # Supplementary material (S1-S6)
│   ├── main.pdf              # Compiled paper
│   └── supplementary.pdf     # Compiled supplementary
├── experiments/
│   ├── fig3_fig4_24systems.py        # Fig 3/4: scatter + cross-validation
│   ├── dkap_b2prime_stochastic.py    # B2' information control experiment
│   ├── multi_agent_interrater.py     # 5-agent inter-rater reliability
│   └── results/                      # Raw experimental outputs
├── tools/
│   ├── dks_analyzer.py               # Automated DKS proxy scorer
│   ├── collect_265_data.py           # GitHub API metadata collector
│   ├── analyze_265_full.py           # Batch clone + DKS analysis
│   ├── repos_all_265.txt             # 265 GitHub repo URLs
│   ├── dks_265_results.csv           # 263 repos: DKS scores + metadata
│   ├── dks_265_results.json          # Full analysis results (JSON)
│   └── external_12_auto_dks.json     # 12 external systems auto vs manual
└── .gitignore
```

## Key Results

### Ablation (Finance Text-to-SQL)
| Condition | EX Accuracy |
|-----------|-------------|
| Full DKAP (L1-L5) | 73.3% |
| B2' (same info, unstructured) | 68.0% |
| Partial L2 (B2) | 47.3% |
| No L2 (B1) | 21.3% |

### 265-Repo Population Study
- 263 Python AI repos (stars >= 10K) analyzed with automated DKS
- Population DKS: mean=42.5, median=44, std=20.0
- 12 external systems are representative of population (KS test p=0.478)

### DKS Analyzer Validity
- Automated DKS accurately scores focused domain systems (EmotiVoice: auto=48, manual=47)
- Overestimates large frameworks due to code-size bias (Spearman rho=0.741 with LOC)
- Not a substitute for expert scoring; useful for population-level characterization

## Scoring Rubrics

**DKS (Domain Knowledge Score)** [0-100]: 8 components measuring domain knowledge depth  
**RSS (Retrieval Sophistication Score)** [0-12]: 5 components measuring retrieval complexity

Full rubric: [`paper/DKS_RSS_Scoring_Sheet_for_Second_Rater.md`](paper/DKS_RSS_Scoring_Sheet_for_Second_Rater.md)

## Reproducing Results

```bash
# Fig 3/4: 24-system scatter + cross-validation
python experiments/fig3_fig4_24systems.py

# B2' stochastic experiment
python experiments/dkap_b2prime_stochastic.py

# Analyze a single GitHub repo
python tools/dks_analyzer.py https://github.com/org/repo

# Batch analyze repos
python tools/dks_analyzer.py --batch tools/repos_all_265.txt
```

## Citation

```bibtex
@article{kwak2026dkap,
  title={Domain Knowledge Structuring as a Measurable Architectural Concern: 
         An Empirical Study of 24 AI Systems},
  author={Kwak, Dooil and Park, Kunwoo},
  journal={IEEE Access},
  year={2026},
  note={Under review}
}
```

## License

This repository contains research code and data for academic purposes.
