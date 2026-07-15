# DKS D1-D8 Component Scores — Provenance

This file documents where `dks_d1_d8_component_scores.json` comes from. It
backs the DKS weight-sensitivity analysis reported in the paper (Section
3.4.1, Supplement S5) and computed by `sensitivity_analysis.py`.

## What this is

Per-component (D1-D8) author scores for the 13 paradigm-matched external
systems used throughout the paper's external validation. Until this file
was assembled, only the *summed* DKS total per system was published in
`paper/DKS_RSS_Scoring_Sheet_for_Second_Rater.md` and
`paper/PeerJ_CS/supplementary.tex` (Table `tab:external_systems`) — the
component-level breakdown needed to test alternative weighting schemes had
never been collected into one place.

## Sources, by system

- **MedRAG, FinSQL, Wheatley, BIRD-SQL, LlamaIndex, SpeechBrain, GraphRAG,
  AudioCraft, EmotiVoice, CrewAI** (10 systems): per-component scores with
  file-level evidence are in `paper/Supplementary_External_System_Analysis.md`
  (sections S1-S9, S12). LlamaIndex uses the "Typical Deployment" column
  (DKS=20) from S5.2, matching the value used everywhere else in the paper,
  not the "Framework Score" column (DKS=41).
- **MetaGPT, Pulse AI, DeepTutor** (3 systems): per-component scores are in
  `paper/PeerJ_CS/supplementary.tex`, the "Scoring notes for new systems"
  itemize block (lines 662-664 as of this writing).

## Verification performed

Every system's D1-D8 sum was checked against the DKS total already
published and cited elsewhere in the paper (the scoring sheet and
`tab:external_systems`). All 13 match exactly — see the verification block
at the bottom of `sensitivity_analysis.py`, which also asserts this at
runtime (`verify_totals()`).

## What this superseded

An earlier attempt to reproduce the sensitivity figures used the 7-judge
OpenRouter LLM-ensemble scores (`experiments/results/interrater_openrouter_v2/raw_results.json`)
as a stand-in for author scores. That data source is a *different*
methodology (independent LLM judges, not the author's own scoring) and,
when run through the same weighting schemes, did not reproduce the
previously reported sensitivity values — the mismatch was largest and in
the wrong direction for the Uniform/Inverted/Rank-only schemes. It was
discarded in favor of the actual author-scored data collected here.

## Result

Recomputing DKS-Stages Spearman correlation from this data under all four
weighting schemes did **not** reproduce the previously reported values
(Original 0.758, Uniform 0.73, Inverted 0.70, Rank-only 0.72). The
computed values (Original 0.745, Uniform 0.799, Inverted 0.864, Rank-only
0.839) are now the values reported in the paper, and this dataset +
`sensitivity_analysis.py` are the source of truth going forward.
