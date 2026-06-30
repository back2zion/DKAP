# Model Paper Candidates for DKAP (IEEE Access)

Compiled 2026-06-11 from a three-track literature search (top-venue empirical SE /
IEEE Access structural templates / controlled-ablation methodology).
Purpose: structural "model papers" per Prof. Park's suggestion — imitate form, not content.

## Primary recommendation (overall arc)

**Kreuzberger, Kühl & Hirschl, "MLOps: Overview, Definition, and Architecture," IEEE Access 11 (2023), DOI 10.1109/ACCESS.2023.3262138** — ~1,500+ GS citations.
Same venue, same shape one abstraction level up: multi-source evidence → derived
principles → component mapping → architecture figure → quotable definition.
Imitable moves: (1) traceability chain — every element of the architecture figure
back-referenced to evidence IDs (DKAP layers ↔ system IDs); (2) a one-paragraph
citable *definition* of the core concept.
https://ieeexplore.ieee.org/document/10081336

## Per-part models

### A. 29-system mining / pattern derivation (Sections III, V)
- **Nahar, Zhang, Lewis, Zhou & Kästner, "The Product Beyond the Model," ICSE 2025** (arXiv:2308.04328).
  Selection protocol → 30-system deep code analysis → inductive architecture findings.
  Closest one-to-one match to the Architecture Mining half. OSF replication package is the
  transparency standard to match. https://arxiv.org/abs/2308.04328
- **Sens, Knopp, Peldszus & Berger, "A Large-Scale Study of Model Integration in ML-Enabled
  Software Systems," ICSE 2025** (arXiv:2408.06226). 2,928-repo source-level mining →
  recurring integration patterns. Useful positioning contrast: their breadth vs. DKAP depth.
- **Cosentino, Cánovas Izquierdo & Cabot, "A Systematic Mapping Study of Software Development
  With GitHub," IEEE Access 5 (2017)**. Model for the selection-funnel presentation
  (initial pool → filters → final corpus, flow figure, per-criterion counts) and the
  construct/internal/external Threats-to-Validity taxonomy.

### B. DKS/RSS rubric + reliability (Sections III-D, VI)
- **Humbatova et al., "Taxonomy of Real Faults in Deep Learning Systems," ICSE 2020**
  (arXiv:1910.11015). Canonical template for inductive coding + multi-rater agreement +
  validation survey. Model for moving DKS from descriptor toward validated instrument.
- **Amershi et al., "Software Engineering for Machine Learning: A Case Study," ICSE-SEIP 2019**
  (~1,000 citations). Precedent for practitioner-sourced industrial data + derived workflow
  pattern + maturity/rubric-style scoring model in a top venue.

### C. Ablation / B2' information-matched control (Section IV)
- **Zhang, Maekawa & Bhutani, "Same Content, Different Representations: A Controlled Study
  for Table QA," ICLR 2026** (arXiv:2509.22983, RePairTQA). Holds content constant, varies
  representation structure — the exact B2' design logic, with difficulty stratification.
  STRONG CITE CANDIDATE, not just a model.
- **Chang & Fosler-Lussier, "How to Prompt LLMs for Text-to-SQL," NeurIPS 2023 TRL workshop**
  (arXiv:2305.11853). Factorial ablation of schema serialization formats — direct precedent
  for the B1_DDL condition. CITE CANDIDATE.
- **Maamari et al., "The Death of Schema Linking?," NeurIPS 2024 TRL** (arXiv:2408.07702).
  Component-isolation showing coverage dominates a structuring step — same
  conditional-benefit genre as the completeness-primary finding. CITE CANDIDATE.
- **Xiang et al., "When to use Graphs in RAG," ICLR 2026** (arXiv:2506.05690, GraphRAG-Bench).
  Graph vs. flat RAG on same corpora with difficulty tiers — supports the applicability-
  threshold framing. CITE CANDIDATE.
- **Cuconasu et al., "The Power of Noise," SIGIR 2024** (arXiv:2401.14887). Typed-context
  composition ablation; methodological standard for context-composition controls.
- **BIRD (Li et al., NeurIPS 2023)** — already cited [6]; note its +20pp domain-evidence
  ablation as quantitative anchor for the +26pp completeness result.

### D. Venue-norm calibration (IEEE Access)
- **Testi et al., "MLOps: A Taxonomy and a Methodology," IEEE Access 10 (2022)** — clean split
  of descriptive (taxonomy) vs. prescriptive (methodology) contributions; uniform per-step
  subsection template reusable for the 5 layers.
- **Fuller et al., "Digital Twin," IEEE Access 8 (2020)** — "what X is NOT" definitional table;
  device for the structuring ≠ more-information distinction.
- **Adadi & Berrada, XAI survey, IEEE Access 6 (2018)** — concept-disambiguation section before
  the taxonomy (structure vs. information vs. injection).
- **Zhang et al., "M-SQL," IEEE Access 8 (2020)** — Access-format Text-to-SQL ablation paper;
  module-aligned ablation table mapped 1:1 to named architecture components.

## Gaps in current draft these models expose

1. **Threats to Validity taxonomy**: current Section VI is "Limitations and Future Work" prose;
   empirical-SE reviewers expect construct/internal/external organization (Cosentino, Nahar).
2. **Selection funnel figure**: 40+ → 13 internal and candidate pool → 16 external is described
   in text; a funnel flow figure with per-criterion counts is the genre convention.
3. **Layer↔evidence traceability**: tag each DKAP layer with the system IDs evidencing it
   (Kreuzberger's P1–P9 device).
4. **B2' precedent citations**: RePairTQA and Chang & Fosler-Lussier give the B2' design
   independent methodological legitimacy — currently uncited.
5. **Quotable definition**: a boxed one-paragraph definition of "domain knowledge structuring
   as an architectural concern" (Kreuzberger move).
