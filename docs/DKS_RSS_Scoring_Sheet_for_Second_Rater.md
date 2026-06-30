# DKS/RSS Independent Scoring Sheet (13 External Systems)

**Purpose**: This scoring sheet enables a second rater to independently score 13 external open-source systems using the DKS/RSS rubrics defined in the paper. Results will be used to compute Cohen's Kappa inter-rater reliability coefficient.

**Updated**: 2026-05-06 — Updated from 12 to 13 systems (added MetaGPT, Pulse AI, DeepTutor; removed LLaVA, Logic-LLM).

**Instructions**:
1. Clone each repository and examine the source code
2. Score each component independently (do NOT reference the first rater's scores)
3. Use the rubric criteria exactly as defined below
4. Record your scores and brief justification for each component
5. Suggested order (ascending DKS): MedRAG → Pulse AI → CrewAI → LlamaIndex → BIRD-SQL → Wheatley → FinSQL → MetaGPT → SpeechBrain → AudioCraft → DeepTutor → EmotiVoice → GraphRAG

---

## Rubric: DKS (Domain Knowledge Score) — Range [0, 100]

| Component | Max | Criteria |
|-----------|-----|----------|
| D1. Domain lexicon size | 15 | 0: none \| 5: 1-50 terms \| 10: 51-200 \| 15: >200 terms |
| D2. Ontology/KG presence | 15 | 0: none \| 5: flat taxonomy (nodes<50) \| 10: hierarchical (nodes<200) \| 15: graph (nodes>=200, edges>=0.5*nodes) |
| D3. Cross-ontology mapping | 10 | 0: none \| 5: single standard \| 10: multi-standard |
| D4. Schema/DDL complexity | 15 | 0: none \| 5: 1-5 tables (no FK) \| 10: 6-15 tables+FK \| 15: >15 tables with constraints+triggers |
| D5. Domain profile count | 10 | 0: none \| 3: <10 \| 6: 10-50 \| 10: >50 profiles/patterns |
| D6. Code-level rule patterns | 10 | 0: none \| 3: <10 patterns \| 6: 10-30 \| 10: >30 regex/rule patterns |
| D7. Multi-language term mapping | 10 | 0: single lang \| 5: bilingual \| 10: trilingual+ or code->natural lang |
| D8. Regulatory/compliance encoding | 15 | 0: none \| 5: basic PII \| 10: sector regulation \| 15: multi-regulation |

## Rubric: RSS (Retrieval Sophistication Score) — Range [0, 12]

| Component | Max | Criteria |
|-----------|-----|----------|
| R1. Intent classification | 3 | 0: none \| 1: keyword \| 2: ML classifier \| 3: multi-stage |
| R2. Vector search sophistication | 3 | 0: none \| 1: flat search \| 2: filtered search \| 3: search+reranking |
| R3. Permission-based filtering | 2 | 0: none \| 1: role-based \| 2: document-level + role-based |
| R4. Multi-index fusion | 2 | 0: single index \| 1: dual \| 2: triple+ |
| R5. Adaptive routing | 2 | 0: fixed \| 1: rule-based \| 2: ML/LLM-based dynamic routing |

---

## Scoring Template

*Copy and complete for each system below.*

### DKS Scoring

| Component | Score (0-max) | Justification |
|-----------|--------------|---------------|
| D1 (lexicon, max 15) | ___ | |
| D2 (ontology/KG, max 15) | ___ | |
| D3 (cross-ontology, max 10) | ___ | |
| D4 (schema/DDL, max 15) | ___ | |
| D5 (profiles, max 10) | ___ | |
| D6 (rules, max 10) | ___ | |
| D7 (multi-lang, max 10) | ___ | |
| D8 (regulatory, max 15) | ___ | |
| **DKS Total** | ___ / 100 | |

### RSS Scoring

| Component | Score (0-max) | Justification |
|-----------|--------------|---------------|
| R1 (intent, max 3) | ___ | |
| R2 (vector search, max 3) | ___ | |
| R3 (permission, max 2) | ___ | |
| R4 (multi-index, max 2) | ___ | |
| R5 (adaptive routing, max 2) | ___ | |
| **RSS Total** | ___ / 12 | |

### Pipeline Stage Count: ___

---

## System 1: MedRAG

**Repository**: https://github.com/Teddy-XiongGZ/MedRAG
**Paper**: Xiong et al., "Benchmarking Retrieval-Augmented Generation for Medicine," Findings of ACL 2024
**Key files**: `src/medrag.py` (main pipeline), `src/` (retrieval + generation)
**Author score**: DKS=5, RSS=5, Total=10, Stages=4

*(Use scoring template above)*

---

## System 2: Pulse AI

**Repository**: https://github.com/tejiri-code/pulse-ai
**Paradigm**: Info Monitoring (content scraping, LLM summarization, multi-platform publishing)
**Key files**: `backend/agent.py` (LLM agent), `backend/scraper.py` + `backend/scraper_reddit.py` (content scraping), `backend/models.py` (data models), `backend/cron_scheduler.py` + `backend/cron_jobs.py` (scheduling), `backend/db.py` + `backend/user_db.py` (storage)
**Note**: Small project (~5.2K Python lines, 26 files). Minimal domain knowledge structuring; no explicit glossary or ontology. Scoring focus: check whether alert/topic definitions constitute D1/D5, and whether cron+scraper pipeline constitutes structured retrieval.
**Author score**: DKS=14, RSS=1, Total=15, Stages=6

*(Use scoring template above)*

---

## System 3: CrewAI

**Repository**: https://github.com/crewAIInc/crewAI
**Key files**: `src/crewai/crew.py` (orchestration), `src/crewai/memory/` (unified memory, recall_flow), `src/crewai/knowledge/` (knowledge storage), `src/crewai/utilities/guardrail.py`
**Author score**: DKS=15, RSS=9, Total=24, Stages=7

---

## System 4: LlamaIndex

**Repository**: https://github.com/run-llama/llama_index
**Key files**: `llama-index-core/.../indices/` (index types), `llama-index-core/.../retrievers/` (retrieval), `llama-index-core/.../postprocessor/` (PII masking), `llama-index-core/.../evaluation/`
**Note**: Framework — score as typical production deployment, not framework ceiling.
**Author score**: DKS=20, RSS=7, Total=27, Stages=6

---

## System 5: BIRD-SQL

**Repository**: https://github.com/AlibabaResearch/DAMO-ConvAI (bird/ subdirectory)
**Paper**: Li et al., "Can LLM Already Serve as A Database Interface?," NeurIPS 2024
**Key files**: `finetuning/tasks/bird.py`, `finetuning/seq2seq_construction/bird.py`, `llm/src/gpt_request.py`, `llm/src/evaluation.py`
**Author score**: DKS=26, RSS=3, Total=29, Stages=7

---

## System 6: Wheatley

**Repository**: https://github.com/jolibrain/wheatley
**Paper**: Musik et al., "Learning to Solve Job Shop Scheduling under Uncertainty," CPAIOR 2024
**Key files**: `args.py` (800+ lines), `jssp/env/state.py` (1000+ lines), `psp/graph/dgl_graph.py`, `jssp/dispatching_rules/validate.py`
**Author score**: DKS=33, RSS=2, Total=35, Stages=8

---

## System 7: FinSQL

**Repository**: https://github.com/bigbigwatermalon/FinSQL
**Paper**: Li et al., "FinSQL: Model-Agnostic LLMs-Based Text-to-SQL Framework," SIGMOD Companion 2024
**Key files**: `Parallel_Cross_Encoder/`, `Hybrid_Data_Augmentation/`, `LoRA_Fine-Tuning/`, `dataset/BULL-en/tables.json`
**Author score**: DKS=35, RSS=6, Total=41, Stages=7

---

## System 8: MetaGPT

**Repository**: https://github.com/FoundationAgents/MetaGPT
**Paper**: Hong et al., "MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework," ICLR 2024 (Oral)
**Key files**: `metagpt/roles/role.py` (base role class, SOP definitions), `metagpt/roles/` (23 role files: architect, product_manager, engineer, qa_engineer, etc.), `metagpt/schema.py` (message/action schemas), `metagpt/memory/` (long-term, brain, role-zero memory), `metagpt/actions/` (action definitions per role), `metagpt/environment/` (team environment)
**Note**: Role-based multi-agent framework. D1: role vocabulary and SOP terminology. D2: workflow/dependency graph between roles. D4: message/action schemas. D5: role profiles (23+ defined roles). D6: constraint rules in SOP templates. Scoring focus: role-specification templates as domain artifacts vs. general-purpose framework scaffolding.
**Author score**: DKS=37, RSS=4, Total=41, Stages=6

*(Use scoring template above)*

---

## System 9: SpeechBrain

**Repository**: https://github.com/speechbrain/speechbrain
**Paper**: Ravanelli et al., "SpeechBrain: A General-Purpose Speech Toolkit," arXiv:2106.04624, 2021
**Key files**: `speechbrain/decoders/seq2seq.py` (77KB), `speechbrain/processing/features.py`, `speechbrain/tokenizers/SentencePiece.py`, `recipes/` (43 datasets)
**Author score**: DKS=37, RSS=4, Total=41, Stages=8

---

## System 10: AudioCraft

**Repository**: https://github.com/facebookresearch/audiocraft
**Paper**: Copet et al., "Simple and Controllable Music Generation," NeurIPS 2023
**Key files**: `audiocraft/models/lm.py`, `audiocraft/modules/conditioners.py` (1700+ lines), `audiocraft/data/music_dataset.py`, `config/` (68 YAML files), `audiocraft/metrics/`
**Author score**: DKS=42, RSS=4, Total=46, Stages=7

---

## System 11: DeepTutor

**Repository**: https://github.com/HKUDS/DeepTutor
**Paradigm**: Educational AI (agent-native personalized tutoring)
**Key files**: `deeptutor/agents/` (question coordinator, visualization pipeline), `deeptutor/knowledge/` (knowledge manager, initializer, progress_tracker, add_documents), `deeptutor/book/` (content engine, compiler, kb_health), `deeptutor/runtime/orchestrator.py` (main orchestration), `deeptutor/tutorbot/` (tutoring logic), `deeptutor/capabilities/` (learner profiling)
**Note**: Full-stack tutoring application (~120K Python lines). Scoring focus: D1 (subject taxonomy/terminology), D2 (knowledge graph of concepts), D4 (curriculum/content schema), D5 (learner profiles), D6 (pedagogical rules), D7 (multilingual support if present).
**Author score**: DKS=42, RSS=6, Total=48, Stages=8

*(Use scoring template above)*

---

## System 12: EmotiVoice

**Repository**: https://github.com/netease-youdao/EmotiVoice
**Key files**: `frontend.py` + `frontend_cn.py` + `frontend_en.py` (G2P pipeline), `models/prompt_tts_modified/simbert.py` (style encoder), `text/cmudict.py` (84 ARPAbet symbols), `data/youdao/text/` (emotion/pitch/energy/speed labels, 2013 speakers), `cn2an/conf.py` (Chinese numeral system)
**Author score**: DKS=47, RSS=3, Total=50, Stages=8

---

## System 13: GraphRAG

**Repository**: https://github.com/microsoft/graphrag
**Paper**: Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization," arXiv:2404.16130, 2024
**Key files**: `graphrag/index/workflows/factory.py` (10-stage pipeline), `graphrag/data_model/` (7 model classes), `graphrag/query/structured_search/` (4 search modes), `graphrag/graphs/hierarchical_leiden.py`, `graphrag/config/`
**Author score**: DKS=57, RSS=8, Total=65, Stages=10

---

## Rater Information

- **Rater Name**: _______________
- **Affiliation**: _______________
- **Date of Scoring**: _______________
- **Time Spent per System** (minutes):

| System | Time | Prior Familiarity |
|--------|------|-------------------|
| MedRAG | ___ | None / Read paper / Used system |
| Pulse AI | ___ | None / Read paper / Used system |
| CrewAI | ___ | None / Read paper / Used system |
| LlamaIndex | ___ | None / Read paper / Used system |
| BIRD-SQL | ___ | None / Read paper / Used system |
| Wheatley | ___ | None / Read paper / Used system |
| FinSQL | ___ | None / Read paper / Used system |
| MetaGPT | ___ | None / Read paper / Used system |
| SpeechBrain | ___ | None / Read paper / Used system |
| AudioCraft | ___ | None / Read paper / Used system |
| DeepTutor | ___ | None / Read paper / Used system |
| EmotiVoice | ___ | None / Read paper / Used system |
| GraphRAG | ___ | None / Read paper / Used system |

## Notes

_Space for additional observations or scoring difficulties encountered._
