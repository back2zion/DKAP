# Supplementary Material: External Open-Source System Analysis

## Purpose

This document provides detailed source-code-level evidence for the DKAP layer mapping, DKS scoring, and RSS scoring of six independently developed open-source systems included for external validation: MedRAG, FinSQL, Wheatley, BIRD-SQL, LlamaIndex, and SpeechBrain. All file paths reference the publicly available GitHub repositories cloned as of March 2026.

---

## S1. MedRAG (ACL 2024 Findings)

**Repository**: https://github.com/Teddy-XiongGZ/MedRAG
**Paper**: Xiong et al., "Benchmarking Retrieval-Augmented Generation for Medicine," Findings of ACL 2024.

### S1.1 Pipeline Stages (4 stages)

| Stage | Component | Evidence File | Description |
|-------|-----------|---------------|-------------|
| 1 | Corpus Loading | `src/medrag.py:L28-45` | Load pre-indexed corpus (PubMed/StatPearls/Textbooks/Wikipedia) |
| 2 | Retrieval | `src/medrag.py:L47-89` | BM25/Contriever/SPECTER/MedCPT retrieval + RRF fusion |
| 3 | LLM Inference | `src/medrag.py:L91-130` | GPT-4/Mixtral/Llama with CoT prompting |
| 4 | Answer Extraction | `src/medrag.py:L132-160` | Regex-based answer parsing from LLM output |

### S1.2 DKS Scoring (Total: 5)

| Component | Score | Evidence |
|-----------|-------|----------|
| D1: Domain lexicon | 0 | No explicit medical glossary file. Medical terms handled by pre-trained retriever embeddings (MedCPT). |
| D2: Ontology/KG | 0 | No ontology graph. Corpora are flat document collections without hierarchical structuring. |
| D3: Cross-ontology mapping | 0 | No mapping between medical coding systems. |
| D4: Schema/DDL | 0 | No database schema. Document-based retrieval only. |
| D5: Domain profile count | 5 | 4 corpus types (PubMed, StatPearls, Textbooks, Wikipedia) with distinct chunking strategies. `src/medrag.py:L28-35` corpus configuration. |
| D6: Code-level rules | 0 | No domain-specific regex/rule patterns. |
| D7: Multi-language | 0 | English only. |
| D8: Regulatory/compliance | 0 | No regulatory encoding. |

**Conservative scoring note**: MedCPT embeddings encode implicit medical knowledge, but DKS measures *explicit, structured* domain artifacts in code. Implicit knowledge in pre-trained models is excluded from DKS by design.

### S1.3 RSS Scoring (Total: 5)

| Component | Score | Evidence |
|-----------|-------|----------|
| R1: Intent classification | 0 | No intent classification. Direct question-to-retrieval. |
| R2: Vector search sophistication | 3 | 4 retrievers (BM25+Contriever+SPECTER+MedCPT) with RRF fusion + FAISS HNSW indexing. `src/medrag.py:L47-89`. |
| R3: Permission-based filtering | 0 | No access control. |
| R4: Multi-index fusion | 2 | RRF (Reciprocal Rank Fusion) across multiple retriever outputs. `src/medrag.py:L75-89`. |
| R5: Adaptive routing | 0 | Fixed retrieval pipeline (no dynamic routing). |

### S1.4 DKAP Layer Mapping

| Layer | Implementation | Evidence |
|-------|---------------|----------|
| L1 | Question preprocessing (minimal) | `src/medrag.py:L20-27` |
| L2 | MedCorp: 4 corpora chunked into snippets (pre-indexed) | `src/medrag.py:L28-45`, corpus config |
| L3 | 4 retrievers + RRF fusion + FAISS HNSW | `src/medrag.py:L47-89` |
| L4 | GPT-4/Mixtral/Llama + CoT prompting (6 LLMs supported) | `src/medrag.py:L91-130` |
| L5 | MIRAGE benchmark (7,663 questions, 5 medical QA datasets) | `src/medrag.py:L132-160`, `evaluation/` |

---

## S2. FinSQL (SIGMOD 2024 Companion)

**Repository**: https://github.com/bigbigwatermalon/FinSQL
**Paper**: Li et al., "FinSQL: Model-Agnostic LLMs-Based Text-to-SQL Framework for Financial Analysis," SIGMOD Companion 2024.

### S2.1 Pipeline Stages (7 stages)

| Stage | Component | Evidence File | Description |
|-------|-----------|---------------|-------------|
| 1 | Dataset Preprocessing | `Parallel_Cross_Encoder/scripts/preprocessing_finsql.sh` | Normalize SQL, extract schema labels, FK injection |
| 2 | Schema Classifier Training | `Parallel_Cross_Encoder/scripts/train_text2sql_schema_item_classifier_finsql.sh` | Train RoBERTa-large Cross-Encoder for schema ranking |
| 3 | Schema Prediction | `Parallel_Cross_Encoder/scripts/generate_text2sql_dataset_finsql.sh` | Generate ranked schema (top-k tables/columns) |
| 4 | Hybrid Data Augmentation | `Hybrid_Data_Augmentation/scripts/hybrid_augmentation.sh` | GPT-3.5 COT/Synonymous/Skeleton generation |
| 5 | LLM LoRA Fine-tuning | `LoRA_Fine-Tuning/ds_sft.sh` | Llama2-13B LoRA training on augmented data |
| 6 | Inference + Post-processing | `LoRA_Fine-Tuning/src/eval_text2sql.py`, `src/utils/sql_post_process.py` | SQL generation + fuzzy schema matching |
| 7 | Self-consistency Voting | `LoRA_Fine-Tuning/src/utils/self_consistency.py` | 8-way semantic SQL comparison |

### S2.2 DKS Scoring (Total: 35)

| Component | Score | Evidence |
|-----------|-------|----------|
| D1: Domain lexicon | 10 | 78 financial tables across 3 databases. Column names encode financial terminology: `mainoperincome`, `grossprofit`, `fundreturn`, `benchgrforthisweek`, `pctofpledger`, `adjfreefloatratio`, `totalashare`. `dataset/BULL-en/tables.json`: 1,170 column names. |
| D2: Ontology/KG | 5 | 3-tier database hierarchy (ccks_stock/ccks_fund/ccks_macro) with FK relationships. No explicit graph/ontology beyond relational schema. |
| D3: Cross-ontology mapping | 5 | Bilingual CN/EN parallel corpus (3,966 identical question pairs). Cross-DB company code linking (stock↔fund via CompanyCode/PersonalCode). |
| D4: Schema/DDL | 10 | ccks_stock: 31 tables, 434 columns, 465 FKs. ccks_fund: 28 tables, 461 columns, 149 FKs. ccks_macro: 19 tables, 275 columns, 0 FKs. Total: 78 tables, 1,170 columns, 614 FKs. `dataset/BULL-en/tables.json`. |
| D5: Domain profile count | 3 | 3 augmentation profiles (COT/Synonymous/Skeleton). `Hybrid_Data_Augmentation/configures/templates.json`. |
| D6: Code-level rules | 0 | No explicit regex/rule patterns. Schema matching uses fuzzy string matching (fuzzywuzzy), not hand-coded rules. |
| D7: Multi-language | 2 | Bilingual (CN/EN). `dataset/BULL-cn/`, `dataset/BULL-en/`. |
| D8: Regulatory/compliance | 0 | No regulatory encoding. |

**Conservative scoring note**: Agent-level analysis yielded DKS=47. The published score of 35 reflects conservative calibration: D1 capped at 10 (column names are domain terms but not a curated glossary), D2 at 5 (relational schema without explicit KG), D7 at 2 (bilingual but not trilingual). This conservative approach is applied consistently across all external systems.

### S2.3 RSS Scoring (Total: 6)

| Component | Score | Evidence |
|-----------|-------|----------|
| R1: Intent classification | 0 | No intent classification. Direct NL→SQL pipeline. |
| R2: Vector search sophistication | 3 | RoBERTa-large Cross-Encoder with BiLSTM + 8-head Cross-Attention for schema item ranking. `Parallel_Cross_Encoder/utils/classifier_model.py`. Focal loss (gamma=2.0, alpha=0.75). Not vector search per se, but learned semantic ranking. |
| R3: Permission-based filtering | 0 | No access control. |
| R4: Multi-index fusion | 1 | Dual ranking: table-level + column-level probability scores combined for final schema selection. `Parallel_Cross_Encoder/text2sql_data_generator_finsql.py`. |
| R5: Adaptive routing | 2 | Learned Cross-Encoder dynamically selects top-k tables (k1=3) and columns (k2=7) per query. Training with noise injection (rate=0.2) for robustness. |

### S2.4 DKAP Layer Mapping

| Layer | Implementation | Evidence |
|-------|---------------|----------|
| L1 | SQL whitespace/lowercase normalization | `Parallel_Cross_Encoder/preprocessing_finsql.py` |
| L2 | BULL financial schema: 78 tables, 1,170 cols, 614 FKs, 3-tier DB (stock/fund/macro), bilingual CN/EN | `dataset/BULL-en/tables.json`, `dataset/BULL-cn/tables.json` |
| L3 | RoBERTa-large + BiLSTM Cross-Encoder schema ranking (Top-k tables/columns) | `Parallel_Cross_Encoder/schema_item_classifier_finsql.py`, `utils/classifier_model.py` |
| L4 | LLaMA2-13B + LoRA (8 projection targets), 3-variant augmentation (COT/Skeleton/Synonymous) | `LoRA_Fine-Tuning/ds_sft.sh`: `--model_name_or_path models/llama2_13B`, `--dataset bull_en_cot,bull_en_skeleton,bull_en_synonymous` |
| L5 | Fuzzy schema matching (fuzzywuzzy) + SQL exec validation (60s timeout) + 8-way self-consistency voting | `LoRA_Fine-Tuning/src/utils/sql_post_process.py`, `self_consistency.py` |

---

## S3. Wheatley (CPAIOR 2024)

**Repository**: https://github.com/jolibrain/wheatley
**Paper**: Musik et al., "Learning to Solve Job Shop Scheduling under Uncertainty," CPAIOR 2024.

### S3.1 Pipeline Stages (8 stages)

| Stage | Component | Evidence File | Description |
|-------|-----------|---------------|-------------|
| 1 | Instance Parsing | `jssp/utils/utils.py:load_taillard_problem()` | Taillard/SM format → affectations[n_j,n_m], durations[n_j,n_m,4] |
| 2 | State Initialization | `jssp/env/state.py:__init__()` | Precedence graph + resource constraint initialization |
| 3 | Feature Extraction | `jssp/env/observation.py` | Node features: TCT, is_affected, selectable, machine_id, mwkr (6+max_m dimensions) |
| 4 | Graph Construction | `psp/graph/dgl_graph.py` | Heterograph: prec/rprec/rc/rp/rrp/pool/self edge types |
| 5 | GNN Inference | `jssp/models/gnn_dgl.py`, `psp/models/gnn_mp.py` | DGL (GIN/GATv2/PNA/GCN2/PDF) or TokenGT → node embeddings |
| 6 | Action Masking + Selection | `jssp/models/agent.py:get_action_and_value()` | Actor MLP → logits → valid action mask → categorical sampling |
| 7 | PPO Training | `alg/ppo.py:train()` | Rollout collection → GAE → clipped surrogate update |
| 8 | Validation + Benchmark | `jssp/dispatching_rules/validate.py`, `generic/agent_validator.py` | Feasibility check + makespan + OR-Tools/heuristic comparison |

### S3.2 DKS Scoring (Total: 33)

| Component | Score | Evidence |
|-----------|-------|----------|
| D1: Domain lexicon | 5 | Scheduling domain terms: makespan, tardiness, precedence, affectation, selectable, resource capacity, conflict clique, frontier, renewable/non-renewable resource. ~30-40 terms across `args.py` (800+ lines of domain parameters). |
| D2: Ontology/KG | 5 | Multi-relational graph with edge types encoding domain constraints. `psp/graph/dgl_graph.py`: prec, rprec, rc, rp, rrp, pool, self (7 edge types). However, scored conservatively at 5 as these are computational graph edges, not a semantic ontology. |
| D3: Cross-ontology mapping | 5 | JSSP ↔ RCPSP shared abstractions (state, transition, reward interfaces). Taillard ↔ SM ↔ Patterson format conversions. `jssp/description.py` vs `psp/description.py`. |
| D4: Schema/DDL | 5 | State representation: `jssp/env/state.py` (520+ lines), `psp/env/state.py` (600+ lines). Node features (6+max_m dimensions), edge attributes (rid, val, valr). Scored conservatively at 5 as this is tensor structure, not relational DDL. |
| D5: Domain profile count | 6 | Reward models: 5 (Sparse, L2D, Tassel, Intrinsic, Uncertain). Transition models: 3 (Simple, L2D, SlotLocking). Dispatching heuristics: 5 (SPT, MWKR, MOPNR, FDD/MWKR, Random). Total: 13 profiles. |
| D6: Code-level rules | 6 | Action masking rules (`state.py:get_selectable()`), precedence enforcement, resource capacity constraints (`psp/utils/resource_flowgraph.py`), fast-forward rules, observation horizon filtering. ~15-20 rule patterns. |
| D7: Multi-language | 0 | English/Python only. |
| D8: Regulatory/compliance | 1 | Feasibility constraints are domain-mandated: job precedence ordering, machine non-overlap, resource capacity limits. These are operational constraints analogous to compliance rules. Scored conservatively at 1. |

**Conservative scoring note**: Detailed code analysis yields DKS=42 when D2 and D4 are scored at their full evidence level (D2=10 for 7 edge types + hierarchical GNN; D4=10 for 520+ lines of constraint propagation). The published score of 33 reflects conservative calibration: graph edges scored as computational structure (5) rather than semantic ontology (10), and tensor-based state scored as data structure (5) rather than relational schema (10). This conservative approach is consistent with FinSQL scoring.

### S3.3 RSS Scoring (Total: 2)

| Component | Score | Evidence |
|-----------|-------|----------|
| R1: Intent classification | 0 | No intent classification. Fixed RL episode structure. |
| R2: Vector search sophistication | 0 | No vector search. Graph-based state representation. |
| R3: Permission-based filtering | 0 | No access control. |
| R4: Multi-index fusion | 0 | Single graph observation. |
| R5: Adaptive routing | 2 | GNN-based dynamic action selection: actor MLP outputs action probabilities conditioned on current graph state. Action masking enforces domain constraints. `jssp/models/agent.py:get_action_and_value()`. |

### S3.4 DKAP Layer Mapping

| Layer | Implementation | Evidence |
|-------|---------------|----------|
| L1 | JSSP: Taillard format parsing. PSP: SM/PSPLib format parsing with precedence DAG extraction. | `jssp/utils/utils.py:load_taillard_problem()`, `psp/description.py` |
| L2 | 6 constraint types encoded in state: (1) job precedence, (2) machine affectation, (3) resource capacity (renewable), (4) resource capacity (non-renewable), (5) temporal constraints (duration bounds), (6) mode selection (PSP multi-mode). State graph: 16-28 features/node depending on configuration. | `jssp/env/state.py` (520+ lines), `psp/env/state.py` (600+ lines), `psp/graph/dgl_graph.py` |
| L3 | Graph construction with 7 edge types (prec/rprec/rc/rp/rrp/pool/self) + observation horizon filtering. GNN extracts node embeddings for action routing. | `psp/graph/dgl_graph.py`, `jssp/env/observation.py` |
| L4 | PPO + 4 GNN variants: (1) DGL with 5 conv types (GIN/GATv2/PNA/GCN2/PDF), (2) TokenGT transformer, (3) GnnFlat, (4) GnnHier. ONNX export supported. | `alg/ppo.py`, `jssp/models/gnn_dgl.py`, `generic/tokengt/` |
| L5 | Episode termination feasibility check: `validate_job_tasks()` (precedence ordering), `validate_machine_tasks()` (non-overlap). Makespan computation. Baseline comparison: OR-Tools + 5 dispatching heuristics. | `jssp/dispatching_rules/validate.py`, `generic/agent_validator.py` |

### S3.5 Constraint Type Detail

| Constraint | JSSP Implementation | PSP Implementation |
|------------|--------------------|--------------------|
| Precedence | Implicit in job task ordering: task (j,t) must precede (j,t+1) | Explicit DAG from problem file: `problem_edges` |
| Machine/Resource | Machine conflicts: tasks on same machine cannot overlap | Renewable resource capacity: `sum(consumption) <= capacity` at each timestep |
| Temporal | Duration bounds: [optimistic, pessimistic, mode] for stochastic problems | Same + mode-dependent durations |
| Mode selection | N/A (fixed affectation) | Multi-mode: each job has multiple execution modes with different duration/resource tradeoffs |

### S3.6 GNN Architecture Variants

| Variant | Architecture | Problem Type | Key File |
|---------|-------------|--------------|----------|
| DGL | GIN/GATv2/PNA/GCN2/PDF conv layers + edge embedding | JSSP | `jssp/models/gnn_dgl.py` |
| TokenGT | Transformer with Laplacian/RWPE positional encoding | JSSP/PSP | `generic/tokengt/tokengt_graph_encoder.py` |
| GnnFlat | Flat message passing (homogeneous graph) | PSP | `psp/models/gnn_flat.py` |
| GnnHier | Hierarchical aggregation (multi-scale) | PSP | `psp/models/gnn_hier.py` |

---

## S4. BIRD-SQL (VLDB 2024)

**Repository**: https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird
**Paper**: Li et al., "Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs," NeurIPS 2024.

### S4.1 Pipeline Stages (7 stages)

| Stage | Component | Evidence File | Description |
|-------|-----------|---------------|-------------|
| 1 | Input Loading | `finetuning/tasks/bird.py:L38-62` | Load dataset JSON with questions, schemas, databases |
| 2 | Schema Extraction | `finetuning/tasks/bird.py:L92-123` | Extract database metadata (tables, columns, types, PK/FK) with per-DB caching |
| 3 | Schema Serialization | `finetuning/seq2seq_construction/bird.py:L212-286` | Convert schema to prompt format (peteshaw/verbose/NL) |
| 4 | Prompt Engineering | `llm/src/gpt_request.py:L138-147` | Combine question + schema + external knowledge into input |
| 5 | Model Inference | `llm/src/gpt_request.py:L159-165` | GPT (code-davinci/gpt-3.5-turbo) or T5-large fine-tuning |
| 6 | SQL Post-processing | `llm/src/post_process_cot.py:L5-22` | Extract SQL from CoT output, normalize whitespace/case |
| 7 | Execution Evaluation | `llm/src/evaluation.py:L17-28`, `evaluation_ves.py:L26-52` | EX (result set comparison) + VES (efficiency scoring) |

### S4.2 DKS Scoring (Total: 26)

| Component | Score | Evidence |
|-----------|-------|----------|
| D1: Domain lexicon | 0 | Custom special tokens (`<`, `<=`) added to tokenizer only. No curated domain glossary. `finetuning/models/unified/finetune.py:L21-23`. |
| D2: Ontology/KG | 5 | Database schema used as implicit ontology: tables, columns, types, PK, FK. No explicit knowledge graph or semantic hierarchy. `finetuning/tasks/bird.py:L38-62`. |
| D3: Cross-ontology mapping | 5 | Three schema serialization formats (peteshaw, verbose, natural language) represent the same schema in different modalities. `finetuning/seq2seq_construction/bird.py:L212-286`. Cross-format mapping, not cross-domain. |
| D4: Schema/DDL | 10 | **95 databases spanning 37 professional domains**, 33.4 GB total. Variable per-database complexity with tables, typed columns, PK/FK constraints. `materials/` dataset directory. Scored at 10 for breadth across domains. |
| D5: Domain profile count | 3 | Value sampling from DB columns (`get_database_matches()`), few-shot templates with/without external knowledge, evidence integration mode. `llm/src/gpt_request.py:L113-134`. |
| D6: Code-level rules | 3 | SQL normalization rules: lowercase (excluding quoted strings), comma fixing, whitespace normalization. SQL keyword escaping (`order`, `by`, `group` → backtick-quoted). Stop token rules: `['--', '\n\n', ';', '#']`. `finetuning/seq2seq_construction/bird.py:L119-134`, `llm/src/gpt_request.py:L82-83`. |
| D7: Multi-language | 0 | English only. All prompts, documentation, and database values in English. |
| D8: Regulatory/compliance | 0 | CC BY-SA 4.0 license. No regulatory encoding. |

**Conservative scoring note**: BIRD-SQL's primary strength is schema breadth (95 DBs, 37 domains) rather than depth in any single domain. D4 scored at 10 reflects multi-domain coverage; individual database schemas are simpler than FinSQL's financial schema (78 tables, 614 FKs). D3 scored at 5 for format mapping (peteshaw/verbose/NL), not cross-domain ontology alignment.

### S4.3 RSS Scoring (Total: 3)

| Component | Score | Evidence |
|-----------|-------|----------|
| R1: Intent classification | 0 | No intent classification. Direct question-to-SQL pipeline. |
| R2: Vector search sophistication | 1 | `bridge_content_encoder` performs string-based value matching from DB columns to question terms. Not neural embedding search. `finetuning/seq2seq_construction/bird.py:L183-190`. |
| R3: Permission-based filtering | 0 | No access control. |
| R4: Multi-index fusion | 0 | Single schema serialization per query. No multi-retriever fusion. |
| R5: Adaptive routing | 2 | Dual-pipeline architecture: ICL (GPT) and fine-tuning (T5) with configurable inference parameters. Value sampling dynamically adapts schema representation based on `schema_serialization_with_db_content` flag. |

### S4.4 DKAP Layer Mapping

| Layer | Implementation | Evidence |
|-------|---------------|----------|
| L1 | SQL normalization: lowercase, comma fix, whitespace. SQL keyword escaping for reserved words. | `finetuning/seq2seq_construction/bird.py:L119-134`, `llm/src/gpt_request.py:L82-83` |
| L2 | 95 databases, 37 domains. Schema = tables + typed columns + PK/FK. External knowledge ("evidence") as optional textual hints. Three serialization formats. | `finetuning/tasks/bird.py:L92-123`, `finetuning/seq2seq_construction/bird.py:L137-286` |
| L3 | `bridge_content_encoder` for value-based schema linking. Direct DB value extraction via SQLite. | `finetuning/seq2seq_construction/bird.py:L183-190`, `llm/src/gpt_request.py:L80-97` |
| L4 | Dual pipeline: (1) GPT ICL (code-davinci-002, text-davinci-003, gpt-3.5-turbo) with CoT, (2) T5-large/3b fine-tuning (AdaFactor, lr=5e-5, 200 epochs). | `llm/src/gpt_request.py:L159-165`, `finetuning/train_bird.py` |
| L5 | Execution Accuracy (EX): result set comparison with 30s timeout. Valid Efficiency Score (VES): execution time ratio with 3-sigma outlier rejection. Difficulty stratification (simple/moderate/challenging). Parallel evaluation (16 CPUs). | `llm/src/evaluation.py:L17-48`, `llm/src/evaluation_ves.py:L26-52` |

### S4.5 BIRD-SQL vs. FinSQL: Same Paradigm, Different DKS

Both systems are Text-to-SQL, yet exhibit different DKS profiles:

| Dimension | BIRD-SQL (DKS 26) | FinSQL (DKS 35) | Interpretation |
|-----------|-------------------|-----------------|----------------|
| D1: Lexicon | 0 (no curated glossary) | 10 (1,170 financial column names) | Domain-specific terminology drives D1 |
| D4: Schema | 10 (95 DBs, 37 domains, breadth) | 10 (78 tables, 614 FKs, depth) | Breadth vs. depth yield comparable D4 |
| D3: Cross-mapping | 5 (format serialization) | 5 (bilingual CN/EN) | Different cross-mapping strategies |
| D6: Rules | 3 (SQL normalization) | 0 (fuzzy matching, no regex) | Rule-based vs. ML-based post-processing |
| Total DKS | **26** | **35** | Domain depth > domain breadth for DKS |

This comparison demonstrates that **domain specificity (depth)** contributes more to DKS than **domain coverage (breadth)**, supporting the DKAP hypothesis that structured domain knowledge, not general-purpose schema handling, drives system complexity.

---

## S5. LlamaIndex (Open-Source Framework)

**Repository**: https://github.com/run-llama/llama_index
**Documentation**: LlamaIndex is a data framework for LLM-based applications, providing tools to ingest, structure, and query domain data.

**Important methodological note**: LlamaIndex is a *framework/toolkit*, not a single-purpose pipeline. DKS scoring reflects built-in capabilities available out-of-the-box, not a specific deployment instance. We additionally report "Typical Deployment DKS" to estimate the score of a representative production deployment.

### S5.1 Pipeline Stages (6 stages, modular)

| Stage | Component | Evidence File | Description |
|-------|-----------|---------------|-------------|
| 1 | Ingestion & Parsing | `llama-index-core/.../ingestion/pipeline.py` | Node parsing (Sentence/Token/Semantic/Code splitters) + metadata extraction (Title/Keywords/Summary/Entity) |
| 2 | Index Construction | `llama-index-core/.../indices/` | VectorStoreIndex, PropertyGraphIndex, SQLStructStoreIndex, MultiModalVectorStoreIndex, KeywordTableIndex |
| 3 | Retrieval | `llama-index-core/.../retrievers/` | Vector, KG pattern matching, Fusion (RRF/relative/distance), Recursive, Router, Auto (metadata filtering) |
| 4 | Post-processing | `llama-index-core/.../postprocessor/` | LLMRerank, SentenceTransformerRerank, RankGPTRerank, NodeRecencyPostprocessor, PresidioPIIMasking |
| 5 | LLM Inference + Synthesis | `llama-index-core/.../response_synthesizers/` | 30+ LLM providers. Synthesis modes: Refine, TreeSummarize, CompactAndRefine, Accumulate, Generation |
| 6 | Evaluation | `llama-index-core/.../evaluation/` | Correctness, Faithfulness, Relevancy, SemanticSimilarity evaluators. BatchEvalRunner. |

### S5.2 DKS Scoring — Framework Capability (Total: 41) / Typical Deployment (Total: ~20)

| Component | Framework Score | Typical Deploy | Evidence |
|-----------|----------------|----------------|----------|
| D1: Domain lexicon | 5 | 0 | `EntityExtractor` with `DEFAULT_ENTITY_MAP` (9 entity types: PER, ORG, LOC, DIS, BIO, etc.) via multinerd NER. Available but rarely deployed with full entity mapping. `llama-index-integrations/extractors/.../entity/base.py`. |
| D2: Ontology/KG | 10 | 5 | **PropertyGraphIndex**: EntityNode + Relation graph with LLM-based triplet extraction (SimpleLLMPathExtractor, SchemaLLMPathExtractor, DynamicLLMPathExtractor). Supports Neo4j, Neptune, Nebula backends. `llama-index-core/.../indices/property_graph/`. Typical deployments use VectorStoreIndex only. |
| D3: Cross-ontology mapping | 5 | 3 | ComposableGraph (multi-index linking), QueryFusionRetriever (RRF/relative/distance fusion), MultiModalVectorStoreIndex (text+image cross-modal). `llama-index-core/.../indices/composability/`, `retrievers/fusion_retriever.py`. |
| D4: Schema/DDL | 5 | 3 | SQLStructStoreIndex + SQLContextContainer + NLSQLTableEngine for database schema handling. JSONQueryEngine for structured documents. `llama-index-core/.../indices/struct_store/`. |
| D5: Domain profile count | 3 | 2 | Global Settings system + 50+ prompt templates in `default_prompts.py` (1000+ lines). PromptHelper for domain customization. Typical deployments customize 3-5 templates. |
| D6: Code-level rules | 3 | 2 | PydanticOutputParser (JSON schema validation), StructuredLLM (schema-constrained generation). `llama-index-core/.../output_parsers/pydantic.py`. |
| D7: Multi-language | 5 | 0 | EntityExtractor multinerd supports 10+ languages. Architecture is language-agnostic (Unicode, configurable tokenizer). Most deployments are English-only. |
| D8: Regulatory/compliance | 5 | 5 | PresidioPostprocessor for PII detection/masking (Microsoft Presidio integration). StorageContext abstraction supports encrypted/remote storage. Instrumentation framework for audit trails. `llama-index-integrations/postprocessor/.../presidio/`. |

**Scoring rationale**: The "Framework Score" (41) represents the ceiling of built-in capabilities. The "Typical Deployment Score" (~20) estimates what a representative production RAG application would activate. For Table I, we use **DKS = 20** (typical deployment) to maintain comparability with the other single-purpose systems analyzed.

### S5.3 RSS Scoring — Typical Deployment (Total: 7)

| Component | Score | Evidence |
|-----------|-------|----------|
| R1: Intent classification | 0 | RouterQueryEngine can route queries to different indices, but no explicit intent taxonomy. |
| R2: Vector search sophistication | 3 | VectorIndexRetriever + 40+ embedding integrations. AutoRetriever adds metadata filtering on top of vector search. `llama-index-core/.../retrievers/`. |
| R3: Permission-based filtering | 0 | No built-in RBAC. Metadata filtering can approximate access control but is not designed for it. |
| R4: Multi-index fusion | 2 | QueryFusionRetriever with 3 fusion modes (RECIPROCAL_RANK, RELATIVE_SCORE, DIST_BASED_SCORE). RecursiveRetriever for hierarchical traversal. `llama-index-core/.../retrievers/fusion_retriever.py`. |
| R5: Adaptive routing | 2 | RouterQueryEngine + LLMSingleSelector/EmbeddingSelector for dynamic query routing. SubQuestionQueryEngine for query decomposition. |

### S5.4 DKAP Layer Mapping (Typical Deployment)

| Layer | Implementation | Evidence |
|-------|---------------|----------|
| L1 | SentenceSplitter/TokenTextSplitter + TitleExtractor/KeywordsExtractor + IngestionCache with SHA256 deduplication | `llama-index-core/.../node_parser/`, `ingestion/pipeline.py` |
| L2 | VectorStoreIndex (typical) or PropertyGraphIndex (advanced). EntityExtractor for NER. SQLStructStoreIndex for database schemas. 50+ prompt templates. | `llama-index-core/.../indices/` |
| L3 | VectorIndexRetriever + optional QueryFusionRetriever (RRF). LLMRerank/SentenceTransformerRerank post-processing. | `llama-index-core/.../retrievers/`, `postprocessor/` |
| L4 | 30+ LLM providers (OpenAI, Anthropic, Cohere, Mistral, Ollama, vLLM). StructuredLLM for schema-constrained output. Function calling support. | `llama-index-core/.../llms/`, `llama-index-integrations/llms/` |
| L5 | Response synthesis (Refine/TreeSummarize/Compact). PydanticOutputParser for JSON validation. Evaluation suite: Correctness, Faithfulness, Relevancy, SemanticSimilarity. PresidioPII for compliance. | `llama-index-core/.../response_synthesizers/`, `evaluation/`, `output_parsers/` |

### S5.5 LlamaIndex as L2 Spectrum Demonstrator

LlamaIndex uniquely illustrates the DKAP framework's sensitivity to L2 configuration depth:

| Deployment Profile | L2 Configuration | Estimated DKS |
|-------------------|-----------------|---------------|
| Minimal (VectorStore only) | Flat chunking → vector index | ~8 |
| Standard (VectorStore + metadata) | Chunking + keyword/entity extraction → vector + metadata filtering | ~15 |
| Structured (PropertyGraph) | LLM triplet extraction → KG index + Cypher queries | ~25 |
| Full (PropertyGraph + SQL + PII) | KG + SQL schema + Presidio PII + multi-index fusion | ~41 |

This gradient supports the DKAP claim that **L2 structuring depth is the primary driver of system complexity**, as the same framework produces radically different DKS scores depending on how domain knowledge is structured.

---

## S6. SpeechBrain (Open-Source Toolkit)

**Repository**: https://github.com/speechbrain/speechbrain
**Paper**: Ravanelli et al., "SpeechBrain: A General-Purpose Speech Toolkit," arXiv:2106.04624, 2021.

**Important methodological note**: SpeechBrain is a non-LLM speech processing toolkit. Its inclusion demonstrates DKAP's AI-engine agnosticism beyond the LLM-dominated landscape. DKS scoring is based on the framework's built-in capabilities as deployed in a representative ASR recipe (LibriSpeech seq2seq with BPE-1000).

### S6.1 Pipeline Stages (8 stages)

| Stage | Component | Evidence File | Description |
|-------|-----------|---------------|-------------|
| 1 | Audio Loading & Resampling | `speechbrain/dataio/audio_io.py` | WAV input → waveform tensor at target sample rate (16kHz) |
| 2 | Feature Extraction | `speechbrain/processing/features.py` | STFT (n_fft=400) → Mel Filterbank (40 bins) → log magnitude → optional MFCC/Deltas |
| 3 | Data Augmentation | `speechbrain/augment/augmenter.py`, `freq_domain.py`, `time_domain.py` | SpecAugment (frequency/time masking), additive noise, reverb, speed perturbation |
| 4 | Tokenization & Batching | `speechbrain/tokenizers/SentencePiece.py` | BPE tokenization (1000/5000 vocab), dynamic batching with bucket sampling |
| 5 | Encoder Forward Pass | `speechbrain/lobes/models/` (CRDNN, Conformer, Branchformer) | Acoustic model encoding: waveform → hidden representations |
| 6 | Beam Search + LM Rescoring | `speechbrain/decoders/seq2seq.py` (77KB), `scorer.py` (71KB) | Beam search (beam=80) with RNNLM/n-gram shallow fusion, coverage penalty, attention shift constraints |
| 7 | Decoder Output | `speechbrain/decoders/ctc.py`, `transducer.py` | CTC/Seq2Seq/Transducer decoding → token sequence → text |
| 8 | Evaluation & Metrics | `speechbrain/dataio/wer.py`, `speechbrain/utils/edit_distance.py` | WER/CER/SER computation via Levenshtein edit distance, Kaldi-compatible output |

### S6.2 DKS Scoring (Total: 37)

| Component | Score | Evidence |
|-----------|-------|----------|
| D1: Domain lexicon | 5 | SentencePiece BPE tokenizer with configurable vocabulary (1000/5000 subword units). Grapheme-to-Phoneme (G2P) conversion recipes. `speechbrain/tokenizers/SentencePiece.py` (23KB), `recipes/LibriSpeech/G2P/`. |
| D2: Ontology/KG | 5 | Dual-model architecture: Acoustic Model (encoder) + Language Model (RNNLM, n-gram ARPA). No explicit KG, but the AM+LM separation constitutes a structured domain decomposition. `speechbrain/lm/ngram.py`, `lm/arpa.py` (11KB). |
| D3: Cross-ontology mapping | 5 | Multi-task learning (ASR+TTS+Speaker in unified framework). Hybrid CTC+attention training with dual-loss computation. Cross-modal: audio features ↔ text tokens ↔ phoneme representations. `recipes/Voicebank/MTL/`. |
| D4: Schema/DDL | 5 | HyperPyYAML configuration schema: 240+ line YAML files with variable references (`!ref`), module instantiation (`!new:`), and function application (`!apply:`). Per-recipe schema defines full pipeline topology. `recipes/LibriSpeech/ASR/seq2seq/hparams/train_BPE_1000.yaml`. |
| D5: Domain profile count | 6 | **43 dataset recipes, 110+ YAML hyperparameter configs, 200+ training scripts**. Dataset categories: ASR (9), TTS (2), Speaker Recognition, Speech Enhancement/Separation, Emotion, SLU, Language ID. `recipes/` directory. |
| D6: Code-level rules | 6 | Beam search decoding rules: beam_size=80, coverage_penalty=1.5, max_attn_shift=240, temperature=1.25, length ratio constraints (min/max_decode_ratio). CTC prefix scoring with blank handling. `speechbrain/decoders/seq2seq.py` (1300+ lines), `decoders/ctc.py` (68KB). |
| D7: Multi-language | 5 | 10+ directly supported languages in CommonVoice recipes (en, fr, de, it, es, pt, rw, zh-CN, ar). VoxLingua107 for 107-language identification. Language-specific tokenizer/G2P adaptation. `recipes/CommonVoice/ASR/seq2seq/hparams/train_{lang}.yaml`. |
| D8: Regulatory/compliance | 0 | No explicit PII redaction or compliance features. Apache 2.0 license. Reproducibility via seeding and checkpoint tracking only. |

**Conservative scoring note**: SpeechBrain's DKS=37 reflects the rich domain knowledge embedded in speech processing: acoustic feature pipelines (D4), extensive recipe configurations (D5), sophisticated decoding constraints (D6), and broad multilingual support (D7). Unlike LLM-based systems where domain knowledge is often implicit in pre-trained weights, SpeechBrain's domain knowledge is explicit in code—feature extraction parameters, beam search rules, and language model integration are all manually specified artifacts.

### S6.3 RSS Scoring (Total: 4)

| Component | Score | Evidence |
|-----------|-------|----------|
| R1: Intent classification | 0 | No intent classification. Fixed ASR/TTS pipeline per recipe. |
| R2: Vector search sophistication | 0 | No vector search. Audio feature space is processed sequentially, not retrieved. |
| R3: Permission-based filtering | 0 | No access control. |
| R4: Multi-index fusion | 2 | Hybrid CTC+attention scoring: dual-path decoding fuses CTC alignment scores with attention-based sequence scores. AM+LM shallow fusion combines acoustic and language model probabilities. `speechbrain/decoders/scorer.py`. |
| R5: Adaptive routing | 2 | 8 attention mechanism variants (ContentBased, LocationAware, MultiHead, RelPosMHAXL, RoPEMHA, etc.) dynamically route decoder focus. Beam search prunes hypothesis space adaptively. `speechbrain/nnet/attention.py` (2400 lines). |

### S6.4 DKAP Layer Mapping

| Layer | Implementation | Evidence |
|-------|---------------|----------|
| L1 | Audio resampling (16kHz) → STFT (n_fft=400, hop=160) → Mel Filterbank (40 bins) → log magnitude → SpecAugment. InputNormalization (global/utterance-level). | `speechbrain/processing/features.py` (62KB), `augment/` |
| L2 | BPE tokenizer (1000/5000 vocab) + G2P. HyperPyYAML schema (240+ lines). 43 dataset recipes × 110+ configs. Language-specific adaptation (10+ languages). AM+LM dual-model architecture. | `speechbrain/tokenizers/`, `recipes/`, `lm/` |
| L3 | Beam search (beam=80) with LM shallow fusion. CTC prefix scoring for alignment. 8 attention variants for decoder routing. Coverage penalty and attention shift constraints. | `speechbrain/decoders/seq2seq.py`, `decoders/ctc.py`, `nnet/attention.py` |
| L4 | 40+ model architectures: Conformer, CRDNN, Branchformer, ECAPA-TDNN, wav2vec2, HuBERT, WavLM, ConvTasNet, HiFi-GAN, FastSpeech2, Tacotron2. Transfer learning from SSL models. | `speechbrain/lobes/models/`, `integrations/huggingface/` |
| L5 | WER/CER/SER via Levenshtein edit distance. Kaldi-compatible output format. Per-utterance and corpus-level aggregation. MetricStats framework. SI-SNR for enhancement tasks. | `speechbrain/dataio/wer.py`, `utils/edit_distance.py`, `utils/metric_stats.py` |

### S6.5 SpeechBrain as Non-LLM DKAP Validation

SpeechBrain's inclusion addresses a critical methodological concern: whether DKAP is merely an artifact of LLM-based system design. Key observations:

| DKAP Property | LLM-Based Systems (MedRAG, FinSQL) | SpeechBrain (Non-LLM) | Implication |
|--------------|-------------------------------------|----------------------|-------------|
| L2 artifacts | Prompt templates, schema DDL, glossaries | Feature extraction params, BPE vocab, beam search rules, YAML configs | L2 takes different forms but serves the same function |
| L3 mechanism | Vector search + reranking | Beam search + LM rescoring + attention routing | Retrieval/routing is paradigm-agnostic |
| L5 validation | SQL execution accuracy, JSON parsing | WER/CER edit distance, Kaldi metrics | Domain-appropriate validation in all cases |
| DKS driver | Schema depth, lexicon size | Recipe complexity, decoding rules, language coverage | Domain knowledge density drives DKS regardless of AI engine |

---

## S7. GraphRAG (Microsoft, Open-Source)

**Repository**: https://github.com/microsoft/graphrag
**Paper**: Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization," arXiv:2404.16130, 2024.

### S7.1 Pipeline Stages (10 stages)

| Stage | Component | Evidence File | Description |
|-------|-----------|---------------|-------------|
| 1 | Input Document Loading | `index/workflows/factory.py` | Ingest raw text documents |
| 2 | Text Unit Creation | `index/workflows/factory.py` | Token/sentence-based chunking |
| 3 | Document Finalization | `index/workflows/factory.py` | Finalize document metadata |
| 4 | Graph Extraction | `index/workflows/factory.py` | LLM-based entity + relationship extraction into KG |
| 5 | Graph Finalization | `index/workflows/factory.py` | Description summarization, deduplication, edge weight computation |
| 6 | Covariate Extraction | `prompts/index/extract_claims.py` | LLM-based claim extraction (subject, object, type, status, dates) |
| 7 | Community Detection | `graphs/hierarchical_leiden.py` | Hierarchical Leiden community detection |
| 8 | Text Unit Linking | `index/workflows/factory.py` | Link text units to entities/relationships/communities |
| 9 | Community Reports | `index/workflows/factory.py` | LLM-generated summary reports per community |
| 10 | Text Embeddings | `index/workflows/factory.py` | Embed entity descriptions, community reports for vector search |

### S7.2 DKS Scoring (Total: 57)

| Component | Score | Evidence |
|-----------|-------|----------|
| D1: Domain lexicon | 10 | Graph-specific vocabulary: entity, relationship, community, covariate, claim, text_unit, edge_weight, node_degree, node_frequency. Default entity types: organization, person, geo, event. 50+ domain field constants in `data_model/schemas.py`. |
| D2: Ontology/KG | 15 | Core purpose is KG construction: Entity-Relationship graph with typed entities, weighted edges, descriptions. Hierarchical Leiden community structure. Community reports as semantic summaries. Covariate/claims layer with structured assertions. |
| D3: Cross-ontology mapping | 3 | Entities linked to text_units, communities, and relationships via cross-referencing IDs. No external ontology mapping. |
| D4: Schema/DDL | 12 | 7 data model classes (Entity, Relationship, Community, CommunityReport, Covariate, TextUnit, Document). Rich config hierarchy: GraphRagConfig with 20+ nested config classes. Parquet-based schemas. Vector store index definitions. |
| D5: Domain profile count | 6 | 4 search profiles (Local, Global, DRIFT, Basic). 2 indexing methods (Standard LLM-based, Fast NLP-based). Update variants for incremental indexing. |
| D6: Code-level rules | 7 | Entity extraction gleaning loop with continue/stop logic. Relationship filtering: in-network priority, budget-based truncation. Community report grounding rules with impact severity rating. Dynamic community selection thresholds. Map-reduce scoring. Graph pruning rules. |
| D7: Multi-language | 3 | Language detection prompt for auto-detecting input language. Graph extraction templates include `{language}` parameter for LLM-mediated translation. No systematic multilingual dictionary. |
| D8: Regulatory/compliance | 1 | Community report prompt mentions "legal compliance" as content category. Claim extraction has status field (TRUE/FALSE/SUSPECTED) with date ranges. |

### S7.3 RSS Scoring (Total: 8)

| Component | Score | Evidence |
|-----------|-------|----------|
| R1: Intent classification | 2 | 4 distinct search modes (local, global, DRIFT, basic). Dynamic community selection rates query-to-community relevancy via LLM. |
| R2: Vector search sophistication | 3 | Semantic similarity search with configurable k and oversample_scaler. Rich filter expression system (AND/OR/NOT, 12 operators). Multiple vector store backends (LanceDB, Azure AI Search, CosmosDB). |
| R3: Permission-based filtering | 0 | No RBAC or tenant isolation. |
| R4: Multi-index fusion | 2 | LocalSearch combines 4 data sources (entities, relationships, community reports, text units) with token-budget proportional allocation. GlobalSearch map-reduce fuses all community report batches. |
| R5: Adaptive routing | 1 | DRIFT search adapts via iterative follow-up queries with depth-limited exploration. Dynamic community selection traverses hierarchy based on relevancy thresholds. |

### S7.4 DKAP Layer Mapping

| Layer | Implementation | Evidence |
|-------|---------------|----------|
| L1 | Document loading, text chunking (token/sentence chunkers) | `load_input_documents`, `create_base_text_units` workflows |
| L2 | Entity-relationship KG, hierarchical community structure, community reports, covariate/claims, configurable entity types | `extract_graph`, `create_communities`, `create_community_reports` |
| L3 | 4 search modes (Local/Global/DRIFT/Basic); vector similarity search; dynamic community selection; map-reduce aggregation | `query/structured_search/` subpackages |
| L4 | LLM completion for extraction, summarization, reports, answering; LLM embedding for vector search | `graphrag_llm` package |
| L5 | Gleaning loops for entity extraction, community report grounding rules, impact severity rating | `graph_extractor.py`, community report prompts |

---

## S8. AudioCraft (Meta, Open-Source)

**Repository**: https://github.com/facebookresearch/audiocraft
**Paper**: Copet et al., "Simple and Controllable Music Generation," NeurIPS 2023.

### S8.1 Pipeline Stages (7 stages)

| Stage | Component | Evidence File | Description |
|-------|-----------|---------------|-------------|
| 1 | Input Ingestion | `audiocraft/data/audio_dataset.py`, `music_dataset.py` | Audio/text input loading and preprocessing |
| 2 | Metadata Extraction | `music_dataset.py` lines 37-50, `jasco_dataset.py` | Extract key, BPM, genre, moods, chords, melody, drums |
| 3 | Condition Embedding | `conditioners.py` lines 345-1003 | T5 text, Chroma, CLAP, chord, melody, drum, style embeddings |
| 4 | Condition Fusion | `conditioners.py` lines 1672-1763 | ConditionFuser merges via sum/prepend/cross-attention/input-interpolate |
| 5 | Audio Tokenization | `models/encodec.py`, `quantization/` | EnCodec encoder compresses audio to discrete tokens via RVQ |
| 6 | Language Model Generation | `models/lm.py` lines 120-250 | Transformer LM generates token sequences with CFG guidance |
| 7 | Audio Decoding | `models/encodec.py` | EnCodec decoder reconstructs waveform from tokens |

### S8.2 DKS Scoring (Total: 42)

| Component | Score | Evidence |
|-----------|-------|----------|
| D1: Domain lexicon | 11 | 80+ audio/music domain terms: key, bpm, genre, moods, chroma, n_chroma, sample_rate, nfft, spectrogram, hop_length. Chord vocabulary (194 Chordino chords). Melody salience (53-dim). Codebook pattern types (Delayed, MusicLM, CoarseFirst, Parallel, Unrolled). |
| D2: Ontology/KG | 4 | `chord_to_index_mapping.pkl` (194 chords). `JascoCondConst` taxonomy. MusicInfo/JascoInfo dataclass hierarchy. No formal KG. |
| D3: Cross-ontology mapping | 2 | `ConditioningAttributes` unifies text, wav, joint_embed, symbolic modalities. `to_condition_attributes()` maps domain dataclasses to unified schema. |
| D4: Schema/DDL | 9 | 68 YAML configs (Hydra-based hierarchy). Conditioner configs, LM scale variants, solver configs. MusicInfo (12 fields), JascoInfo dataclasses. |
| D5: Domain profile count | 7 | MusicGen (4 scales), AudioGen, JASCO (2 scales), MAGNeT, EnCodec (3 variants), multi-band diffusion. 10 conditioner profiles. |
| D6: Code-level rules | 6 | Musical key validation, BPM normalization, metadata-based text augmentation, ChromaStemConditioner (filter drums/bass before chroma), codebook pattern validation, CFG dropout rules. |
| D7: Multi-language | 1 | English only (spacy `en_core_web_sm`). |
| D8: Regulatory/compliance | 2 | Audio watermarking module with WMDetectionLoss. LICENSE_weights for model distribution compliance. |

### S8.3 RSS Scoring (Total: 4)

| Component | Score | Evidence |
|-----------|-------|----------|
| R1: Intent classification | 1 | ConditioningAttributes differentiates condition types. ConditionFuser maps conditions to fusion methods. |
| R2: Vector search | 1 | EmbeddingCache for pre-computed embeddings. CLAP embedding with sliding window. No vector database. |
| R3: Permission filtering | 0 | No access control. |
| R4: Multi-index fusion | 1 | ConditionFuser merges multiple modalities (text, chroma, chords, melody, drums, style). |
| R5: Adaptive routing | 1 | ClassifierFreeGuidanceDropout, multi-source CFG in JASCO. |

### S8.4 DKAP Layer Mapping

| Layer | Implementation | Evidence |
|-------|---------------|----------|
| L1 | Audio waveforms + text descriptions + symbolic conditions (chords, melody) | `audio_dataset.py`, `music_dataset.py`, `jasco_dataset.py` |
| L2 | Music metadata schema (key, BPM, genre, moods), chord vocabulary (194), melody salience (53-dim), chroma features (12-bin), audio quality metrics | `MusicInfo`, `JascoInfo`, `chord_to_index_mapping.pkl` |
| L3 | Condition embedding + fusion: T5, CLAP, ChromaStem, Style, Chords, Melody, Drums conditioners; ConditionFuser; EmbeddingCache | `conditioners.py`, `jasco_conditioners.py` |
| L4 | Transformer LM (up to 3.3B), Flow Matching (JASCO), EnCodec with RVQ, multi-band diffusion, MAGNeT | `lm.py`, `flow_matching.py`, `encodec.py` |
| L5 | Audio quality metrics: FAD, ViSQOL, PESQ, CLAP consistency, chroma cosine similarity, KLD, MIoU. Watermark detection. | `metrics/`, `losses/wmloss.py` |

---

## S9. EmotiVoice (NetEase Youdao, Open-Source)

**Repository**: https://github.com/netease-youdao/EmotiVoice
**Description**: Multi-voice, emotion-controllable text-to-speech system.

### S9.1 Pipeline Stages (8 stages)

| Stage | Component | Evidence File | Description |
|-------|-----------|---------------|-------------|
| 1 | Text Input & Language Detection | `frontend.py:23-27` | Regex splits Chinese vs English text segments |
| 2 | Text Normalization | `frontend_cn.py:86-100`, `text/numbers.py` | cn2an number-to-Chinese, English number/currency expansion, abbreviation expansion |
| 3 | Grapheme-to-Phoneme | `frontend_cn.py:102-121`, `frontend_en.py:38-78` | Chinese pinyin via pypinyin + English via CMUDict + g2p_en |
| 4 | Phoneme-to-Token Encoding | `openaiapi.py:118` | Maps phoneme strings to integer IDs (502 tokens) |
| 5 | Style/Emotion Embedding | `models/prompt_tts_modified/simbert.py:33-72` | BERT classifies emotion, pitch, energy, speed from prompt |
| 6 | Acoustic Model (PromptTTS) | `models/prompt_tts_modified/model_open_source.py` | Encoder-decoder with duration/pitch/energy predictors, Gaussian upsampling |
| 7 | Vocoder (HiFi-GAN) | `models/prompt_tts_modified/jets.py:26-71` | Mel-spectrogram to waveform conversion |
| 8 | Audio Post-processing | `openaiapi.py:162-184` | Speed stretch, format conversion (WAV/MP3), API response |

### S9.2 DKS Scoring (Total: 47)

| Component | Score | Evidence |
|-----------|-------|----------|
| D1: Domain lexicon | 10 | 502 phoneme tokens, 84 ARPAbet symbols (`text/cmudict.py`), 206K-entry LibriSpeech pronunciation lexicon, Chinese pinyin decomposition rules (~30 rules), 7 emotion labels, 3 each pitch/energy/speed labels. |
| D2: Ontology/KG | 4 | Structured prosodic taxonomy: emotion (7) × pitch (3) × energy (3) × speed (3). Speaker identity taxonomy: 2,013 speakers. Chinese number ontology (`cn2an/conf.py`). Flat label files, not KG. |
| D3: Cross-ontology mapping | 3 | Chinese-English phoneme bridging via unified phoneme space with cross-language spacing tokens. Style prompt to prosodic features mapping via BERT. |
| D4: Schema/DDL | 8 | Model config YAML (105 lines). Python config classes with sections for text, speaker, emotion, audio, STFT, mel, training. Pipe-delimited data format schema. |
| D5: Domain profile count | 7 | 2,013 speaker profiles, 7 emotion profiles, 3 pitch/energy/speed profiles. 189 theoretical style combinations. |
| D6: Code-level rules | 7 | ~30 pinyin splitting rules for initial/final decomposition, Chinese/English mixed text handling, punctuation detection rules, number normalization, abbreviation expansion, mel trimming, duration clamping. |
| D7: Multi-language | 7 | Full Chinese-English bilingual TTS. Chinese number words mapping (simplified + traditional). cc_cedict integration. Chinese BERT (SimBERT) for style encoding. |
| D8: Regulatory/compliance | 1 | User agreement PDF referenced in demo. Apache 2.0 license. |

### S9.3 RSS Scoring (Total: 3)

| Component | Score | Evidence |
|-----------|-------|----------|
| R1: Intent classification | 1 | Style encoder classifies emotion intent from prompt text. |
| R2: Vector search | 1 | BERT-based style embeddings computed and cached. No vector database. |
| R3: Permission filtering | 0 | No access control. |
| R4: Multi-index fusion | 0 | No multi-source retrieval. |
| R5: Adaptive routing | 1 | Language-adaptive routing: Chinese vs English G2P pipeline selection. |

### S9.4 DKAP Layer Mapping

| Layer | Implementation | Evidence |
|-------|---------------|----------|
| L1 | Text input with speaker ID, emotion prompt, content text. API endpoint. | `openaiapi.py`, `demo_page.py` |
| L2 | Phoneme lexicons (CMUDict, LibriSpeech 206K), prosodic knowledge (7 emotions, pitch/energy/speed levels), 2,013 speaker profiles, Chinese numeral system | `text/cmudict.py`, `data/youdao/text/`, `cn2an/conf.py` |
| L3 | Language detection + routing (Chinese vs English G2P). Style embedding retrieval via BERT. Speaker/token lookup. | `frontend.py`, `simbert.py` |
| L4 | SimBERT style encoder, PromptTTS acoustic model, HiFi-GAN vocoder, JETS generator | `simbert.py`, `model_open_source.py`, `jets.py` |
| L5 | Audio normalization, format conversion, speed adjustment. Training-time TTSLoss (mel, prosody, forward-sum, bin losses). | `openaiapi.py`, `loss.py` |

---

## S10. LLaVA (UWisconsin, Open-Source)

**Repository**: https://github.com/haotian-liu/LLaVA
**Paper**: Liu et al., "Visual Instruction Tuning," NeurIPS 2023.

### S10.1 Pipeline Stages (6 stages)

| Stage | Component | Evidence File | Description |
|-------|-----------|---------------|-------------|
| 1 | Image Preprocessing | `llava/mm_utils.py:12-96` | Image loading, resolution selection, padding/resizing, patch division |
| 2 | Visual Encoding | `llava/model/multimodal_encoder/clip_encoder.py:35-57` | CLIP ViT encodes patches into visual features with layer selection |
| 3 | Multi-modal Projection | `llava/model/multimodal_projector/builder.py:33-51` | MLP projects vision features into LLM embedding space |
| 4 | Conversation Template Formatting | `llava/conversation.py:32-107` | Prompt construction using 13 conversation templates |
| 5 | Multi-modal Fusion & LLM Forward | `llava/model/llava_arch.py:145-324` | Interleave image embeddings with text tokens, feed to LLM |
| 6 | Text Generation | `llava/eval/run_llava.py:114-127` | Autoregressive generation with stopping criteria |

### S10.2 DKS Scoring (Total: 12)

| Component | Score | Evidence |
|-----------|-------|----------|
| D1: Domain lexicon | 3 | ~15-20 vision-language tokens: IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, mm_projector, vision_tower, image_aspect_ratio, anyres. Small, focused vocabulary. |
| D2: Ontology/KG | 0 | No knowledge graphs or ontologies. Relies on pre-trained CLIP + LLM knowledge. |
| D3: Cross-ontology mapping | 0 | Vision-language alignment via learned MLP projection, not explicit mapping. |
| D4: Schema/DDL | 4 | Conversation dataclass with system, roles, messages, sep_style, version. SeparatorStyle enum (5 styles). ModelArguments/DataArguments/TrainingArguments (30+ fields). DeepSpeed configs. |
| D5: Domain profile count | 3 | 13 conversation template profiles. 4 projector types. 3 image aspect ratio modes. Multiple training profiles. |
| D6: Code-level rules | 2 | Model name routing rules, conversation mode auto-inference, preprocess dispatch, image processing mode rules. Structural routing, not deep domain rules. |
| D7: Multi-language | 0 | English only. |
| D8: Regulatory/compliance | 0 | Llama-2 system prompt has safety text but not structured compliance. |

**Scoring rationale**: LLaVA is an AI-engine-centric system (L4 dominant) with minimal explicit domain knowledge. Domain knowledge resides in pre-trained weights (CLIP + LLaMA), not in structured code artifacts. This is characteristic of end-to-end neural approaches.

### S10.3 RSS Scoring (Total: 2)

| Component | Score | Evidence |
|-----------|-------|----------|
| R1: Intent classification | 1 | Basic model name pattern matching for conversation mode. Worker dispatch (LOTTERY/SHORTEST_QUEUE). |
| R2: Vector search | 0 | No vector search or RAG. Direct model inference. |
| R3: Permission filtering | 0 | No access control. |
| R4: Multi-index fusion | 0 | No index fusion. |
| R5: Adaptive routing | 1 | Worker dispatch routing. Model-variant selection based on model name. |

### S10.4 DKAP Layer Mapping

| Layer | Implementation | Evidence |
|-------|---------------|----------|
| L1 | Image loading/preprocessing, text input parsing, conversation template formatting | `llava/mm_utils.py`, `llava/conversation.py` |
| L2 | Conversation templates (13 profiles), special tokens vocabulary, image processing modes, model config parameters. **Minimal explicit domain knowledge.** | `llava/constants.py`, `llava/conversation.py` |
| L3 | Model variant routing by name pattern. Worker dispatch. No vector retrieval. | `llava/model/builder.py`, `llava/serve/controller.py` |
| L4 | CLIP Vision Tower + MLP Projector + LLaMA/MPT/Mistral LMs. LoRA/QLoRA fine-tuning. | `llava/model/` |
| L5 | Keywords stopping criteria, evaluation scripts (ScienceQA, TextVQA, POPE, MMBench), regex answer parsing. | `llava/eval/` |

---

## S11. Logic-LLM (Yale, Open-Source)

**Repository**: https://github.com/teacherpeterpan/Logic-LLM
**Paper**: Pan et al., "Logic-LLM: Integrating Formal Logic Reasoning with LLMs," ACL 2024.

### S11.1 Pipeline Stages (6 stages)

| Stage | Component | Evidence File | Description |
|-------|-----------|---------------|-------------|
| 1 | Input Parsing | `logic_program.py:69-72` | Load dataset JSON, extract context/question/options |
| 2 | Prompt Construction | `logic_program.py:21-67` | Dataset-specific prompt template selection (5 datasets) |
| 3 | LLM Logic Program Generation | `logic_program.py:74-101` | GPT-3/4 translates NL to formal logic programs |
| 4 | Logic Program Parsing | `prover9_solver.py`, `pyke_solver.py`, `csp_solver.py`, `sat_problem_solver.py` | Parse LLM output into solver-specific structures |
| 5 | Symbolic Execution | `logic_inference.py:43-57` | Execute via Prover9/Pyke/CSP/Z3 solvers |
| 6 | Answer Mapping + Backup | `logic_inference.py:43-57`, `backup_answer_generation.py` | Map solver output to choices; fallback to random/LLM |

### S11.2 DKS Scoring (Total: 41)

| Component | Score | Evidence |
|-----------|-------|----------|
| D1: Domain lexicon | 8 | ~60-70 formal logic terms: XOR, OR, AND, IMPLIES, IFF, FORALL, EXISTS, NOT (+ Unicode ⊕∨∧→↔∀∃¬). Prover9 operators, Z3 vocabulary (EnumSort, IntSort, ForAll, Distinct, Count), CSP vocabulary, Pyke vocabulary. |
| D2: Ontology/KG | 7 | FOL Context-Free Grammar (`fol_parser.py:11-20`). Symbol resolution taxonomy (VAR, PRED, CONST). Dataset-to-solver type mapping (5 problem types → 4 solver paradigms). Logic program structure schemas per solver. |
| D3: Cross-ontology mapping | 5 | NL-to-FOL mapping (all prompts). FOL-to-Prover9 parser (Unicode to Prover9 syntax). Logic program-to-Z3 Python translator. Logic program-to-Pyke translator. Answer mapping across solver representations. |
| D4: Schema/DDL | 6 | FOL CFG schema (8 production rules). Z3 declaration schema (Declarations/Constraints/Options). CSP schema (Domain/Variables/Constraints/Query). Pyke schema (Query/Rules/Facts/Predicates). |
| D5: Domain profile count | 7 | 5 dataset profiles (FOLIO→Prover9, ProntoQA→Pyke, ProofWriter→Pyke, LogicalDeduction→CSP, AR-LSAT→Z3). 2 self-correction profiles. Backup strategy profiles. |
| D6: Code-level rules | 7 | FOL grammar production rules. 13 yacc grammar rules for FOL-to-Prover9 (XOR expansion, implication). CSP constraint parsing with regex. Z3 code translation rules (Count, Distinct, quantifiers). Three-valued logic (run prover on goal + negated goal). Program validation/repair heuristics. |
| D7: Multi-language | 1 | Unicode logic symbols alongside ASCII equivalents. No natural language multilingual support. |
| D8: Regulatory/compliance | 0 | No regulatory encoding. |

**Scoring rationale**: Logic-LLM scores highly on DKS despite being a relatively small codebase because formal logic domain knowledge is deeply encoded: the FOL grammar, four distinct solver DSLs, cross-representation translation layers, and dataset-specific formalization profiles represent rich, structured domain artifacts.

### S11.3 RSS Scoring (Total: 3)

| Component | Score | Evidence |
|-----------|-------|----------|
| R1: Intent classification | 2 | Dataset-name-based routing to prompt creators and solver executors. Model-aware prompt selection (AR-LSAT vs AR-LSAT-long). |
| R2: Vector search | 0 | No vector search. Template-based prompting. |
| R3: Permission filtering | 0 | No access control. |
| R4: Multi-index fusion | 0 | Single solver per dataset. |
| R5: Adaptive routing | 1 | Error-based adaptive fallback (parsing/execution failure → backup). Self-refinement loop with iterative correction rounds. |

### S11.4 DKAP Layer Mapping

| Layer | Implementation | Evidence |
|-------|---------------|----------|
| L1 | JSON dataset loading with context, question, options, answer fields | `logic_program.py:69-72` |
| L2 | FOL grammar, logic formalization rules in prompts, 5 dataset-specific schemas, predicate/fact/rule/constraint structures, 4 solver-specific DSLs | `fol_parser.py`, `models/prompts/`, solver parsers |
| L3 | Dataset-name-based routing to prompt creators and solver executors | `logic_program.py:21-25`, `logic_inference.py:22-27` |
| L4 | OpenAI GPT-3/4 for NL-to-logic translation; Prover9, Pyke, Z3, CSP symbolic solvers | `utils.py`, solver files |
| L5 | Answer mapping, exact-match evaluation, self-refinement error correction loop | `evaluation.py`, `self_refinement.py` |

---

## S12. CrewAI (Open-Source Framework)

**Repository**: https://github.com/crewAIInc/crewAI
**Description**: Multi-agent AI orchestration framework.

**Important methodological note**: Like LlamaIndex, CrewAI is a *framework/platform*, not a domain-specific application. DKS scoring reflects the framework's built-in domain knowledge artifacts, which are intentionally minimal because domain specificity is deferred to user configuration at runtime.

### S12.1 Pipeline Stages (7 stages)

| Stage | Component | Evidence File | Description |
|-------|-----------|---------------|-------------|
| 1 | Input Reception | `crew.py:668-713` | Receive user inputs, interpolate placeholders into agent/task descriptions |
| 2 | Planning | `planning_handler.py:57-78` | LLM-generated step-by-step execution plans (optional) |
| 3 | Knowledge Query | `crew.py:1611-1619`, `knowledge_storage.py:55-85` | Vector search against knowledge sources |
| 4 | Agent Execution | `crew_agent_executor.py:85+`, `step_executor.py:60+` | LLM calls, tool execution, action parsing |
| 5 | Memory Save/Recall | `encoding_flow.py`, `recall_flow.py`, `unified_memory.py` | Unified Memory with 5-step batch encoding + adaptive-depth retrieval |
| 6 | Guardrail Validation | `guardrail.py`, `hallucination_guardrail.py` | Task guardrails + hallucination guardrail validate output |
| 7 | Output Aggregation | `crew.py:1510-1543` | Assemble final CrewOutput from task outputs |

### S12.2 DKS Scoring (Total: 15)

| Component | Score | Evidence |
|-----------|-------|----------|
| D1: Domain lexicon | 2 | Generic agent-orchestration terms (role, goal, backstory, task, crew, delegation, kickoff). No domain-specific glossary. |
| D2: Ontology/KG | 1 | Memory scope hierarchy (/company/team/user), category-based organization. Lightweight taxonomy, not KG. |
| D3: Cross-ontology mapping | 0 | No mapping between knowledge schemas. |
| D4: Schema/DDL | 5 | Rich Pydantic schemas: MemoryRecord (11 fields), MemoryConfig (13 params), Crew (30+ fields), Task (extensive config), EncodingState/RecallState flow schemas. |
| D5: Domain profile count | 2 | 2 process profiles (sequential, hierarchical). 18 embedder provider profiles. Infrastructure profiles, not domain-specific. |
| D6: Code-level rules | 3 | Task validation (async cannot be conditional, no future context references). Memory composite scoring formula. RecallFlow confidence thresholds. Orchestration rules, not domain business rules. |
| D7: Multi-language | 1 | I18N infrastructure (`utilities/i18n.py`) with translation support, but only English implemented. |
| D8: Regulatory/compliance | 1 | SecurityConfig with fingerprinting (TODO). Memory private field for privacy filtering. |

**Scoring rationale**: CrewAI's low DKS (15) is expected for a generic framework — domain knowledge is deferred to runtime. The high RSS (9) reflects sophisticated retrieval infrastructure.

### S12.3 RSS Scoring (Total: 9)

| Component | Score | Evidence |
|-----------|-------|----------|
| R1: Intent classification | 2 | RecallFlow uses LLM-based query analysis: complexity classification (simple/complex), sub-query distillation, temporal hint extraction. |
| R2: Vector search sophistication | 3 | Full pipeline: 18 embedding providers (OpenAI, Cohere, Jina, HuggingFace), 3 vector store backends (ChromaDB, Qdrant, LanceDB), composite scoring (semantic + recency + importance), score thresholds, oversampling, batch dedup. |
| R3: Permission filtering | 1 | Memory records have private boolean and source field. RecallState includes source and include_private filters. Basic private/public filtering. |
| R4: Multi-index fusion | 1 | Knowledge (ChromaDB/Qdrant) and Memory (LanceDB) are separate retrieval systems contributing to context. RecallFlow searches across multiple scopes in parallel. No explicit rank fusion algorithm. |
| R5: Adaptive routing | 2 | RecallFlow: confidence_threshold_high/low, complex_query_threshold, exploration_budget for LLM-driven deeper exploration. Router decorator pattern in Flow framework. |

### S12.4 DKAP Layer Mapping

| Layer | Implementation | Evidence |
|-------|---------------|----------|
| L1 | `Crew.kickoff(inputs=...)`, FlowConfig, InputProvider | `crew.py:668-713`, `flow/input_provider.py` |
| L2 | Knowledge (vector store), KnowledgeStorage, knowledge sources (CSV, PDF, JSON, Excel, text), MemoryRecord with scope/categories | `knowledge/`, `memory/types.py` |
| L3 | RecallFlow (adaptive-depth retrieval), KnowledgeStorage.search(), RAG factory (ChromaDB/Qdrant/LanceDB), query analysis | `memory/recall_flow.py`, `rag/factory.py` |
| L4 | CrewAgentExecutor, StepExecutor, LLM (multi-provider), CrewPlanner, agent delegation | `agents/`, `llm.py` |
| L5 | GuardrailResult, HallucinationGuardrail, TaskEvaluator, ConditionalTask condition checks | `utilities/guardrail.py`, `tasks/` |

### S12.5 CrewAI vs. LlamaIndex: Framework DKS Comparison

Both CrewAI and LlamaIndex are frameworks, but they exhibit different DKS/RSS profiles:

| Dimension | CrewAI (DKS=15, RSS=9) | LlamaIndex (DKS=20, RSS=7) |
|-----------|------------------------|----------------------------|
| Domain knowledge | Almost entirely deferred to users | Some built-in (EntityExtractor, PresidioPII, PropertyGraph) |
| Retrieval sophistication | Strong: adaptive-depth memory, composite scoring, multi-provider | Strong: fusion retrieval, multi-index, router query engine |
| Architecture focus | Agent orchestration (multi-agent coordination) | Data structuring (ingestion, indexing, retrieval) |
| L2 driver | Memory scope hierarchy, knowledge sources | PropertyGraphIndex, EntityExtractor, SQL schema |

This comparison demonstrates that frameworks can have comparable RSS but diverge in DKS depending on how much built-in domain structuring they provide.

---

## S13. Cross-System Comparison

### S13.1 DKS Summary — All 12 External Systems

| # | System | DKS | RSS | DKS+RSS | Stages | AI Engine | LLM? | Paradigm |
|---|--------|-----|-----|---------|--------|-----------|------|----------|
| 1 | MedRAG | 5 | 5 | 10 | 4 | LLM+RAG | Yes | Healthcare RAG |
| 2 | LLaVA | 12 | 2 | 14 | 6 | VLM | Yes | VLM |
| 3 | LlamaIndex | 20 | 7 | 27 | 6 | LLM RAG | Yes | Document RAG |
| 4 | CrewAI | 15 | 9 | 24 | 7 | LLM Agents | Yes | Agentic AI |
| 5 | BIRD-SQL | 26 | 3 | 29 | 7 | LLM+T5 FT | Yes | Text-to-SQL |
| 6 | Wheatley | 33 | 2 | 35 | 8 | RL+GNN | **No** | RL+Scheduling |
| 7 | FinSQL | 35 | 6 | 41 | 7 | LLM+LoRA | Yes | Domain Fine-Tuning |
| 8 | SpeechBrain | 37 | 4 | 41 | 8 | ASR/TTS | **No** | Speech |
| 9 | Logic-LLM | 41 | 3 | 44 | 6 | LLM+Solvers | Yes | Neuro-Symbolic |
| 10 | AudioCraft | 42 | 4 | 46 | 7 | Transformer LM | **No** | Music Gen |
| 11 | EmotiVoice | 47 | 3 | 50 | 8 | BERT+TTS | **No** | Emotion/Voice |
| 12 | GraphRAG | 57 | 8 | 65 | 10 | LLM+KG | Yes | KG/Ontology |

**Distribution properties (n=12)**:
- DKS+RSS range: 10–65 (55-point spread)
- 4 non-LLM systems: Wheatley (RL+GNN), SpeechBrain (ASR/TTS), AudioCraft (Transformer LM), EmotiVoice (BERT+TTS)
- 10 distinct AI paradigms represented
- **Pearson r = 0.812 (p = 0.001)**: strong linear association between DKS+RSS and pipeline stages
- **Spearman ρ = 0.678 (p = 0.015)**: significant monotonic association at α = 0.05
- **R² = 0.659**: DKS+RSS explains 66% of pipeline stage count variance

### S13.2 Key Observations from n=12 External Expansion

**(1) Framework vs. Application DKS divergence.** LlamaIndex (DKS=20) and CrewAI (DKS=15) are both frameworks with intentionally low built-in domain knowledge. In contrast, domain-specific systems like EmotiVoice (DKS=47) and GraphRAG (DKS=57) encode rich domain artifacts. This validates the DKS design principle: measuring explicit, structured domain artifacts in code.

**(2) Non-LLM systems span the DKS range.** With 4 non-LLM systems (DKS+RSS: 35, 41, 46, 50), the claim that DKAP is AI-engine-agnostic is substantially strengthened. These systems score comparably to LLM-based systems of similar domain complexity.

**(3) Neuro-Symbolic systems show high DKS despite small codebases.** Logic-LLM (DKS=41) demonstrates that formal logic domain knowledge — grammars, solver DSLs, cross-representation translations — yields high DKS even in compact implementations. This supports the interpretation that DKS measures knowledge density, not code volume.

**(4) VLM systems show minimal explicit domain knowledge.** LLaVA (DKS=12) represents the end-to-end neural approach where domain knowledge resides in pre-trained weights rather than structured code artifacts, resulting in the lowest DKS among all 12 external systems.

### S13.3 External vs. Internal System Comparison (n=24)

| System | Origin | DKS+RSS | Stages | Paradigm |
|--------|--------|---------|--------|----------|
| MedRAG | External | 10 | 4 | Healthcare RAG |
| bidwatch | Internal | 10 | 5 | Agentic AI |
| LLaVA | External | 14 | 6 | VLM |
| Emotive-Dub | Internal | 15 | 10 | Emotion AI |
| babel-audio | Internal | 18 | 7 | ASR/TTS |
| CrewAI | External | 24 | 7 | Agentic AI |
| LlamaIndex | External | 27 | 6 | Document RAG |
| BIRD-SQL | External | 29 | 7 | Text-to-SQL |
| Wheatley | External | 35 | 8 | RL+GNN |
| aistream | Internal | 39 | 10 | KG/Ontology |
| FinSQL | External | 41 | 7 | Text-to-SQL |
| SpeechBrain | External | 41 | 8 | Speech |
| nama | Internal | 42 | 8 | Text-to-SQL |
| Logic-LLM | External | 44 | 6 | Neuro-Symbolic |
| AudioCraft | External | 46 | 7 | Music Gen |
| DIRE | Internal | 47 | 12 | Neuro-Symbolic |
| ex-GPT | Internal | 49 | 22 | RAG |
| EmotiVoice | External | 50 | 8 | Emotion/Voice |
| Babel-Beats | Internal | 61 | 11 | Music Gen |
| GraphRAG | External | 65 | 10 | KG/Ontology |
| ShipSched | Internal | 66 | 18 | RL+GNN |
| ClinicalRAG | Internal | 79 | 14 | RAG |
| ai-governance | Internal | 81 | 16 | Text-to-SQL |
| AutoInspect | Internal | 101 | 20 | VLM |

**Combined statistics (n=24)**:
- Pearson r = 0.781 (p < 0.001), Spearman ρ = 0.895 (p < 0.001)
- Excluding outlier ex-GPT (n=23): Pearson r = 0.879 (p < 0.001), R² = 0.773

**Paired Cross-Validation (12 paradigm pairs)**:
- Internal DKS+RSS > External: 8/12 pairs
- External DKS+RSS > Internal: 4/12 pairs
- Wilcoxon signed-rank: W=25.0, p=0.301 (no significant difference)
- Mean difference (Internal − External): +15.2

The Wilcoxon test indicates no statistically significant difference between internal and external DKS+RSS scores when matched by paradigm, supporting the claim that DKAP captures genuine architectural patterns independent of development context.

### S13.4 Cross-Paradigm DKS+RSS Comparison (10 Paradigms)

| Paradigm | External System | DKS+RSS | Internal System | DKS+RSS | Same DKAP? |
|----------|----------------|---------|-----------------|---------|------------|
| Healthcare RAG | MedRAG | 10 | ClinicalRAG | 79 | Yes |
| Document RAG | LlamaIndex | 27 | ex-GPT | 49 | Yes |
| Text-to-SQL (General) | BIRD-SQL | 29 | ai-governance | 81 | Yes |
| Domain Fine-Tuning | FinSQL | 41 | nama | 42 | Yes |
| RL+Scheduling | Wheatley | 35 | ShipSched | 66 | Yes |
| Speech | SpeechBrain | 41 | babel-audio | 18 | Yes |
| Music Gen | AudioCraft | 46 | Babel-Beats | 61 | Yes |
| KG/Ontology | GraphRAG | 65 | aistream | 39 | Yes |
| VLM | LLaVA | 14 | Saeyon | 101 | Yes |
| Emotion/Voice | EmotiVoice | 50 | Emotive-Dub | 15 | Yes |
| Neuro-Symbolic | Logic-LLM | 44 | DIRE | 47 | Yes |
| Agentic AI | CrewAI | 24 | bidwatch | 10 | Yes |

All 12 paradigm pairs exhibit the DKAP five-layer decomposition despite independent development, providing strong evidence that DKAP captures a universal architectural pattern.

---

## S14. Reproducibility

All analyses can be reproduced by cloning the following repositories and applying the four-step Architecture Mining Protocol described in Section 3.B:

```bash
# Original 6 external systems (S1-S6)
git clone https://github.com/Teddy-XiongGZ/MedRAG
git clone https://github.com/bigbigwatermalon/FinSQL
git clone https://github.com/jolibrain/wheatley
git clone https://github.com/AlibabaResearch/DAMO-ConvAI  # BIRD-SQL in bird/ subdirectory
git clone https://github.com/run-llama/llama_index          # LlamaIndex framework
git clone https://github.com/speechbrain/speechbrain        # SpeechBrain toolkit

# Additional 6 external systems (S7-S12)
git clone https://github.com/microsoft/graphrag             # GraphRAG
git clone https://github.com/facebookresearch/audiocraft    # AudioCraft (MusicGen)
git clone https://github.com/netease-youdao/EmotiVoice      # EmotiVoice
git clone https://github.com/haotian-liu/LLaVA              # LLaVA
git clone https://github.com/teacherpeterpan/Logic-LLM      # Logic-LLM
git clone https://github.com/crewAIInc/crewAI               # CrewAI
```

Specific file paths and line numbers referenced in this document are based on repository states as of March 2026. Readers should verify against the latest repository versions, noting that active development may alter file locations.

**Framework version notes**: LlamaIndex and CrewAI are under active development with frequent API changes. DKS scoring references built-in capabilities at the time of analysis. Readers should consult specific commit hashes for exact file path correspondence.
