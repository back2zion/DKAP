#!/usr/bin/env python3
"""
Multi-Agent Inter-Rater Reliability Evaluation for DKS/RSS
==========================================================
5 LLM agent personas independently score 12 external systems using
the refined DKS/RSS rubric, then compute Fleiss' κ, ICC, pairwise
Cohen's κ, and Spearman ρ.

Paper: "DKAP: Domain Knowledge Adaptation Pattern for Cross-Domain
        Industrial AI System Design" (IEEE Access)

Author: Dooil Kwak
Date: 2026-04-03
"""

import json
import os
import sys
import time
import re
import logging
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

try:
    import requests
except ImportError:
    os.system("pip install requests --break-system-packages -q")
    import requests

try:
    import numpy as np
except ImportError:
    os.system("pip install numpy --break-system-packages -q")
    import numpy as np

try:
    from scipy import stats as scipy_stats
except ImportError:
    os.system("pip install scipy --break-system-packages -q")
    from scipy import stats as scipy_stats

# ============================================================
# Configuration
# ============================================================
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "default-model")
MAX_TOKENS = 4096
TEMPERATURE = 0.0
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results/interrater"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("multi_agent_interrater.log"),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
# 1. Refined DKS/RSS Rubric (with clarifications)
# ============================================================
REFINED_RUBRIC = """
## DKS (Domain Knowledge Score) Rubric — Range [0, 100]

### D1. Domain Lexicon Size (max 15)
Definition: Count of **explicitly declared** domain-specific terms in dedicated
glossary files, configuration constants, enum values, or docstring-based
dictionaries. Variable names in code do NOT count unless they are part of a
curated list (e.g., `GLOSSARY = {...}`, `DOMAIN_TERMS = [...]`).
- 0: No explicit domain glossary or term list
- 5: 1–50 curated terms (e.g., small enum or config dict)
- 10: 51–200 curated terms (e.g., taxonomy file, large enum)
- 15: >200 curated terms (e.g., full medical/financial ontology vocabulary)

**Worked example**: FinSQL has 1,170 column names in `tables.json` encoding
financial terminology (mainoperincome, grossprofit, fundreturn) → D1=10.
MedRAG has no glossary file (medical terms live in pre-trained embeddings) → D1=0.
**Clarification**: Typed field constants in data model classes (e.g., Pydantic/dataclass
fields like `entity.title`, `relationship.weight`) count toward D1 if they define
domain entity attributes, even without a separate glossary file.

### D2. Ontology/Knowledge Graph Presence (max 15)
Definition: Explicit **semantic** graph or taxonomy where nodes represent domain
concepts and edges represent domain relationships (is-a, part-of, causes, etc.).
Computational graphs (GNN edges, attention layers) do NOT count unless they
encode domain-semantic relationships. FOL grammars count only if they define a
domain concept hierarchy.
- 0: No ontology or semantic graph
- 5: Flat taxonomy with <50 concept nodes OR non-semantic graph used for domain modeling
- 10: Hierarchical ontology with <200 concept nodes
- 15: Rich graph with ≥200 nodes and ≥0.5×nodes edges, encoding domain semantics

**Worked example**: GraphRAG constructs entity-relation KGs as its core purpose
(entity extraction → community detection → hierarchical summarization) → D2=15.
Wheatley uses 7 edge types (prec/rprec/rc/rp/rrp/pool/self) but these are
computational scheduling constraints, not semantic concepts → D2=5.

### D3. Cross-Ontology Mapping (max 10)
Definition: Explicit mappings between different coding standards, classification
systems, or representation formats for the same domain.
- 0: No cross-standard mapping
- 5: Single standard or single cross-format mapping (e.g., ICD-10 only, or CN↔EN)
- 10: Multi-standard mapping (e.g., ICD-10 ↔ SNOMED ↔ KCD-8)

### D4. Schema/DDL Complexity (max 15)
Definition: Structured data definitions including SQL DDL, JSON Schema, YAML
configs with typed fields, Protocol Buffers, or **Python dataclasses/Pydantic
models** that define domain entities with typed attributes. Pure Python classes
with only methods (no typed fields) do NOT count.
- 0: No structured data schema
- 5: 1–5 entity types with no FK/relationships OR dataclasses with <20 typed fields
- 10: 6–15 entity types with FK/relationships OR dataclasses with 20–100 typed fields
- 15: >15 entity types with constraints/triggers OR >100 typed fields with validation

**Worked example**: GraphRAG has 7 data model classes (Entity, Relationship,
Community, CommunityReport, Document, TextUnit, Covariate) with typed fields
→ D4=10 (not 5, because they encode relational structure; not 15, because no
SQL-level constraints). BIRD-SQL has 95 databases spanning 37 domains → D4=10.

### D5. Domain Profile Count (max 10)
Definition: Number of distinct configuration profiles, templates, or parameter
presets that encode domain-specific behavioral variations.
- 0: No domain profiles
- 3: <10 profiles/templates
- 6: 10–50 profiles
- 10: >50 profiles

### D6. Code-Level Rule Patterns (max 10)
Definition: Explicit domain **business rules** encoded as regex patterns, rule
engines, validation logic, constraint checks, OR LLM-orchestrated procedural rules
(e.g., iterative extraction loops, budget-based truncation, graph pruning rules).
Programming language patterns (SQL normalization, whitespace handling) count only
if they encode domain-specific semantics.
- 0: No domain rules
- 3: <10 domain rule patterns
- 6: 10–30 domain rule patterns
- 10: >30 domain rule patterns

### D7. Multi-Language Term Mapping (max 10)
Definition: Explicit support for domain terms in multiple natural languages,
or systematic code-to-natural-language mappings.
- 0: Single language only
- 5: Bilingual (e.g., EN/CN parallel terms)
- 10: Trilingual+ OR comprehensive code→natural language mapping

### D8. Regulatory/Compliance Encoding (max 15)
Definition: Explicit encoding of legal, regulatory, or compliance requirements
in code (not just a license file).
- 0: No regulatory encoding
- 5: Basic PII handling (e.g., masking, anonymization)
- 10: Sector-specific regulation (e.g., HIPAA, GDPR, financial reporting rules)
- 15: Multi-regulation compliance (e.g., HIPAA + state laws + audit requirements)


## RSS (Retrieval Sophistication Score) Rubric — Range [0, 12]

### R1. Intent Classification (max 3)
- 0: No intent classification
- 1: Keyword-based routing
- 2: ML classifier for query routing
- 3: Multi-stage intent decomposition

### R2. Vector Search Sophistication (max 3)
Definition: Explicit retrieval using embedding-based similarity search.
Neural feature extraction for model inference (e.g., CLIP encoding for VLM,
mel-spectrogram for TTS) does NOT count as "vector search."
- 0: No vector/embedding-based retrieval
- 1: Flat vector search (single embedding space)
- 2: Filtered vector search (metadata + embedding)
- 3: Vector search + reranking pipeline

**Worked example**: MedRAG uses BM25 + Contriever + SPECTER + MedCPT with
RRF fusion + FAISS HNSW indexing → R2=3. AudioCraft uses neural conditioning
(T5 text encoder → cross-attention) which is model inference, not retrieval → R2=0.

### R3. Permission-Based Filtering (max 2)
- 0: No access control
- 1: Role-based filtering
- 2: Document-level + role-based filtering

### R4. Multi-Index Fusion (max 2)
- 0: Single index/retriever
- 1: Dual index/retriever fusion
- 2: Triple+ index/retriever fusion

### R5. Adaptive Routing (max 2)
- 0: Fixed pipeline
- 1: Rule-based routing
- 2: ML/LLM-based dynamic routing
"""

# ============================================================
# 2. System Evidence Summaries (from Supplementary Analysis)
# ============================================================
SYSTEM_EVIDENCE = {
    "MedRAG": {
        "repo": "https://github.com/Teddy-XiongGZ/MedRAG",
        "paper": "Xiong et al., Findings of ACL 2024",
        "domain": "Medical QA (RAG)",
        "evidence": """
- Pipeline: Corpus Loading → Retrieval (BM25/Contriever/SPECTER/MedCPT + RRF) → LLM Inference → Answer Extraction
- No explicit medical glossary file. Medical knowledge in pre-trained embeddings (MedCPT).
- No ontology graph. Corpora are flat document collections (PubMed, StatPearls, Textbooks, Wikipedia).
- No database schema. Document-based retrieval only.
- 4 corpus types with distinct chunking strategies.
- No domain-specific regex/rule patterns.
- English only. No regulatory encoding.
- RRF fusion across 4 retrievers with FAISS HNSW indexing.
- No intent classification. No access control. No adaptive routing.
""",
        "r1_dks": 5, "r1_rss": 5, "r1_stages": 4,
    },
    "LLaVA": {
        "repo": "https://github.com/haotian-liu/LLaVA",
        "paper": "Liu et al., NeurIPS 2023",
        "domain": "Vision-Language Model",
        "evidence": """
- Pipeline: Image encoding (CLIP ViT-L/14) → Projection (MLP) → LLM generation (LLaMA/Vicuna)
- 13 conversation templates in conversation.py for different model variants.
- IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN constants for cross-modal alignment.
- No explicit domain glossary. Visual concepts in CLIP pre-trained embeddings.
- No ontology. No database schema. No domain profiles beyond conversation templates.
- No retrieval pipeline (end-to-end neural generation).
- Multi-modal: image + text, but single language (English).
- No regulatory encoding. No access control.
""",
        "r1_dks": 12, "r1_rss": 2, "r1_stages": 6,
    },
    "CrewAI": {
        "repo": "https://github.com/crewAIInc/crewAI",
        "paper": "N/A (open-source framework)",
        "domain": "Agentic AI Framework",
        "evidence": """
- Pipeline: Agent definition → Task planning → Memory storage → Knowledge retrieval → Tool execution → Output
- Unified memory system: ShortTermMemory + LongTermMemory + EntityMemory + UserMemory with recall_flow.
- Knowledge storage system with EmbedChain integration for document ingestion.
- Guardrail system (utilities/guardrail.py) for output validation.
- Role-based agent definition with goal/backstory/tools configuration.
- Multi-provider vector search (embedchain supports multiple embedding providers).
- Adaptive memory with composite scoring (recency + relevance + importance).
- No explicit domain glossary (domain-agnostic framework).
- No ontology/KG construction. No SQL schema.
- Process types: sequential, hierarchical, consensual.
""",
        "r1_dks": 15, "r1_rss": 9, "r1_stages": 7,
    },
    "LlamaIndex": {
        "repo": "https://github.com/run-llama/llama_index",
        "paper": "N/A (open-source framework)",
        "domain": "RAG Framework",
        "evidence": """
- Framework with modular pipeline: Ingestion → Index Construction → Retrieval → Post-processing → Synthesis → Evaluation
- PropertyGraphIndex with LLM-based triplet extraction (entity/relation KG construction).
- SQLStructStoreIndex for database schema handling.
- EntityExtractor with DEFAULT_ENTITY_MAP (9 entity types via multinerd NER).
- 50+ prompt templates in default_prompts.py.
- QueryFusionRetriever with 3 fusion modes (RRF, relative score, distance-based).
- RouterQueryEngine + LLMSingleSelector for dynamic query routing.
- PresidioPostprocessor for PII detection/masking.
- VectorStoreIndex with 40+ embedding integrations.
- **IMPORTANT scoring note**: LlamaIndex is a framework/toolkit. Score as a TYPICAL
  PRODUCTION DEPLOYMENT, not the framework ceiling. A typical deployment activates
  VectorStoreIndex (not PropertyGraphIndex), uses 3-5 prompt templates (not 50+),
  and does not activate EntityExtractor or PresidioPostprocessor unless specifically
  configured. Score what a representative RAG app built on LlamaIndex would look like,
  not the full API surface.
- Scored as typical deployment, not framework ceiling: DKS~20, RSS=7.
""",
        "r1_dks": 20, "r1_rss": 7, "r1_stages": 6,
    },
    "BIRD-SQL": {
        "repo": "https://github.com/AlibabaResearch/DAMO-ConvAI (bird/ subdirectory)",
        "paper": "Li et al., NeurIPS 2024",
        "domain": "Text-to-SQL Benchmark",
        "evidence": """
- Pipeline: Input Loading → Schema Extraction → Schema Serialization → Prompt Engineering → Model Inference → SQL Post-processing → Execution Evaluation
- 95 databases spanning 37 professional domains, 33.4 GB total.
- Three schema serialization formats (peteshaw, verbose, natural language).
- Value sampling from DB columns (get_database_matches()).
- SQL normalization rules: lowercase, comma fixing, whitespace normalization, keyword escaping.
- No curated domain glossary (column names are schema metadata, not a glossary).
- DB schema as implicit ontology (tables, columns, types, PK, FK).
- bridge_content_encoder for string-based value matching (not neural embedding).
- English only. No regulatory encoding.
- Dual pipeline: ICL (GPT) and fine-tuning (T5).
- EX (execution accuracy) + VES (valid efficiency score) evaluation.
""",
        "r1_dks": 26, "r1_rss": 3, "r1_stages": 7,
    },
    "Wheatley": {
        "repo": "https://github.com/jolibrain/wheatley",
        "paper": "Musik et al., CPAIOR 2024",
        "domain": "RL Job-Shop Scheduling",
        "evidence": """
- Pipeline: Instance Parsing → State Init → Feature Extraction → Graph Construction → GNN Inference → Action Masking → PPO Training → Validation
- 7 edge types in heterograph (prec/rprec/rc/rp/rrp/pool/self) — computational scheduling constraints.
- State representation: 16-28 features per node (TCT, is_affected, selectable, machine_id, MWKR).
- args.py: 800+ lines of domain parameters.
- 5 reward models (Sparse, L2D, Tassel, Intrinsic, Uncertain), 3 transition models, 5 dispatching heuristics = 13 profiles.
- Action masking rules, precedence enforcement, resource capacity constraints (~15-20 rule patterns).
- Taillard/SM/Patterson format conversions (cross-format, not cross-ontology).
- English/Python only. Minimal regulatory (feasibility constraints as operational rules).
- No vector search. GNN-based action selection (not retrieval).
""",
        "r1_dks": 33, "r1_rss": 2, "r1_stages": 8,
    },
    "FinSQL": {
        "repo": "https://github.com/bigbigwatermalon/FinSQL",
        "paper": "Li et al., SIGMOD Companion 2024",
        "domain": "Financial Text-to-SQL",
        "evidence": """
- Pipeline: Preprocessing → Schema Classifier Training → Schema Prediction → Hybrid Data Augmentation → LoRA Fine-tuning → Inference → Self-consistency Voting
- 78 financial tables across 3 databases (ccks_stock/ccks_fund/ccks_macro).
- 1,170 column names encoding financial terminology (mainoperincome, grossprofit, fundreturn, benchgrforthisweek, adjfreefloatratio).
- 614 FK relationships. ccks_stock: 31 tables, 434 columns.
- Bilingual CN/EN parallel corpus (3,966 identical question pairs).
- RoBERTa-large Cross-Encoder with BiLSTM + 8-head Cross-Attention for schema ranking.
- 3 augmentation profiles (COT/Synonymous/Skeleton).
- No explicit regex/rule patterns (fuzzy matching via fuzzywuzzy).
- No regulatory encoding.
- Learned Cross-Encoder dynamically selects top-k tables/columns per query.
- 8-way self-consistency voting for SQL generation.
""",
        "r1_dks": 35, "r1_rss": 6, "r1_stages": 7,
    },
    "SpeechBrain": {
        "repo": "https://github.com/speechbrain/speechbrain",
        "paper": "Ravanelli et al., arXiv:2106.04624, 2021",
        "domain": "General-Purpose Speech Toolkit",
        "evidence": """
- Comprehensive speech toolkit: ASR, TTS, speaker verification, language ID, etc.
- BPE tokenizer (1000/5000 vocab) + grapheme-to-phoneme (G2P) conversion.
- HyperPyYAML configs (240+ lines per recipe). 43 datasets × 110+ YAML configs.
- Beam search rules (beam=80, coverage penalty, attention shift) + CTC prefix scoring.
- seq2seq.py: 77KB decoder with attention mechanisms.
- Feature extraction: MFCC, Fbank, spectrogram parameters explicitly configured.
- VoxLingua107 (107 languages). Multi-language support.
- No retrieval pipeline (speech processing, not information retrieval).
- No regulatory/compliance encoding beyond standard licensing.
""",
        "r1_dks": 37, "r1_rss": 4, "r1_stages": 8,
    },
    "Logic-LLM": {
        "repo": "https://github.com/teacherpeterpan/Logic-LLM",
        "paper": "Pan et al., ACL 2024",
        "domain": "Neuro-Symbolic Reasoning",
        "evidence": """
- Pipeline: Problem parsing → Logic program generation → Solver dispatch → Inference → Answer extraction
- FOL grammar parser (fol_parser.py) defining first-order logic syntax.
- 4 solver DSLs: Prover9 (FOL), PyKE (forward/backward chaining), CSP (constraint satisfaction), SAT.
- logic_program.py: prompt routing to select solver based on problem type.
- code_translator.py: cross-representation translation (NL → FOL → solver-specific DSL).
- Formal logic terms: universal/existential quantifiers, predicates, constants, implications.
- ~30-40 logic-specific terms but NOT in a curated glossary file — they are grammar rules.
- No ontology/KG file. FOL grammar is a computational structure, not a semantic ontology.
- No database schema. No multi-language support.
- No vector search. Rule-based solver dispatch (not ML routing).
""",
        "r1_dks": 41, "r1_rss": 3, "r1_stages": 6,
    },
    "AudioCraft": {
        "repo": "https://github.com/facebookresearch/audiocraft",
        "paper": "Copet et al., NeurIPS 2023",
        "domain": "Music/Audio Generation",
        "evidence": """
- Multi-model architecture: MusicGen, AudioGen, JASCO, MAGNeT, EnCodec.
- conditioners.py: 1700+ lines defining conditioning mechanisms (text, melody, chroma, beat).
- 68 YAML configuration files for different model variants and training recipes.
- Music-specific features: chroma extraction, beat tracking, melody conditioning.
- EnCodec: neural audio codec with quantization (RVQ).
- music_dataset.py: audio dataset handling with genre/tempo/instrument metadata.
- metrics/: audio quality metrics (FAD, IS, CLAP score, etc.).
- Multi-model conditioning: T5 text encoder → cross-attention for text-to-music.
- No vector search or retrieval (generative model, not retrieval-based).
- No regulatory encoding. No access control.
- English text conditioning only (T5-based).
""",
        "r1_dks": 42, "r1_rss": 4, "r1_stages": 7,
    },
    "EmotiVoice": {
        "repo": "https://github.com/netease-youdao/EmotiVoice",
        "paper": "N/A (NetEase Youdao)",
        "domain": "Emotional Text-to-Speech",
        "evidence": """
- Pipeline: Text normalization → G2P (frontend) → Style encoding (SimBERT) → Acoustic model → Vocoder
- Bilingual Chinese-English TTS with separate frontends (frontend_cn.py, frontend_en.py).
- 502 phoneme tokens in lexicon. cmudict.py with 84 ARPAbet symbols.
- 206K Chinese pronunciation entries (cn2an system for numeral conversion).
- 2,013 speaker profiles with emotion/pitch/energy/speed labels.
- 7 emotion categories encoded in style tokens.
- SimBERT-based style encoder (simbert.py) for emotion embedding.
- Chinese numeral system (cn2an/conf.py) with extensive conversion rules.
- G2P pipeline: text → pinyin → phoneme with tone markers.
- No vector search. No retrieval pipeline.
- No regulatory encoding.
""",
        "r1_dks": 47, "r1_rss": 3, "r1_stages": 8,
    },
    "GraphRAG": {
        "repo": "https://github.com/microsoft/graphrag",
        "paper": "Edge et al., arXiv:2404.16130, 2024",
        "domain": "Graph-Based RAG",
        "evidence": """
- 10-stage pipeline: Document chunking → Entity extraction → Relationship extraction → Community detection → Report generation → Indexing → Query routing → Local search → Global search → Response synthesis
- 7 data model classes: Entity, Relationship, Community, CommunityReport, Document, TextUnit, Covariate.
- KG construction as core purpose: LLM-based entity/relationship extraction → hierarchical Leiden community detection.
- 4 search modes: local, global, drift, basic.
- hierarchical_leiden.py: multi-level community detection algorithm.
- Graph database integration (default: LanceDB for vector, NetworkX/Neo4j for graph).
- Entity extraction prompts with domain-configurable entity types.
- No curated domain glossary FILE — but entity types and graph vocabulary in prompts.
- factory.py: 10-stage workflow orchestration.
- Community reports as hierarchical summarization (like an auto-generated ontology).
- No regulatory encoding. English-focused.
- D1 note: 7 data model classes (Entity, Relationship, Community, etc.) have 50+ typed
  field constants (id, title, type, description, rank, weight, etc.) that define domain
  entity attributes. These field constants count toward D1 per rubric clarification.
- D6 note: LLM-orchestrated procedural rules count as code-level rules: gleaning loops
  (iterative entity extraction until saturation), budget-based context truncation,
  community-level graph pruning, claim extraction with status tracking. ~10-15 rule patterns.
- D7 note: Configurable {language} template parameter for multi-language entity extraction prompts.
""",
        "r1_dks": 57, "r1_rss": 8, "r1_stages": 10,
    },
    "MetaGPT": {
        "repo": "https://github.com/geekan/MetaGPT",
        "paper": "Hong et al., ICLR 2024",
        "domain": "Role-Based Multi-Agent SE Framework",
        "evidence": """
- Pipeline: User requirement → TeamLeader routing → ProductManager (PRD) → Architect (design) → ProjectManager (task plan) → Engineer (code) → QA (test); ~6-stage role-SOP assembly line.
- No domain glossary/lexicon. Closest is a software-engineering TaskType enum (EDA, feature_engineering, model_train), not a subject-matter lexicon.
- No domain ontology. A graph repository encodes code structure (class/function edges for UML reconstruction), not knowledge-domain concepts.
- No SQL schema. Extensive Pydantic typed models (Message, Task, Plan, UMLClassView, CodingContext) as structured artifacts.
- Strong role/SOP/template structuring: 16 preset role classes (profile/goal/constraints) and ActionNode PRD/design templates with typed expected fields.
- Code-level rules are SE guardrails (LLM-output repair regex, tree-sitter AST sanitizing), not domain validation.
- Output-language config (Chinese/English) on some roles; not localized domain logic.
- No regulatory/PII/compliance encoding.
- Retrieval: pluggable RAG (BM25/FAISS/Chroma/Elasticsearch) with hybrid fusion and rerankers; dynamic LLM tool-command routing and dual-tier memory; no trained intent classifier, no access control.
""",
        "r1_dks": 37, "r1_rss": 4, "r1_stages": 6,
    },
    "DeepTutor": {
        "repo": "https://github.com/deeptutor (educational AI tutor)",
        "paper": "N/A (open-source application)",
        "domain": "Educational AI Tutor (RAG)",
        "evidence": """
- Pipeline: Document ingestion → file-type routing → text extraction → embedding → vector indexing (LlamaIndex) → top-k retrieval → LLM generation; the book-generation flow adds ideation → source exploration → spine/concept-graph synthesis → critique → page planning → block compilation (~8 stages).
- No curated domain glossary/lexicon. Domain knowledge in pre-trained embeddings/LLM.
- Has a per-document concept graph (ConceptNode/ConceptEdge/ConceptGraph), rendered as Mermaid; generated per document, not a curated global ontology.
- No cross-standard mappings (only embedding-provider API adapters and a MIME/extension table).
- Has DB schema: SQLite DDL (sessions/messages/turns tables) for session memory; on-disk persisted vector indexes; Pydantic models throughout (Book/Spine/Page/ConceptGraph).
- Rich prompt-template library: ~98 YAML prompt files across 11 agent/block stages, each in English and Chinese.
- Code-level guardrails: regex filename sanitization, extension/MIME allowlists, size caps; email channel behind a consent flag. No domain extraction rules.
- Bilingual English/Chinese (paired en/zh prompts, web i18n). No regulatory/student-data compliance.
- Retrieval: single-mode dense vector (top-k=5); a SmartRetriever adds LLM multi-query expansion; no BM25/reranker (a "hybrid" mode is configured but not wired); SQLite session memory; no intent classifier, no access control.
""",
        "r1_dks": 42, "r1_rss": 6, "r1_stages": 8,
    },
    "PulseAI": {
        "repo": "https://github.com/pulse-ai (AI/ML information monitoring)",
        "paper": "N/A (open-source application)",
        "domain": "AI/ML Information Monitoring",
        "evidence": """
- Pipeline: Scrape (multi-source) → dedupe (hash-similarity) → summarize (LLM) → publish (X/Medium) → finalize; ~6 linear LangGraph nodes.
- Has a shallow domain lexicon: hardcoded AI/ML keyword filter lists and a 25-term tag vocabulary, inline in code (not an external glossary file).
- No ontology/knowledge-graph. Flat tag strings only.
- No cross-standard/cross-format mappings.
- Has DB schema: SQLAlchemy ORM with ~10 tables (news_items, summaries, reports, posts, users, sessions) plus Pydantic models; SQLite.
- Minimal config presets: hardcoded RSS feed list, subreddit defaults, publish-target toggles via env vars.
- Code-level rules: keyword-substring filters and a fixed similarity threshold (0.85); minimal regex; no rule engine.
- English only. No regulatory/PII compliance (security hygiene only: bcrypt/JWT/Fernet for auth).
- Multi-source fetch + keyword filter + embedding dedup/novelty ranking + time-based scheduling; no query routing, no intent classification, no fusion, no content-pipeline access control.
""",
        "r1_dks": 14, "r1_rss": 1, "r1_stages": 6,
    },
}

# ============================================================
# 3. Agent Personas
# ============================================================
AGENT_PERSONAS = {
    "SW_Architect": {
        "name": "Software Architect",
        "background": """You are a senior software architect with 12+ years of experience
designing large-scale production systems. You have deep expertise in system design,
microservices, database schemas, API design, and production deployment patterns.
You are strict about what counts as "structured" — you value explicit schemas,
typed interfaces, and well-documented configurations. You are skeptical of counting
implicit knowledge or pre-trained model weights as domain artifacts.""",
        "bias_note": "Tends to score D4 (Schema) strictly — expects SQL DDL or typed schemas. "
                     "Scores RSS strictly — distinguishes retrieval from model inference.",
    },
    "ML_Researcher": {
        "name": "ML Researcher",
        "background": """You are a machine learning researcher who has published at
NeurIPS, ACL, and ICML. You review papers regularly and evaluate systems based on
their technical novelty and domain modeling depth. You understand that knowledge can
be encoded in various forms — ontologies, embeddings, grammars, graph structures.
You are more generous in interpreting domain knowledge artifacts, recognizing that
FOL grammars, GNN edge types, and conditioning mechanisms can encode domain semantics.
However, you distinguish between domain-specific knowledge and general ML machinery.""",
        "bias_note": "May score D2 (Ontology) more generously for graph-based systems. "
                     "Careful about D1 — distinguishes curated terms from model vocabulary.",
    },
    "Data_Engineer": {
        "name": "Data Engineer",
        "background": """You are a senior data engineer with 10 years of experience
building ETL pipelines, data warehouses, and data governance systems. You work with
SQL DDL daily, manage data catalogs, and enforce data quality rules. You value
explicit data schemas, column-level documentation, and business rule validation.
You are practical — if column names encode domain terms (like financial column names),
you count that as a domain lexicon even without a separate glossary file.""",
        "bias_note": "May score D1 and D4 more generously for data-rich systems. "
                     "Strict on R3 (permissions) and D8 (regulatory).",
    },
    "Domain_Specialist": {
        "name": "Cross-Domain Specialist",
        "background": """You are an interdisciplinary researcher who has worked across
healthcare informatics, financial technology, and NLP. You understand domain-specific
standards (ICD-10, HL7 FHIR, XBRL) and can evaluate whether a system properly
encodes domain knowledge. You focus on D3 (cross-ontology mapping), D7 (multi-language),
and D8 (regulatory compliance) as these require genuine domain expertise to implement.
You are strict about D3 — format conversions (e.g., JSON vs YAML) are NOT cross-ontology.""",
        "bias_note": "Strict on D3 — requires genuine cross-standard mapping. "
                     "Generous on D8 when domain constraints are encoded as operational rules.",
    },
    "Junior_Developer": {
        "name": "Junior Developer",
        "background": """You are a 3rd-year software developer with a CS degree.
You follow rubric criteria literally and precisely. If the rubric says "glossary file,"
you look for an actual file named glossary or a dictionary data structure with term
definitions. If it says "ontology," you look for a graph data structure with nodes
and edges representing concepts. You do not infer or extrapolate — you score based
on what is explicitly visible in the codebase. You may miss subtle domain knowledge
artifacts but you provide a consistent, conservative baseline.""",
        "bias_note": "Most literal rubric interpretation. Conservative baseline. "
                     "May underscore systems where knowledge is implicitly encoded.",
    },
}

# ============================================================
# 4. Scoring Prompt Template
# ============================================================
SCORING_PROMPT_TEMPLATE = """You are {persona_name}.

{persona_background}

## Your Task

Score the following open-source system using the DKS/RSS rubric below.
You must score INDEPENDENTLY — do not try to guess what other raters scored.
For each component, provide:
1. A numeric score (must be one of the valid values in the rubric)
2. A 1-2 sentence justification citing specific evidence

{rubric}

---

## System to Score: {system_name}

**Repository**: {system_repo}
**Paper**: {system_paper}
**Domain**: {system_domain}

### Source Code Evidence

{system_evidence}

---

## Output Format

You MUST respond with ONLY a valid JSON object (no markdown fences, no explanation before/after).
Use this exact structure:

{{
  "system": "{system_name}",
  "rater": "{persona_id}",
  "dks": {{
    "D1": {{"score": <int>, "justification": "<str>"}},
    "D2": {{"score": <int>, "justification": "<str>"}},
    "D3": {{"score": <int>, "justification": "<str>"}},
    "D4": {{"score": <int>, "justification": "<str>"}},
    "D5": {{"score": <int>, "justification": "<str>"}},
    "D6": {{"score": <int>, "justification": "<str>"}},
    "D7": {{"score": <int>, "justification": "<str>"}},
    "D8": {{"score": <int>, "justification": "<str>"}}
  }},
  "rss": {{
    "R1": {{"score": <int>, "justification": "<str>"}},
    "R2": {{"score": <int>, "justification": "<str>"}},
    "R3": {{"score": <int>, "justification": "<str>"}},
    "R4": {{"score": <int>, "justification": "<str>"}},
    "R5": {{"score": <int>, "justification": "<str>"}}
  }},
  "dks_total": <int>,
  "rss_total": <int>,
  "total": <int>
}}
"""

# ============================================================
# 5. LLM Inference
# ============================================================
def call_vllm(prompt: str, max_retries: int = 3) -> str:
    """Call vLLM via /completions with empty think block to suppress Qwen3 reasoning."""
    filled_prompt = (
        "<|im_start|>user\n"
        + prompt
        + "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n"
    )
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{VLLM_BASE_URL}/completions",
                json={
                    "model": MODEL_NAME,
                    "prompt": filled_prompt,
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                    "stop": ["<|im_end|>", "<|im_start|>"],
                },
                timeout=180,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["text"].strip()
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                logger.error("All retries failed")
                return ""
    return ""


def parse_scoring_response(raw: str) -> Optional[Dict]:
    """Parse JSON from LLM response, handling markdown fences."""
    # Strip markdown code fences
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    try:
        data = json.loads(raw)
        # Validate structure
        dks_total = sum(data["dks"][k]["score"] for k in ["D1","D2","D3","D4","D5","D6","D7","D8"])
        rss_total = sum(data["rss"][k]["score"] for k in ["R1","R2","R3","R4","R5"])
        data["dks_total"] = dks_total
        data["rss_total"] = rss_total
        data["total"] = dks_total + rss_total
        return data
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error(f"Failed to parse response: {e}\nRaw: {raw[:500]}")
        return None


# ============================================================
# 6. Run All Scorings
# ============================================================
def run_all_scorings() -> List[Dict]:
    """Run 5 personas × 12 systems = 60 scorings."""
    results = []
    total = len(AGENT_PERSONAS) * len(SYSTEM_EVIDENCE)
    done = 0

    for persona_id, persona in AGENT_PERSONAS.items():
        for system_name, system in SYSTEM_EVIDENCE.items():
            done += 1
            logger.info(f"[{done}/{total}] {persona_id} scoring {system_name}...")

            prompt = SCORING_PROMPT_TEMPLATE.format(
                persona_name=persona["name"],
                persona_background=persona["background"],
                persona_id=persona_id,
                rubric=REFINED_RUBRIC,
                system_name=system_name,
                system_repo=system["repo"],
                system_paper=system["paper"],
                system_domain=system["domain"],
                system_evidence=system["evidence"],
            )

            raw_response = call_vllm(prompt)
            parsed = parse_scoring_response(raw_response)

            if parsed:
                parsed["persona_id"] = persona_id
                parsed["persona_name"] = persona["name"]
                results.append(parsed)
                logger.info(
                    f"  → DKS={parsed['dks_total']}, RSS={parsed['rss_total']}, "
                    f"Total={parsed['total']}"
                )
            else:
                logger.error(f"  → FAILED to parse response for {persona_id}/{system_name}")
                # Save raw response for debugging
                fail_path = RESULTS_DIR / f"failed_{persona_id}_{system_name}.txt"
                fail_path.write_text(raw_response)

            # Small delay to avoid overwhelming the server
            time.sleep(0.5)

    return results


# ============================================================
# 7. Statistical Analysis
# ============================================================
def cohens_kappa_weighted(r1: np.ndarray, r2: np.ndarray, n_bins: int = 4) -> float:
    """Compute quadratic-weighted Cohen's Kappa after binning scores."""
    # Bin into categories
    bins = np.linspace(0, max(r1.max(), r2.max()) + 1, n_bins + 1)
    r1_binned = np.digitize(r1, bins) - 1
    r2_binned = np.digitize(r2, bins) - 1
    n_bins_actual = max(r1_binned.max(), r2_binned.max()) + 1

    # Confusion matrix
    cm = np.zeros((n_bins_actual, n_bins_actual), dtype=float)
    for i in range(len(r1_binned)):
        cm[r1_binned[i], r2_binned[i]] += 1

    n = cm.sum()
    if n == 0:
        return 0.0

    # Quadratic weights
    w = np.zeros((n_bins_actual, n_bins_actual))
    for i in range(n_bins_actual):
        for j in range(n_bins_actual):
            w[i, j] = (i - j) ** 2 / (n_bins_actual - 1) ** 2

    # Expected
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    expected = np.outer(row_sum, col_sum) / n

    po = 1 - (w * cm).sum() / n
    pe = 1 - (w * expected).sum() / n

    if pe == 0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(ratings_matrix: np.ndarray, n_categories: int) -> float:
    """
    Compute Fleiss' Kappa for multiple raters.

    Args:
        ratings_matrix: (n_subjects × n_raters) matrix of category assignments
        n_categories: number of categories
    """
    n_subjects, n_raters = ratings_matrix.shape

    # Count category assignments per subject
    counts = np.zeros((n_subjects, n_categories), dtype=float)
    for i in range(n_subjects):
        for j in range(n_raters):
            cat = int(ratings_matrix[i, j])
            if 0 <= cat < n_categories:
                counts[i, cat] += 1

    # P_i for each subject
    P_i = np.zeros(n_subjects)
    for i in range(n_subjects):
        P_i[i] = (np.sum(counts[i] ** 2) - n_raters) / (n_raters * (n_raters - 1))

    P_bar = P_i.mean()

    # P_j for each category
    p_j = counts.sum(axis=0) / (n_subjects * n_raters)
    P_e_bar = np.sum(p_j ** 2)

    if P_e_bar == 1.0:
        return 1.0 if P_bar == 1.0 else 0.0
    return (P_bar - P_e_bar) / (1 - P_e_bar)


def icc_2k(ratings_matrix: np.ndarray) -> float:
    """
    Compute ICC(2,k) — two-way random, average measures.

    Args:
        ratings_matrix: (n_subjects × n_raters) matrix of continuous scores
    """
    n, k = ratings_matrix.shape
    grand_mean = ratings_matrix.mean()

    # Mean squares
    row_means = ratings_matrix.mean(axis=1)
    col_means = ratings_matrix.mean(axis=0)

    SS_rows = k * np.sum((row_means - grand_mean) ** 2)
    SS_cols = n * np.sum((col_means - grand_mean) ** 2)
    SS_total = np.sum((ratings_matrix - grand_mean) ** 2)
    SS_error = SS_total - SS_rows - SS_cols

    MS_rows = SS_rows / (n - 1) if n > 1 else 0
    MS_cols = SS_cols / (k - 1) if k > 1 else 0
    MS_error = SS_error / ((n - 1) * (k - 1)) if (n > 1 and k > 1) else 0

    # ICC(2,k)
    denom = MS_rows + (MS_cols - MS_error) / n
    if denom == 0:
        return 0.0
    return (MS_rows - MS_error) / denom


def compute_all_statistics(results: List[Dict]) -> Dict:
    """Compute comprehensive inter-rater reliability statistics."""
    systems = list(SYSTEM_EVIDENCE.keys())
    personas = list(AGENT_PERSONAS.keys())
    n_systems = len(systems)
    n_raters = len(personas)

    # Build score matrices: (n_systems × n_raters)
    dks_matrix = np.full((n_systems, n_raters), np.nan)
    rss_matrix = np.full((n_systems, n_raters), np.nan)
    total_matrix = np.full((n_systems, n_raters), np.nan)

    for r in results:
        sys_idx = systems.index(r["system"]) if r["system"] in systems else -1
        rat_idx = personas.index(r["persona_id"]) if r["persona_id"] in personas else -1
        if sys_idx >= 0 and rat_idx >= 0:
            dks_matrix[sys_idx, rat_idx] = r["dks_total"]
            rss_matrix[sys_idx, rat_idx] = r["rss_total"]
            total_matrix[sys_idx, rat_idx] = r["total"]

    # Check for missing data
    valid_mask = ~np.isnan(total_matrix).any(axis=1)
    if valid_mask.sum() < n_systems:
        logger.warning(
            f"Missing data for {n_systems - valid_mask.sum()} systems, "
            "using only complete cases."
        )

    dks_valid = dks_matrix[valid_mask]
    rss_valid = rss_matrix[valid_mask]
    total_valid = total_matrix[valid_mask]

    stats = {
        "n_systems": int(valid_mask.sum()),
        "n_raters": n_raters,
        "n_scorings": len(results),
    }

    if valid_mask.sum() < 2:
        logger.error("Not enough valid data for statistics")
        return stats

    # --- A. Per-rater summary ---
    rater_summaries = {}
    for j, pid in enumerate(personas):
        col = total_valid[:, j]
        rater_summaries[pid] = {
            "mean_total": float(np.mean(col)),
            "std_total": float(np.std(col)),
            "mean_dks": float(np.mean(dks_valid[:, j])),
            "mean_rss": float(np.mean(rss_valid[:, j])),
        }
    stats["rater_summaries"] = rater_summaries

    # --- B. Pairwise Cohen's Kappa (quadratic weighted, 4-bin) ---
    pairwise_kappa = {}
    for i, j in itertools.combinations(range(n_raters), 2):
        key = f"{personas[i]}_vs_{personas[j]}"
        kappa = cohens_kappa_weighted(total_valid[:, i], total_valid[:, j], n_bins=4)
        pairwise_kappa[key] = round(float(kappa), 3)
    stats["pairwise_cohens_kappa"] = pairwise_kappa
    stats["mean_pairwise_kappa"] = round(
        float(np.mean(list(pairwise_kappa.values()))), 3
    )

    # --- C. Pairwise Spearman ρ ---
    pairwise_spearman = {}
    for i, j in itertools.combinations(range(n_raters), 2):
        key = f"{personas[i]}_vs_{personas[j]}"
        rho, p_val = scipy_stats.spearmanr(total_valid[:, i], total_valid[:, j])
        pairwise_spearman[key] = {"rho": round(float(rho), 3), "p": round(float(p_val), 4)}
    stats["pairwise_spearman"] = pairwise_spearman
    mean_rho = float(np.mean([v["rho"] for v in pairwise_spearman.values()]))
    stats["mean_spearman_rho"] = round(mean_rho, 3)

    # --- D. Pairwise Pearson r ---
    pairwise_pearson = {}
    for i, j in itertools.combinations(range(n_raters), 2):
        key = f"{personas[i]}_vs_{personas[j]}"
        r_val, p_val = scipy_stats.pearsonr(total_valid[:, i], total_valid[:, j])
        pairwise_pearson[key] = {"r": round(float(r_val), 3), "p": round(float(p_val), 4)}
    stats["pairwise_pearson"] = pairwise_pearson

    # --- E. Fleiss' Kappa (4-bin categorization) ---
    max_score = np.nanmax(total_valid) + 1
    n_bins = 4
    bins = np.linspace(0, max_score, n_bins + 1)
    binned = np.clip(np.digitize(total_valid, bins) - 1, 0, n_bins - 1)
    fk = fleiss_kappa(binned, n_bins)
    stats["fleiss_kappa_4bin"] = round(float(fk), 3)

    # --- F. ICC(2,k) — Intraclass Correlation Coefficient ---
    icc_dks = icc_2k(dks_valid)
    icc_rss = icc_2k(rss_valid)
    icc_total = icc_2k(total_valid)
    stats["icc_2k"] = {
        "dks": round(float(icc_dks), 3),
        "rss": round(float(icc_rss), 3),
        "total": round(float(icc_total), 3),
    }

    # --- G. R1 (Author) vs Agent Mean comparison ---
    agent_means = np.nanmean(total_valid, axis=1)
    r1_totals = np.array([
        SYSTEM_EVIDENCE[s]["r1_dks"] + SYSTEM_EVIDENCE[s]["r1_rss"]
        for s in systems if systems.index(s) < len(total_valid)
    ])[:len(agent_means)]

    r1_vs_agent_spearman, r1_p = scipy_stats.spearmanr(r1_totals, agent_means)
    r1_vs_agent_pearson, r1_p2 = scipy_stats.pearsonr(r1_totals, agent_means)
    r1_kappa = cohens_kappa_weighted(r1_totals, agent_means, n_bins=4)

    stats["r1_vs_agent_mean"] = {
        "spearman_rho": round(float(r1_vs_agent_spearman), 3),
        "spearman_p": round(float(r1_p), 4),
        "pearson_r": round(float(r1_vs_agent_pearson), 3),
        "pearson_p": round(float(r1_p2), 4),
        "cohens_kappa_4bin": round(float(r1_kappa), 3),
        "mean_abs_diff": round(float(np.mean(np.abs(r1_totals - agent_means))), 1),
    }

    # --- H. Per-system score table ---
    system_table = []
    for i, s in enumerate(systems):
        if i >= len(total_valid):
            break
        row = {
            "system": s,
            "r1_dks": SYSTEM_EVIDENCE[s]["r1_dks"],
            "r1_rss": SYSTEM_EVIDENCE[s]["r1_rss"],
            "r1_total": SYSTEM_EVIDENCE[s]["r1_dks"] + SYSTEM_EVIDENCE[s]["r1_rss"],
        }
        for j, pid in enumerate(personas):
            row[f"{pid}_dks"] = int(dks_valid[i, j])
            row[f"{pid}_rss"] = int(rss_valid[i, j])
            row[f"{pid}_total"] = int(total_valid[i, j])
        row["agent_mean_total"] = round(float(np.mean(total_valid[i])), 1)
        row["agent_std_total"] = round(float(np.std(total_valid[i])), 1)
        system_table.append(row)
    stats["system_scores"] = system_table

    return stats


# ============================================================
# 8. Report Generation
# ============================================================
def generate_report(stats: Dict) -> str:
    """Generate a markdown report from statistics."""
    lines = [
        "# Multi-Agent Inter-Rater Reliability Report",
        f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Model**: {MODEL_NAME}",
        f"**Raters**: {stats['n_raters']} agent personas",
        f"**Systems**: {stats['n_systems']} external systems",
        f"**Total scorings**: {stats['n_scorings']}",
        "",
        "## 1. Summary Statistics",
        "",
        "| Metric | Value | Interpretation |",
        "|--------|-------|----------------|",
        f"| Fleiss' κ (4-bin) | {stats.get('fleiss_kappa_4bin', 'N/A')} | "
        f"{'Substantial' if stats.get('fleiss_kappa_4bin', 0) >= 0.61 else 'Moderate' if stats.get('fleiss_kappa_4bin', 0) >= 0.41 else 'Fair'} agreement |",
        f"| Mean pairwise Cohen's κ | {stats.get('mean_pairwise_kappa', 'N/A')} | Weighted, 4-bin |",
        f"| Mean Spearman ρ | {stats.get('mean_spearman_rho', 'N/A')} | Rank-order agreement |",
        f"| ICC(2,k) Total | {stats.get('icc_2k', {}).get('total', 'N/A')} | "
        f"{'Excellent' if stats.get('icc_2k', {}).get('total', 0) >= 0.75 else 'Good' if stats.get('icc_2k', {}).get('total', 0) >= 0.60 else 'Moderate'} reliability |",
        f"| ICC(2,k) DKS | {stats.get('icc_2k', {}).get('dks', 'N/A')} | DKS subscale |",
        f"| ICC(2,k) RSS | {stats.get('icc_2k', {}).get('rss', 'N/A')} | RSS subscale |",
        "",
        "## 2. R1 (Author) vs Agent Mean",
        "",
    ]

    r1_stats = stats.get("r1_vs_agent_mean", {})
    lines.extend([
        "| Metric | Value |",
        "|--------|-------|",
        f"| Spearman ρ | {r1_stats.get('spearman_rho', 'N/A')} (p={r1_stats.get('spearman_p', 'N/A')}) |",
        f"| Pearson r | {r1_stats.get('pearson_r', 'N/A')} (p={r1_stats.get('pearson_p', 'N/A')}) |",
        f"| Cohen's κ (4-bin) | {r1_stats.get('cohens_kappa_4bin', 'N/A')} |",
        f"| Mean abs diff | {r1_stats.get('mean_abs_diff', 'N/A')} |",
        "",
        "## 3. Per-System Scores",
        "",
    ])

    # Build per-system table
    personas = list(AGENT_PERSONAS.keys())
    header = "| System | R1 | " + " | ".join(personas) + " | Agent Mean ± Std |"
    sep = "|--------|----" + "|----" * len(personas) + "|-----------------|"
    lines.extend([header, sep])

    for row in stats.get("system_scores", []):
        cells = [
            row["system"],
            str(row["r1_total"]),
        ]
        for pid in personas:
            cells.append(str(row.get(f"{pid}_total", "?")))
        cells.append(f"{row['agent_mean_total']} ± {row['agent_std_total']}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## 4. Pairwise Cohen's κ Matrix",
        "",
    ])
    for key, val in stats.get("pairwise_cohens_kappa", {}).items():
        lines.append(f"- {key}: κ = {val}")

    lines.extend([
        "",
        f"**Mean pairwise κ = {stats.get('mean_pairwise_kappa', 'N/A')}**",
        "",
        "## 5. Pairwise Spearman ρ",
        "",
    ])
    for key, val in stats.get("pairwise_spearman", {}).items():
        lines.append(f"- {key}: ρ = {val['rho']} (p={val['p']})")

    lines.extend([
        "",
        f"**Mean ρ = {stats.get('mean_spearman_rho', 'N/A')}**",
        "",
        "## 6. Per-Rater Summary",
        "",
        "| Rater | Mean DKS | Mean RSS | Mean Total | Std Total |",
        "|-------|----------|----------|------------|-----------|",
    ])
    for pid, summary in stats.get("rater_summaries", {}).items():
        lines.append(
            f"| {pid} | {summary['mean_dks']:.1f} | {summary['mean_rss']:.1f} | "
            f"{summary['mean_total']:.1f} | {summary['std_total']:.1f} |"
        )

    return "\n".join(lines)


# ============================================================
# 9. Main
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("Multi-Agent Inter-Rater Reliability Evaluation")
    logger.info(f"Model: {MODEL_NAME} | Temperature: {TEMPERATURE}")
    logger.info(f"Personas: {len(AGENT_PERSONAS)} | Systems: {len(SYSTEM_EVIDENCE)}")
    logger.info(f"Total scorings: {len(AGENT_PERSONAS) * len(SYSTEM_EVIDENCE)}")
    logger.info("=" * 60)

    # Run all scorings
    results = run_all_scorings()
    logger.info(f"\nCompleted {len(results)} scorings successfully.")

    # Save raw results
    raw_path = RESULTS_DIR / "interrater_raw_results.json"
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Raw results saved to {raw_path}")

    # Compute statistics
    stats = compute_all_statistics(results)

    # Save statistics
    stats_path = RESULTS_DIR / "interrater_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Statistics saved to {stats_path}")

    # Generate report
    report = generate_report(stats)
    report_path = RESULTS_DIR / "interrater_report.md"
    report_path.write_text(report)
    logger.info(f"Report saved to {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Fleiss' κ (4-bin):       {stats.get('fleiss_kappa_4bin', 'N/A')}")
    print(f"Mean pairwise Cohen's κ: {stats.get('mean_pairwise_kappa', 'N/A')}")
    print(f"Mean Spearman ρ:         {stats.get('mean_spearman_rho', 'N/A')}")
    print(f"ICC(2,k) Total:          {stats.get('icc_2k', {}).get('total', 'N/A')}")
    print(f"ICC(2,k) DKS:            {stats.get('icc_2k', {}).get('dks', 'N/A')}")
    print(f"ICC(2,k) RSS:            {stats.get('icc_2k', {}).get('rss', 'N/A')}")

    r1s = stats.get("r1_vs_agent_mean", {})
    print(f"\nR1 vs Agent Mean:")
    print(f"  Spearman ρ:   {r1s.get('spearman_rho', 'N/A')} (p={r1s.get('spearman_p', 'N/A')})")
    print(f"  Cohen's κ:    {r1s.get('cohens_kappa_4bin', 'N/A')}")
    print(f"  Mean |diff|:  {r1s.get('mean_abs_diff', 'N/A')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
