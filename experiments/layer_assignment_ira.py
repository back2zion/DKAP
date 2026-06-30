#!/usr/bin/env python3
"""
Layer Assignment Inter-Rater Reliability (IRA) Experiment
==========================================================
5 LLM agent personas independently assign each pipeline stage of
13 external systems to DKAP layers L1-L5.
Computes Fleiss' kappa, pairwise Cohen's kappa, and Spearman rho.

Author: Dooil Kwak, 2026-05-22
"""

import json, time, itertools, math, re, sys
from pathlib import Path
try:
    import requests
except ImportError:
    import subprocess; subprocess.run([sys.executable,"-m","pip","install","requests","-q"])
    import requests

VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME    = "Qwen3.6-27B-AWQ"
RESULTS_DIR   = Path("results/layer_assignment_ira")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# DKAP Layer Definitions (for agent prompt)
# ============================================================
LAYER_DEFINITIONS = """
## DKAP Five-Layer Architecture
Assign each pipeline stage to exactly ONE of the following layers:

**L1 – Input Normalization**
Purpose: Standardizes user input (text, files, images, audio) and performs
security/format validation. The stage receives raw external input and produces
a clean, normalized representation. Examples: tokenization, format conversion,
PII masking of input, prompt injection defense, input schema validation.

**L2 – Domain Knowledge Structuring**
Purpose: Produces STATIC domain artifacts before query time — declarative
representations (glossaries, schemas, ontologies, rule sets, knowledge graphs,
embeddings of domain corpora) that encode domain semantics and are prepared
ONCE and consumed repeatedly by L3. Key criterion: does this stage build a
persistent domain representation that is used by later retrieval/routing stages?
Examples: OMOP CDM schema indexing, financial taxonomy construction, KG building,
training domain-specific embeddings, preprocessing domain corpora into indices,
schema classification training.

**L3 – Domain-Conditioned Retrieval & Routing**
Purpose: Consumes L2 artifacts at RUNTIME to make dynamic retrieval and routing
decisions per query. Examples: vector search, BM25, intent classification,
permission-based document filtering, schema linking, adaptive routing between
models, re-ranking, RRF fusion across retrievers.

**L4 – AI Inference with Fallback**
Purpose: Executes AI inference using domain-conditioned context from L3. The
primary model inference step. Engine-agnostic: LLM, RL agent, VLM, ML ensemble.
Examples: LLM call with retrieved context, SQL generation, reinforcement learning
inference, VLM classification, ensemble voting, self-consistency sampling.

**L5 – Output Validation & Audit**
Purpose: Validates AI outputs and maintains audit/compliance records. Examples:
SQL syntax validation, PII masking of outputs, answer extraction from LLM
response, self-consistency voting (when used for validation), evaluation metric
computation, audit logging.

**Assignment rules:**
- Each stage gets exactly one label: L1, L2, L3, L4, or L5
- If a stage spans two layers, assign to the PRIMARY function
- Training/offline preparation stages = L2 (they build domain artifacts)
- Runtime retrieval = L3, Runtime inference = L4, Post-processing = L5
"""

# ============================================================
# 13 External Systems — Pipeline Stages
# (extracted from supplementary analysis)
# ============================================================
SYSTEMS = {
    "MedRAG": {
        "stages": [
            "Corpus Loading: Load pre-indexed corpora (PubMed/StatPearls/Textbooks/Wikipedia) into FAISS index",
            "Retrieval: BM25/Contriever/SPECTER/MedCPT multi-retriever search + RRF fusion",
            "LLM Inference: GPT-4/Mixtral/Llama CoT prompting with retrieved snippets",
            "Answer Extraction: Regex-based answer parsing from LLM output",
        ]
    },
    "Pulse_AI": {
        "stages": [
            "Input Parsing: Parse news article URL or text input",
            "Source Classification: Classify source credibility and type",
            "Content Fetching: Scrape and normalize article content",
            "Summarization: LLM-based extractive/abstractive summarization",
            "Output Formatting: Structure summary with metadata tags",
            "Delivery: Push notification or API response",
        ]
    },
    "CrewAI": {
        "stages": [
            "Task Decomposition: Parse user goal into sub-tasks",
            "Agent Role Assignment: Assign specialized agent personas from role library",
            "Memory Initialization: Load short-term and long-term agent memory",
            "Tool Binding: Bind task-relevant tools to agents",
            "Multi-agent Execution: Sequential/hierarchical agent task execution",
            "Inter-agent Communication: Pass outputs between agents",
            "Result Aggregation: Merge agent outputs into final response",
        ]
    },
    "LlamaIndex": {
        "stages": [
            "Document Loading: Ingest documents via data connectors (PDF, web, DB)",
            "Chunking & Indexing: Split documents, embed chunks, build vector index",
            "Query Understanding: Parse and classify user query",
            "Retrieval: Vector similarity search over indexed chunks",
            "Re-ranking: Optional reranker model to refine retrieval",
            "Response Synthesis: LLM generates response from retrieved context",
        ]
    },
    "BIRD_SQL": {
        "stages": [
            "Schema Preprocessing: Extract table/column metadata, FK constraints from DB",
            "Schema Classification: RoBERTa cross-encoder scores table/column relevance",
            "Schema Linking: Select top-k tables and columns for prompt",
            "Data Augmentation: GPT-3.5 CoT/synonym/skeleton augmentation",
            "Hint Generation: External knowledge hints for complex queries",
            "SQL Generation: LLM generates SQL with schema-linked prompt",
            "SQL Validation: Execution-based validation and self-consistency",
        ]
    },
    "Wheatley": {
        "stages": [
            "Job Shop Problem Loading: Parse Taillard benchmark instances (n jobs, m machines)",
            "State Feature Extraction: Compute scheduling state features (load, makespan, etc.)",
            "Dispatching Rule Application: Apply MWKR/composite heuristic rules",
            "GNN Encoding: Graph neural network encodes job-machine graph topology",
            "RL Policy Inference: Actor network selects next job-machine assignment",
            "Schedule Update: Update partial schedule and compute reward",
            "Constraint Validation: Check precedence and machine conflict constraints",
            "Makespan Evaluation: Compute final makespan and optimality gap",
        ]
    },
    "FinSQL": {
        "stages": [
            "Dataset Preprocessing: Normalize SQL, extract schema labels, FK injection",
            "Schema Classifier Training: Train RoBERTa Cross-Encoder for schema ranking",
            "Schema Prediction: Generate ranked schema (top-k tables/columns per query)",
            "Hybrid Data Augmentation: GPT-3.5 CoT/synonymous/skeleton generation",
            "LLM LoRA Fine-tuning: Llama2-13B LoRA training on augmented data",
            "Inference + Post-processing: SQL generation + fuzzy schema matching",
            "Self-consistency Voting: 8-way semantic SQL comparison",
        ]
    },
    "MetaGPT": {
        "stages": [
            "Role & Workflow Loading: Initialize agent roles (PM, Architect, Engineer) from templates",
            "Requirement Analysis: PM agent parses user requirement into structured spec",
            "System Design: Architect agent generates class/sequence diagrams and API spec",
            "Code Generation: Engineer agents generate code per module spec",
            "Code Review: QA agent reviews generated code against spec",
            "Integration Testing: Run tests and validate output against requirement",
        ]
    },
    "SpeechBrain": {
        "stages": [
            "Audio Loading & Normalization: Load waveform, resample, normalize amplitude",
            "Feature Extraction: Compute MFCC/Fbank/raw waveform features",
            "Encoder: ECAPA-TDNN/wav2vec2 speech encoder",
            "Domain Adaptation Layer: Speaker/language adaptation embeddings",
            "Decoder / CTC / Attention: ASR beam search, TTS spectrogram synthesis",
            "Post-processing: Detokenization, prosody normalization",
            "Evaluation: WER/SER computation",
            "Experiment Logging: HyperPy YAML config + CSV logging",
        ]
    },
    "AudioCraft": {
        "stages": [
            "Audio Conditioning: Load genre/instrument profiles and style conditioning tensors",
            "Text Prompt Encoding: T5 encodes textual description into condition vector",
            "Codec Encoding: EnCodec compresses reference audio into discrete tokens",
            "Transformer LM Inference: MusicGen/AudioGen auto-regressive token generation",
            "Codec Decoding: EnCodec decodes discrete tokens to waveform",
            "Audio Post-processing: Loudness normalization, format conversion",
            "Output Delivery: Save/stream generated audio",
        ]
    },
    "DeepTutor": {
        "stages": [
            "Document Ingestion: Parse PDF textbooks into structured page/paragraph chunks",
            "Knowledge Graph Construction: Build concept dependency graph from textbook",
            "Student Model Initialization: Initialize student knowledge state vector",
            "Query Classification: Classify student question type (factual/conceptual/procedural)",
            "Context Retrieval: Retrieve relevant passages using student model + KG context",
            "Tutoring Response Generation: LLM generates Socratic hint or explanation",
            "Student Model Update: Update knowledge state based on student response",
            "Session Logging: Log interaction for learning analytics",
        ]
    },
    "EmotiVoice": {
        "stages": [
            "Text Preprocessing: Tokenize Chinese/English text, normalize numbers/symbols",
            "Phoneme Conversion: G2P conversion using 502-phoneme vocabulary",
            "Emotion Conditioning: Load emotion embedding (happy/sad/angry/etc.) from 2013 speaker profiles",
            "Acoustic Model: BERT+FastSpeech2 generates mel-spectrogram",
            "Vocoder: HiFi-GAN converts mel-spectrogram to waveform",
            "Speaker Adaptation: Apply speaker identity embedding for voice cloning",
            "Audio Post-processing: Resample, normalize, format conversion",
            "Evaluation: MOS estimation, speaker similarity scoring",
        ]
    },
    "GraphRAG": {
        "stages": [
            "Document Chunking: Split documents into fixed-size text chunks",
            "Entity & Relation Extraction: LLM extracts entity-relation triples from chunks",
            "Community Detection: Leiden algorithm builds hierarchical KG communities",
            "Community Report Generation: LLM summarizes each community into report",
            "Embedding Indexing: Embed entities, reports, chunks into vector store",
            "Query Routing: Classify query as local (entity) or global (community) search",
            "Retrieval: Local = entity-based KG search; Global = community report retrieval",
            "Response Generation: LLM synthesizes answer from retrieved context",
            "Map-Reduce Aggregation: For global queries, aggregate partial answers",
            "Final Response Formatting: Format and return answer with citations",
        ]
    },
}

# ============================================================
# Agent Personas
# ============================================================
PERSONAS = [
    {
        "id": "A1",
        "name": "Software Architect",
        "description": "Senior software architect with 15 years experience in enterprise AI system design. Focuses on separation of concerns and layered architecture principles.",
    },
    {
        "id": "A2",
        "name": "ML Research Scientist",
        "description": "ML researcher specializing in NLP and retrieval systems. Focuses on model training pipelines and inference optimization.",
    },
    {
        "id": "A3",
        "name": "Data Engineer",
        "description": "Data engineer with expertise in ETL pipelines, knowledge graphs, and data preprocessing for AI systems.",
    },
    {
        "id": "A4",
        "name": "Cross-Domain AI Specialist",
        "description": "AI consultant who has deployed AI systems across finance, healthcare, manufacturing. Focuses on domain adaptation and knowledge management.",
    },
    {
        "id": "A5",
        "name": "Junior Developer",
        "description": "Software engineer with 2 years AI development experience. Reads documentation carefully and applies definitions literally.",
    },
]

# ============================================================
# Helper functions
# ============================================================

def call_vllm(user_content: str, system_content: str = "", max_retries: int = 3) -> str:
    # Use /completions with pre-filled empty think block so Qwen3 skips thinking mode
    sys_part = f"<|im_start|>system\n{system_content}<|im_end|>\n" if system_content else ""
    filled_prompt = (
        sys_part
        + "<|im_start|>user\n"
        + user_content
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
                    "max_tokens": 2048,
                    "temperature": 0.0,
                    "stop": ["<|im_end|>", "<|im_start|>"],
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["text"].strip()
        except Exception as e:
            print(f"  [Attempt {attempt+1}] Error: {e}")
            time.sleep(5)
    return ""


def parse_assignments(raw: str, n_stages: int) -> list:
    """Extract L1-L5 assignments from agent response."""
    lines = raw.split("\n")
    assignments = []
    for line in lines:
        m = re.search(r'\bL([1-5])\b', line)
        if m:
            assignments.append(int(m.group(1)))
    # Trim or pad to n_stages
    if len(assignments) > n_stages:
        assignments = assignments[:n_stages]
    while len(assignments) < n_stages:
        assignments.append(3)  # Default L3 if parse fails
    return assignments


def build_prompt(persona: dict, system_name: str, stages: list) -> str:
    stages_text = "\n".join(
        f"  Stage {i+1}: {s}" for i, s in enumerate(stages)
    )
    return f"""You are a {persona['name']}. {persona['description']}

Your task: Assign each pipeline stage of the AI system "{system_name}" to one of five DKAP layers.

{LAYER_DEFINITIONS}

## System: {system_name}
Pipeline stages:
{stages_text}

## Instructions
For each stage, output EXACTLY ONE line in the format:
Stage N: LX

Where N is the stage number (1 to {len(stages)}) and X is the layer number (1-5).
Output all {len(stages)} assignments, one per line, with no additional commentary.
"""


def fleiss_kappa(ratings: list) -> float:
    """
    ratings: list of lists, ratings[i][j] = layer assigned by rater j to stage i
    Returns Fleiss' kappa.
    """
    n_items = len(ratings)
    n_raters = len(ratings[0])
    n_cats = 5  # L1..L5

    # Build count matrix
    counts = []
    for item_ratings in ratings:
        row = [0] * n_cats
        for r in item_ratings:
            row[r - 1] += 1
        counts.append(row)

    # P_i (proportion of agreeing pairs per item)
    P_i = []
    for row in counts:
        n = sum(row)
        if n * (n - 1) == 0:
            P_i.append(0.0)
        else:
            P_i.append((sum(c * (c - 1) for c in row)) / (n * (n - 1)))
    P_bar = sum(P_i) / n_items

    # p_j (marginal proportion for each category)
    total = n_items * n_raters
    p_j = [sum(counts[i][j] for i in range(n_items)) / total for j in range(n_cats)]
    P_e = sum(p ** 2 for p in p_j)

    if abs(1 - P_e) < 1e-10:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def cohens_kappa(r1: list, r2: list) -> float:
    n = len(r1)
    cats = list(range(1, 6))
    # Observed agreement
    po = sum(1 for a, b in zip(r1, r2) if a == b) / n
    # Expected agreement
    pe = sum(
        (r1.count(c) / n) * (r2.count(c) / n) for c in cats
    )
    return (po - pe) / (1 - pe) if abs(1 - pe) > 1e-10 else 1.0


def spearman_rho(r1: list, r2: list) -> float:
    n = len(r1)
    if n == 0:
        return 0.0
    def rank(lst):
        sorted_vals = sorted(set(lst))
        return [sorted_vals.index(v) + 1 for v in lst]
    xr, yr = rank(r1), rank(r2)
    mx, my = sum(xr) / n, sum(yr) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xr, yr))
    den = math.sqrt(sum((x - mx) ** 2 for x in xr) * sum((y - my) ** 2 for y in yr))
    return num / den if den else 0.0


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("DKAP Layer Assignment Inter-Rater Reliability Experiment")
    print("=" * 60)
    print(f"Systems: {len(SYSTEMS)}, Agents: {len(PERSONAS)}")

    # Check vLLM is ready
    for attempt in range(20):
        try:
            r = requests.get(f"{VLLM_BASE_URL}/models", timeout=10)
            if r.status_code == 200:
                print(f"vLLM ready: {r.json()['data'][0]['id']}")
                break
        except:
            print(f"Waiting for vLLM... ({attempt+1}/20)")
            time.sleep(15)
    else:
        print("ERROR: vLLM not ready. Exiting.")
        sys.exit(1)

    all_results = {}

    for sys_name, sys_data in SYSTEMS.items():
        stages = sys_data["stages"]
        print(f"\n[{sys_name}] {len(stages)} stages")
        all_results[sys_name] = {"stages": stages, "agent_assignments": {}}

        for persona in PERSONAS:
            prompt = build_prompt(persona, sys_name, stages)
            print(f"  Agent {persona['id']} ({persona['name']})...", end=" ", flush=True)
            raw = call_vllm(prompt)
            assignments = parse_assignments(raw, len(stages))
            all_results[sys_name]["agent_assignments"][persona["id"]] = assignments
            print(f"done: {assignments}")
            time.sleep(1)

    # Save raw results
    out_path = RESULTS_DIR / "layer_assignments_raw.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nRaw results saved: {out_path}")

    # --------------------------------------------------------
    # Compute statistics
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("INTER-RATER RELIABILITY STATISTICS")
    print("=" * 60)

    # Flatten all stage-level ratings across all systems
    all_ratings = []  # list of [a1,a2,a3,a4,a5] per stage
    for sys_name, sys_data in all_results.items():
        n_stages = len(sys_data["stages"])
        agent_ids = [p["id"] for p in PERSONAS]
        for stage_idx in range(n_stages):
            row = []
            for aid in agent_ids:
                row.append(sys_data["agent_assignments"][aid][stage_idx])
            all_ratings.append(row)

    n_total_stages = len(all_ratings)
    print(f"Total stage assignments: {n_total_stages} stages × {len(PERSONAS)} raters")

    # Fleiss' kappa
    fk = fleiss_kappa(all_ratings)
    print(f"Fleiss' κ (all stages, all systems): {fk:.3f}")

    # Per-system Fleiss' kappa
    print("\nPer-system Fleiss' κ:")
    for sys_name, sys_data in all_results.items():
        n_stages = len(sys_data["stages"])
        agent_ids = [p["id"] for p in PERSONAS]
        sys_ratings = []
        for stage_idx in range(n_stages):
            row = [sys_data["agent_assignments"][aid][stage_idx] for aid in agent_ids]
            sys_ratings.append(row)
        sk = fleiss_kappa(sys_ratings)
        print(f"  {sys_name}: κ={sk:.3f} (n={n_stages} stages)")

    # Pairwise Cohen's kappa between agents
    agent_ids = [p["id"] for p in PERSONAS]
    print("\nPairwise Cohen's κ between agents:")
    pairwise_kappas = []
    for a1, a2 in itertools.combinations(range(len(PERSONAS)), 2):
        r1 = [all_ratings[i][a1] for i in range(n_total_stages)]
        r2 = [all_ratings[i][a2] for i in range(n_total_stages)]
        ck = cohens_kappa(r1, r2)
        rho = spearman_rho(r1, r2)
        pairwise_kappas.append(ck)
        print(f"  {agent_ids[a1]} vs {agent_ids[a2]}: κ={ck:.3f}, ρ={rho:.3f}")

    mean_ck = sum(pairwise_kappas) / len(pairwise_kappas)
    print(f"\nMean pairwise Cohen's κ: {mean_ck:.3f}")

    # Agreement rate (exact match)
    agree_count = sum(1 for row in all_ratings if len(set(row)) == 1)
    agree_rate = agree_count / n_total_stages
    print(f"Full consensus (all 5 agree): {agree_count}/{n_total_stages} ({agree_rate:.1%})")

    # Layer distribution
    flat = [v for row in all_ratings for v in row]
    total = len(flat)
    print("\nLayer distribution:")
    for layer in range(1, 6):
        cnt = flat.count(layer)
        print(f"  L{layer}: {cnt} ({cnt/total:.1%})")

    # Save summary
    summary = {
        "total_stages": n_total_stages,
        "n_raters": len(PERSONAS),
        "fleiss_kappa": round(fk, 3),
        "mean_pairwise_cohens_kappa": round(mean_ck, 3),
        "full_consensus_rate": round(agree_rate, 3),
    }
    with open(RESULTS_DIR / "ira_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {RESULTS_DIR/'ira_summary.json'}")
    print("\n✓ Layer Assignment IRA experiment complete.")


if __name__ == "__main__":
    main()
