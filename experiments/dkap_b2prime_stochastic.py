#!/usr/bin/env python3
"""
DKAP Extended Experiments: B2' (Unstructured Equal-Token) + Stochastic Robustness
==================================================================================
Two experiments in one script:

(a) B2' Condition: Same domain information as DKAP L2 artifacts, but written as
    unstructured prose (~same token count). Compares B2 vs B2' vs DKAP to isolate
    the STRUCTURING effect from the INFORMATION effect.

(c) Stochastic Robustness: Run B2/B2'/DKAP at temperature=0.3 × 5 repeats,
    compute mean±std, compare with temperature=0.0 deterministic baseline.

Key improvements over original dkap_experiment.py:
  - JSONL incremental saving (no data loss on crash/OOM)
  - Resume support (skip already-completed queries)
  - Memory-efficient: results flushed to disk per query
  - GPU-safe: configurable batch delay to prevent vLLM KV cache OOM

Author: Dooil Kwak
Date: 2026-04-03
"""

import json
import time
import sqlite3
import os
import sys
import re
import logging
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import requests
except ImportError:
    os.system("pip install requests --break-system-packages -q")
    import requests

# ============================================================
# Configuration
# ============================================================
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "default-model")
MAX_TOKENS = 2048
RESULTS_DIR = Path("results/b2prime_stochastic")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path("/tmp/finance_benchmark.db")

# Delay between API calls (seconds) to prevent vLLM KV cache pressure
# Increase if you see CUDA OOM on the vLLM server
INTER_QUERY_DELAY = float(os.environ.get("INTER_QUERY_DELAY", "0.5"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("dkap_b2prime_stochastic.log"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. Import schema, seed data, benchmark from original experiment
# ============================================================
# We import these by reading the original file to avoid duplication.
# If import fails, inline definitions are provided as fallback.

sys.path.insert(0, str(Path(__file__).parent))
try:
    from dkap_experiment import (
        SCHEMA_DDL,
        SEED_DATA_SQL,
        BENCHMARK,
        PROMPT_B1,
        PROMPT_B2,
        PROMPT_DKAP,
        get_b2_chunks,
        setup_database,
        execute_sql,
        normalize_sql,
        evaluate_ex,
        evaluate_em,
    )
    logger.info("Successfully imported from dkap_experiment.py")
except ImportError as e:
    logger.error(f"Failed to import dkap_experiment: {e}")
    logger.error("Please ensure dkap_experiment.py is in the same directory.")
    sys.exit(1)


# ============================================================
# 2. B2' Prompt: Unstructured Prose with IDENTICAL Information
# ============================================================
# Target: ~1,900 tokens (matching DKAP L2 artifact token count)
# Contains ALL the same facts as DKAP prompt but in narrative form
# — no tables, no structured headers, no domain annotations

PROMPT_B2_PRIME = """You are a SQL expert. Convert the following Korean natural language question into a valid SQLite SQL query.

The database you're working with is a Korean financial card system. Here's everything you need to know about it:

The first table is called ods_card_member and it stores card member information. It has these columns: 기준년월 which is a TEXT type and represents the data reference year-month in YYYYMM format and is not nullable, then 발급회원번호 which is a TEXT type serving as the primary key and represents the unique card member identification number, 고객명 is a TEXT column for the customer's full name, VIP등급코드 is TEXT and indicates the VIP grade where 01 means VVIP and 02 means VIP and 03 means Gold and 04 means 일반 which is general/regular, 입회일자 is TEXT formatted as YYYYMMDD showing when the member joined, 최종이용일자 is TEXT showing the last card usage date, 신용등급 is an INTEGER representing credit rating from 1 to 10 where 1 is the best rating according to Korean Financial Supervisory Service standards, 남녀구분코드 is TEXT where M means male and F means female, and 연령대코드 is TEXT representing age bracket with values like 20 30 40 50 60.

The second table is ods_card_transaction for card transaction records. It contains 거래일련번호 as an INTEGER PRIMARY KEY with AUTOINCREMENT for unique transaction numbers, 기준년월 as TEXT NOT NULL for the reference year-month, 발급회원번호 as TEXT NOT NULL which is a foreign key referencing ods_card_member, 승인번호 as TEXT for the approval number, 가맹점명 as TEXT for merchant name, 업종코드 as TEXT for business category codes, 이용금액 as INTEGER representing the transaction amount in Korean Won, 이용일자 as TEXT in YYYYMMDD format for the transaction date, and 할부개월수 as INTEGER defaulting to 0 where 0 means lump sum payment and any positive number means installment months.

The third table is ods_card_credit for credit information with 기준년월 as TEXT NOT NULL and part of the primary key, 발급회원번호 as TEXT NOT NULL also part of the primary key and referencing ods_card_member, 카드이용한도금액 as INTEGER for the credit limit in Won, 잔액 as INTEGER for current balance in Won, 연체일수 as INTEGER defaulting to 0 where 0 means normal and positive means days overdue, 연체잔액 as INTEGER defaulting to 0 for overdue balance in Won, and 한도소진율 as REAL representing the credit utilization ratio from 0.0 to 1.0 where higher is riskier.

The fourth table is ods_card_billing for billing information containing 기준년월 TEXT NOT NULL as part of primary key, 발급회원번호 TEXT NOT NULL also primary key and foreign key to ods_card_member, 청구금액 INTEGER for billed amount in Won, 결제일 TEXT for payment due date in DD format, and 납부상태코드 TEXT where 01 means fully paid and 02 means unpaid and 03 means partially paid.

The fifth table is ods_fin_product for financial product information with 상품코드 TEXT as PRIMARY KEY for product code, 상품명 TEXT for product name, 상품유형 TEXT for product type which can be 카드 or 대출 or 펀드 or 보험, 기준금리 REAL for base interest rate as percentage, and 가입건수 INTEGER defaulting to 0 for cumulative subscription count.

About the domain terminology: 기준년월 is the partition key in YYYYMM format. 발급회원번호 is the universal join key across all tables. For VIP grades, lower numbers are higher ranks. Credit ratings go from 1 being best to 10 being worst. 한도소진율 is calculated as balance divided by credit limit and ranges from 0 to 1 with higher being more risky. Payment status codes are 01 for paid 02 for unpaid 03 for partial. Transaction amounts are always in Korean Won as integers. Date columns like 입회일자 and 이용일자 use YYYYMMDD string format so string comparison works for ordering. Grade and code columns look like numbers but are TEXT type so use string comparison. Ratio columns like 한도소진율 and 기준금리 are REAL type in 0.0 to 1.0 range.

For joining tables: ods_card_member connects to ods_card_transaction through 발급회원번호 AND 기준년월. Similarly ods_card_member connects to ods_card_credit through 발급회원번호 AND 기준년월. And ods_card_member connects to ods_card_billing through 발급회원번호 AND 기준년월. The ods_fin_product table is independent and can be joined via 상품코드 but currently has no foreign key relationship.

Now let me elaborate on some important details about how to work with this data correctly. When you need to count distinct members you should use COUNT(DISTINCT 발급회원번호) because the same member can appear in multiple 기준년월 periods. The VIP등급코드 values are stored as strings so you should compare them using string comparison like WHERE VIP등급코드='01' for VVIP members. Similarly the 납부상태코드 uses string values 01 02 and 03, not integers, so always use quotes when comparing. The 신용등급 column however is an INTEGER type so you can use numeric comparisons and ordering directly, remembering that grade 1 is the best and grade 10 is the worst. For transaction analysis, the 이용금액 column contains the total transaction amount in Korean Won and is stored as INTEGER so you can aggregate it directly with SUM or AVG functions. When analyzing card utilization, the 한도소진율 in the credit table represents what fraction of the credit limit has been used, calculated as 잔액 divided by 카드이용한도금액, and a value closer to 1.0 indicates higher credit utilization which is generally considered higher risk. For date-based filtering, since all date columns are stored as TEXT in YYYYMMDD format, you can use standard string comparison operators for date range queries, for example WHERE 이용일자 >= '20250101' AND 이용일자 <= '20250131' to get January 2025 transactions. The 할부개월수 column in the transaction table indicates installment months where 0 means the purchase was made as a lump sum payment and any positive integer indicates the number of installment months. When working with the billing table, the 결제일 column contains only the day portion as a two-digit string like 05 or 15 or 25, representing the regular monthly payment date. The 청구금액 represents the total billed amount for that billing period in Won. For the financial product table, 기준금리 is stored as a percentage value so 4.5 means 4.5 percent interest rate, and 가입건수 tracks the cumulative number of subscriptions for each product. When performing joins across tables, always include both 발급회원번호 and 기준년월 in the JOIN condition to ensure you are matching records from the same time period, as member data can change between periods. The 연체일수 in the credit table starts at 0 for members in good standing and increases by one for each day of payment delinquency, while 연체잔액 shows the outstanding overdue amount. For aggregation queries involving multiple tables, it is recommended to first join the tables and then apply GROUP BY, rather than using subqueries, to maintain query readability and performance.

Output ONLY the SQL query, nothing else. Do not include explanation.

Question: {question}

SQL:"""


# ============================================================
# 3. Experiment Configuration
# ============================================================
@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    name: str
    conditions: List[str]
    temperature: float
    num_runs: int
    output_file: str


EXPERIMENT_CONFIGS = {
    # (a) B2' comparison at temperature=0.0
    "b2prime_deterministic": ExperimentConfig(
        name="B2' Information Control (temp=0.0)",
        conditions=["B2", "B2_PRIME", "DKAP"],
        temperature=0.0,
        num_runs=3,
        output_file="b2prime_deterministic.jsonl",
    ),
    # (c) Stochastic robustness at temperature=0.3
    "stochastic_t03": ExperimentConfig(
        name="Stochastic Robustness (temp=0.3)",
        conditions=["B2", "B2_PRIME", "DKAP"],
        temperature=0.3,
        num_runs=5,
        output_file="stochastic_t03.jsonl",
    ),
}


# ============================================================
# 4. LLM Inference (with OOM-safe settings)
# ============================================================
def call_vllm(prompt: str, temperature: float = 0.0, max_retries: int = 3) -> str:
    """
    Call vLLM API with OOM-safe settings.

    OOM prevention:
      - max_tokens capped at 2048 (not 4096)
      - Single request at a time (no batching)
      - Inter-query delay configured via INTER_QUERY_DELAY
      - Timeout at 120s to prevent hung connections
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
        "stop": ["Question:", "---"],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    # For stochastic runs, add a seed for reproducibility tracking
    # (vLLM supports seed parameter)
    if temperature > 0:
        payload["seed"] = int(time.time() * 1000) % (2**31)

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{VLLM_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            # Remove Qwen3 <think>...</think> blocks
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            content = re.sub(r"<think>.*$", "", content, flags=re.DOTALL).strip()
            # Clean markdown fences
            content = re.sub(r"^```sql\s*", "", content)
            content = re.sub(r"^```\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            content = content.strip()
            content = " ".join(content.split())
            content = content.rstrip(";").strip()
            return content
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt+1}/{max_retries}")
            # On timeout, wait longer — vLLM might be under memory pressure
            time.sleep(5 * (attempt + 1))
        except requests.exceptions.ConnectionError:
            logger.error("Connection error — is vLLM server running?")
            time.sleep(10)
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                logger.error("All retries failed")
                return ""
    return ""


# ============================================================
# 5. Prompt Dispatch
# ============================================================
def get_prompt(condition: str, question: str) -> str:
    """Get the prompt for a given condition and question."""
    if condition == "B1":
        return PROMPT_B1.format(question=question)
    elif condition == "B2":
        return PROMPT_B2.format(question=question, retrieved_chunks=get_b2_chunks(question))
    elif condition == "B2_PRIME":
        return PROMPT_B2_PRIME.format(question=question)
    elif condition == "DKAP":
        return PROMPT_DKAP.format(question=question)
    else:
        raise ValueError(f"Unknown condition: {condition}")


# ============================================================
# 6. Incremental JSONL Writer
# ============================================================
class IncrementalWriter:
    """
    Writes results to JSONL file incrementally.
    Each line is a complete JSON object — safe against crashes.
    Also tracks completed (condition, run_id, question_id) tuples for resume.
    """

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.completed: set = set()
        self._load_existing()

    def _load_existing(self):
        """Load already-completed entries for resume support."""
        if self.filepath.exists():
            with open(self.filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        key = (entry["condition"], entry["run_id"], entry["question_id"])
                        self.completed.add(key)
                    except (json.JSONDecodeError, KeyError):
                        continue
            logger.info(f"Resumed: {len(self.completed)} entries from {self.filepath}")

    def is_done(self, condition: str, run_id: int, question_id: str) -> bool:
        return (condition, run_id, question_id) in self.completed

    def write(self, result: dict):
        """Append a single result to the JSONL file (atomic per line)."""
        with open(self.filepath, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        key = (result["condition"], result["run_id"], result["question_id"])
        self.completed.add(key)

    def read_all(self) -> List[dict]:
        """Read all results back."""
        results = []
        if self.filepath.exists():
            with open(self.filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return results


# ============================================================
# 7. Run Experiment
# ============================================================
def run_experiment(config: ExperimentConfig):
    """Run a single experiment configuration with incremental saving."""
    output_path = RESULTS_DIR / config.output_file
    writer = IncrementalWriter(output_path)

    logger.info("=" * 60)
    logger.info(f"Experiment: {config.name}")
    logger.info(f"Conditions: {config.conditions}")
    logger.info(f"Temperature: {config.temperature}")
    logger.info(f"Runs: {config.num_runs}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)

    # Setup database
    conn = setup_database(DB_PATH)

    total = len(BENCHMARK) * len(config.conditions) * config.num_runs
    skipped = 0
    done = 0

    for run_id in range(1, config.num_runs + 1):
        for cond in config.conditions:
            for item in BENCHMARK:
                done += 1
                qid = item["id"]

                # Resume: skip already-completed
                if writer.is_done(cond, run_id, qid):
                    skipped += 1
                    continue

                prompt = get_prompt(cond, item["nl"])

                start_time = time.time()
                pred_sql = call_vllm(prompt, temperature=config.temperature)
                latency_ms = (time.time() - start_time) * 1000

                # Evaluate
                ex_ok, _ = execute_sql(conn, pred_sql)
                ex_score = evaluate_ex(conn, pred_sql, item["gold_sql"]) if ex_ok else False
                em_score = evaluate_em(pred_sql, item["gold_sql"])

                result = {
                    "experiment": config.name,
                    "condition": cond,
                    "run_id": run_id,
                    "question_id": qid,
                    "difficulty": item["difficulty"],
                    "nl_question": item["nl"],
                    "gold_sql": item["gold_sql"],
                    "pred_sql": pred_sql,
                    "ex_score": ex_score,
                    "em_score": em_score,
                    "execution_ok": ex_ok,
                    "latency_ms": round(latency_ms, 1),
                    "temperature": config.temperature,
                    "timestamp": datetime.now().isoformat(),
                }
                writer.write(result)

                status = "EX" if ex_score else ("EXEC" if ex_ok else "FAIL")
                logger.info(
                    f"[{done}/{total}] R{run_id} {cond:>8s} {qid} ({item['difficulty']}) "
                    f"→ {status} ({latency_ms:.0f}ms)"
                )

                # OOM prevention: small delay between calls
                time.sleep(INTER_QUERY_DELAY)

                # Periodic GC to keep Python memory lean
                if done % 50 == 0:
                    gc.collect()

    conn.close()

    if skipped > 0:
        logger.info(f"Skipped {skipped} already-completed entries (resume)")

    return output_path


# ============================================================
# 8. Analysis
# ============================================================
def analyze_results(filepath: Path, experiment_name: str) -> Dict:
    """Analyze results from a JSONL file."""
    writer = IncrementalWriter(filepath)
    all_results = writer.read_all()

    if not all_results:
        logger.error(f"No results found in {filepath}")
        return {}

    conditions = sorted(set(r["condition"] for r in all_results))
    runs = sorted(set(r["run_id"] for r in all_results))

    summary = {"experiment": experiment_name, "conditions": {}}

    for cond in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        n = len(cond_results)

        ex_total = sum(1 for r in cond_results if r["ex_score"])
        em_total = sum(1 for r in cond_results if r["em_score"])

        # Per-run EX% for mean±std
        run_ex_pcts = []
        for run_id in runs:
            run_results = [r for r in cond_results if r["run_id"] == run_id]
            rn = len(run_results)
            if rn > 0:
                pct = sum(1 for r in run_results if r["ex_score"]) * 100 / rn
                run_ex_pcts.append(pct)

        # Per-difficulty breakdown
        diff_breakdown = {}
        for diff in ["E", "M", "H"]:
            diff_results = [r for r in cond_results if r["difficulty"] == diff]
            dn = len(diff_results)
            if dn > 0:
                diff_ex = sum(1 for r in diff_results if r["ex_score"])

                # Per-run for this difficulty
                diff_run_pcts = []
                for run_id in runs:
                    dr = [r for r in diff_results if r["run_id"] == run_id]
                    if dr:
                        diff_run_pcts.append(sum(1 for r in dr if r["ex_score"]) * 100 / len(dr))

                diff_breakdown[diff] = {
                    "n": dn,
                    "EX": diff_ex,
                    "EX_pct": round(diff_ex * 100 / dn, 1),
                    "mean_EX_pct": round(sum(diff_run_pcts) / len(diff_run_pcts), 1) if diff_run_pcts else 0,
                    "std_EX_pct": round(
                        (sum((x - sum(diff_run_pcts)/len(diff_run_pcts))**2 for x in diff_run_pcts) / len(diff_run_pcts))**0.5, 1
                    ) if len(diff_run_pcts) > 1 else 0,
                }

        import statistics
        mean_ex = statistics.mean(run_ex_pcts) if run_ex_pcts else 0
        std_ex = statistics.stdev(run_ex_pcts) if len(run_ex_pcts) > 1 else 0

        summary["conditions"][cond] = {
            "n_total": n,
            "n_runs": len(runs),
            "EX_overall_pct": round(ex_total * 100 / n, 1) if n > 0 else 0,
            "EM_overall_pct": round(em_total * 100 / n, 1) if n > 0 else 0,
            "mean_EX_pct": round(mean_ex, 1),
            "std_EX_pct": round(std_ex, 1),
            "run_EX_pcts": [round(x, 1) for x in run_ex_pcts],
            "avg_latency_ms": round(
                sum(r["latency_ms"] for r in cond_results) / n, 1
            ) if n > 0 else 0,
            "by_difficulty": diff_breakdown,
        }

    # Comparison: DKAP vs B2 vs B2'
    print("\n" + "=" * 70)
    print(f"ANALYSIS: {experiment_name}")
    print("=" * 70)
    print(f"{'Condition':<12} {'EX% (mean±std)':<20} {'EM%':<10} {'Latency(ms)':<12} {'Runs'}")
    print("-" * 70)
    for cond in conditions:
        s = summary["conditions"][cond]
        print(
            f"{cond:<12} {s['mean_EX_pct']:>5.1f} ± {s['std_EX_pct']:<5.1f}      "
            f"{s['EM_overall_pct']:<10} {s['avg_latency_ms']:<12} "
            f"{s['run_EX_pcts']}"
        )

    print("\nBy Difficulty:")
    for cond in conditions:
        print(f"  {cond}:")
        for diff in ["E", "M", "H"]:
            d = summary["conditions"][cond]["by_difficulty"].get(diff, {})
            if d:
                print(
                    f"    {diff}: {d['mean_EX_pct']:.1f} ± {d['std_EX_pct']:.1f}% "
                    f"({d['EX']}/{d['n']})"
                )

    # Statistical comparison: B2' vs DKAP (structure effect)
    if "DKAP" in summary["conditions"] and "B2_PRIME" in summary["conditions"]:
        dkap_runs = summary["conditions"]["DKAP"]["run_EX_pcts"]
        b2p_runs = summary["conditions"]["B2_PRIME"]["run_EX_pcts"]
        delta = summary["conditions"]["DKAP"]["mean_EX_pct"] - summary["conditions"]["B2_PRIME"]["mean_EX_pct"]
        print(f"\n*** STRUCTURING EFFECT: DKAP - B2' = {delta:+.1f}pp ***")
        if delta > 0:
            print("    → Positive delta confirms that STRUCTURING (not just information) improves performance.")
        else:
            print("    → No structuring effect detected. Information alone may be sufficient.")

    return summary


# ============================================================
# 9. Token Count Verification
# ============================================================
def verify_token_parity():
    """Verify that DKAP and B2' prompts have similar token counts."""
    import unicodedata

    def estimate_tokens(text):
        korean = sum(1 for c in text if unicodedata.category(c).startswith('Lo'))
        ascii_w = len(re.findall(r'[a-zA-Z_]+', text))
        symbols = len(re.findall(r'[|{}\[\]()=<>:,;.!?\-/]', text))
        return int(korean * 1.5 + ascii_w * 1.3 + symbols)

    dkap_prompt = PROMPT_DKAP.format(question="TEST")
    b2p_prompt = PROMPT_B2_PRIME.format(question="TEST")
    b2_prompt = PROMPT_B2.format(question="TEST", retrieved_chunks="TEST CHUNKS")

    dkap_tokens = estimate_tokens(dkap_prompt)
    b2p_tokens = estimate_tokens(b2p_prompt)
    b2_tokens = estimate_tokens(b2_prompt)

    print("\n=== Token Count Verification ===")
    print(f"DKAP prompt:     ~{dkap_tokens} tokens ({len(dkap_prompt)} chars)")
    print(f"B2' prompt:      ~{b2p_tokens} tokens ({len(b2p_prompt)} chars)")
    print(f"B2 prompt (avg): ~{b2_tokens} tokens ({len(b2_prompt)} chars)")
    print(f"B2' / DKAP ratio: {b2p_tokens/dkap_tokens:.2f}")

    if abs(b2p_tokens - dkap_tokens) / dkap_tokens > 0.15:
        logger.warning(
            f"Token count mismatch >15%: DKAP={dkap_tokens}, B2'={b2p_tokens}. "
            "Consider adjusting B2' prompt length."
        )
    else:
        print("✓ Token counts within 15% — parity confirmed.")

    return {"DKAP": dkap_tokens, "B2_PRIME": b2p_tokens, "B2": b2_tokens}


# ============================================================
# 10. Main
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="DKAP B2' + Stochastic Experiments")
    parser.add_argument(
        "--experiment",
        choices=["b2prime", "stochastic", "both", "analyze", "verify_tokens"],
        default="both",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Delay between API calls in seconds (OOM prevention)",
    )
    args = parser.parse_args()

    if args.delay is not None:
        global INTER_QUERY_DELAY
        INTER_QUERY_DELAY = args.delay

    if args.experiment == "verify_tokens":
        verify_token_parity()
        return

    if args.experiment in ("b2prime", "both"):
        # Test vLLM connectivity first
        logger.info("Testing vLLM connectivity...")
        test = call_vllm("SELECT 1", temperature=0.0)
        if not test:
            logger.error("Cannot connect to vLLM server. Start it with:")
            logger.error(
                "  vllm serve <model> --tensor-parallel-size 2 "
                "--max-model-len 4096 --gpu-memory-utilization 0.85"
            )
            sys.exit(1)
        logger.info(f"vLLM OK: {test[:50]}")

        verify_token_parity()

        config = EXPERIMENT_CONFIGS["b2prime_deterministic"]
        path = run_experiment(config)
        summary = analyze_results(path, config.name)
        with open(RESULTS_DIR / "b2prime_summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.experiment in ("stochastic", "both"):
        config = EXPERIMENT_CONFIGS["stochastic_t03"]
        path = run_experiment(config)
        summary = analyze_results(path, config.name)
        with open(RESULTS_DIR / "stochastic_summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.experiment == "analyze":
        # Analyze existing results without running
        for key, config in EXPERIMENT_CONFIGS.items():
            path = RESULTS_DIR / config.output_file
            if path.exists():
                summary = analyze_results(path, config.name)
                with open(RESULTS_DIR / f"{key}_summary.json", "w") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("\nAll experiments complete.")


if __name__ == "__main__":
    main()
