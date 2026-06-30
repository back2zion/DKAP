#!/usr/bin/env python3
"""
L2 Ablation Experiment for Wheatley RL-based Job Shop Scheduling.

This script defines and orchestrates a 3-condition ablation study to measure the
impact of domain knowledge structuring (L2) on RL scheduling performance, drawing
an analogy to the 26pp accuracy drop observed when removing L2 from LLM-based
Text-to-SQL systems.

Conditions:
  B1 (No L2 / Vanilla RL)   -- Minimal features, precedence-only graph, no domain heuristics
  B2 (Partial L2 / Basic)   -- Basic domain features, precedence-only graph, standard GNN
  DKAP (Full L2)             -- All domain features, conflict cliques, resource precedence, full GNN

Usage:
  python wheatley_l2_ablation.py [--dry-run] [--device cpu|cuda] [--seeds 42 123 456]

NOTE: Training requires a GPU and substantial compute time. Use --dry-run to
      verify configuration without launching training.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WHEATLEY_ROOT = Path(__file__).resolve().parent.parent / "external_systems" / "wheatley"
INSTANCES_DIR = WHEATLEY_ROOT / "instances" / "taillard"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "wheatley_l2_ablation"
TRAIN_SCRIPT = WHEATLEY_ROOT / "jssp" / "train.py"

# ---------------------------------------------------------------------------
# Taillard benchmark instances and known optimal makespans
# ---------------------------------------------------------------------------
# Taillard instances are grouped by size (jobs x machines).
# ta01-ta10: 15x15,  ta11-ta20: 20x15,  ta21-ta30: 20x20
# We select representative instances from each group.
BENCHMARK_INSTANCES = {
    # ---- 15x15 (small) ----
    "ta01": {"n_j": 15, "n_m": 15, "optimal": 1231},
    "ta02": {"n_j": 15, "n_m": 15, "optimal": 1244},
    "ta03": {"n_j": 15, "n_m": 15, "optimal": 1218},
    # ---- 20x15 (medium) ----
    "ta11": {"n_j": 20, "n_m": 15, "optimal": 1361},
    "ta12": {"n_j": 20, "n_m": 15, "optimal": 1367},
    "ta13": {"n_j": 20, "n_m": 15, "optimal": 1342},
    # ---- 20x20 (large) ----
    "ta21": {"n_j": 20, "n_m": 20, "optimal": 1644},
    "ta22": {"n_j": 20, "n_m": 20, "optimal": 1600},
    "ta23": {"n_j": 20, "n_m": 20, "optimal": 1557},
}

# ---------------------------------------------------------------------------
# Training budget -- same for all conditions to ensure fair comparison
# ---------------------------------------------------------------------------
TOTAL_TIMESTEPS = 1_000_000   # 1M steps (adjust upward for larger instances)
N_EPOCHS = 10
BATCH_SIZE = 128
N_STEPS_EPISODE = 1024
LR = 2e-4
N_VALIDATION_ENV = 10
VALIDATION_FREQ = 10
N_WORKERS = 4

# GNN architecture held constant across conditions
GNN_LAYERS = 6
GNN_HIDDEN = 64
GNN_MLP_LAYERS = 3
GNN_HEADS = 4
GCONV_TYPE = "gatv2"


# ---------------------------------------------------------------------------
# Experiment condition definitions
# ---------------------------------------------------------------------------
@dataclass
class ExperimentCondition:
    """Specifies a single ablation condition via Wheatley CLI args."""
    name: str
    description: str
    # Feature list (passed to --features)
    features: List[str]
    # Conflict encoding: "clique" | "att" | "node"
    conflicts: str
    # Resource precedence edges: "all" | "frontier" | "frontier_strict" | "none"
    add_rp_edges: str
    # Whether to disable explicit TCT computation
    no_tct: bool
    # Whether to add machine_id in edge type
    mid_in_edges: bool
    # Whether to normalize input
    normalize_input: bool
    # Additional CLI flags as a dict
    extra_args: Dict[str, str] = field(default_factory=dict)

    def to_cli_args(
        self,
        instance_name: str,
        instance_path: str,
        n_j: int,
        n_m: int,
        seed: int,
        output_path: str,
        device: str = "cuda",
    ) -> List[str]:
        """Build the full CLI argument list for jssp/train.py."""
        args = [
            sys.executable, str(TRAIN_SCRIPT),
            "--load_problem", instance_path,
            "--n_j", str(n_j),
            "--n_m", str(n_m),
            "--max_n_j", str(n_j),
            "--max_n_m", str(n_m),
            "--seed", str(seed),
            "--device", device,
            "--total_timesteps", str(TOTAL_TIMESTEPS),
            "--n_epochs", str(N_EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--n_steps_episode", str(N_STEPS_EPISODE),
            "--lr", str(LR),
            "--n_validation_env", str(N_VALIDATION_ENV),
            "--validation_freq", str(VALIDATION_FREQ),
            "--n_workers", str(N_WORKERS),
            # GNN architecture (held constant)
            "--gconv_type", GCONV_TYPE,
            "--n_layers_features_extractor", str(GNN_LAYERS),
            "--hidden_dim_features_extractor", str(GNN_HIDDEN),
            "--n_mlp_layers_features_extractor", str(GNN_MLP_LAYERS),
            "--n_attention_heads", str(GNN_HEADS),
            "--graph_pooling", "learn",
            # Condition-specific settings
            "--features", *self.features,
            "--conflicts", self.conflicts,
            "--add_rp_edges", self.add_rp_edges,
            # Output
            "--path", output_path,
            "--fixed_problem",
            "--fixed_validation",
            "--first_machine_id_is_one",
            "--transition_model_config", "simple",
            "--reward_model_config", "Sparse",
            "--duration_type", "deterministic",
            "--criterion", "makespan",
            "--residual_gnn",
            "--normalize_gnn",
            "--disable_visdom",
        ]

        if self.no_tct:
            args.append("--no_tct")

        if self.mid_in_edges:
            args.append("--mid_in_edges")

        if not self.normalize_input:
            args.append("--dont_normalize_input")

        if self.conflicts == "clique":
            args.append("--precompute_cliques")

        # Custom heuristic baselines for comparison
        args.extend(["--custom_heuristic_names", "SPT", "MWKR", "MOPNR"])

        for k, v in self.extra_args.items():
            args.extend([k, v])

        return args


# ---- B1: No L2 (Vanilla RL) ----
# Strip all domain knowledge. Only keep the bare minimum features that the
# environment requires: duration (processing time of each operation) plus the
# mandatory is_affected, selectable, and machine_id that are always present.
# No conflict clique edges (use attention-based conflict which adds no extra
# edges). No resource precedence edges. Disable explicit TCT bounds.
B1_NO_L2 = ExperimentCondition(
    name="B1_no_L2",
    description=(
        "Vanilla RL baseline with minimal domain knowledge. "
        "Only basic duration feature; no conflict cliques, no resource precedence "
        "edges, no TCT bounds. Analogous to a raw RL agent without L2 structuring."
    ),
    features=["duration"],   # minimal -- only processing times
    conflicts="att",          # attention-based (no explicit clique edges)
    add_rp_edges="none",      # no resource precedence edges
    no_tct=True,              # disable task completion time bounds
    mid_in_edges=False,       # no machine_id in edges
    normalize_input=True,
)

# ---- B2: Partial L2 (Basic Domain) ----
# Include some domain-informed features (duration, mopnr, mwkr) but omit
# advanced graph structure. Use precedence edges only (no conflict cliques,
# no resource precedence). Keep TCT but do not embed machine_id in edges.
B2_PARTIAL_L2 = ExperimentCondition(
    name="B2_partial_L2",
    description=(
        "Partial domain knowledge. Includes duration, MOPNR, and MWKR features "
        "but uses only precedence edges (no conflict cliques, no resource "
        "precedence edges). TCT is computed. Machine_id NOT in edge types."
    ),
    features=["duration", "mopnr", "mwkr"],
    conflicts="att",          # no explicit clique edges
    add_rp_edges="none",      # no resource precedence edges
    no_tct=False,             # keep TCT
    mid_in_edges=False,
    normalize_input=True,
)

# ---- DKAP: Full L2 (Full Domain Knowledge-Aligned Pipeline) ----
# All domain features, full graph structure with conflict cliques and
# resource precedence edges, machine_id embedded in edge types.
DKAP_FULL_L2 = ExperimentCondition(
    name="DKAP_full_L2",
    description=(
        "Full domain knowledge-aligned pipeline. All features including duration, "
        "MOPNR, MWKR, job/machine completion percentages, total job/machine times. "
        "Conflict cliques, resource precedence edges (frontier_strict), machine_id "
        "in edge types. Complete L2 structuring."
    ),
    features=[
        "duration",
        "mopnr",
        "mwkr",
        "total_job_time",
        "total_machine_time",
        "job_completion_percentage",
        "machine_completion_percentage",
    ],
    conflicts="clique",               # full conflict clique edges
    add_rp_edges="frontier_strict",   # resource precedence edges
    no_tct=False,                     # keep TCT
    mid_in_edges=True,                # machine_id in edge types
    normalize_input=True,
)

CONDITIONS = [B1_NO_L2, B2_PARTIAL_L2, DKAP_FULL_L2]


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_single_experiment(
    condition: ExperimentCondition,
    instance_name: str,
    seed: int,
    device: str = "cuda",
    dry_run: bool = False,
) -> Dict:
    """Run a single (condition, instance, seed) experiment and return results."""
    inst = BENCHMARK_INSTANCES[instance_name]
    instance_path = str(INSTANCES_DIR / f"{instance_name}.txt")

    run_id = f"{condition.name}__{instance_name}__seed{seed}"
    output_path = str(RESULTS_DIR / run_id)
    os.makedirs(output_path, exist_ok=True)

    cli_args = condition.to_cli_args(
        instance_name=instance_name,
        instance_path=instance_path,
        n_j=inst["n_j"],
        n_m=inst["n_m"],
        seed=seed,
        output_path=output_path,
        device=device,
    )

    result = {
        "run_id": run_id,
        "condition": condition.name,
        "instance": instance_name,
        "n_j": inst["n_j"],
        "n_m": inst["n_m"],
        "optimal_makespan": inst["optimal"],
        "seed": seed,
        "cli_command": " ".join(cli_args),
        "status": "pending",
        "makespan": None,
        "optimality_gap_pct": None,
        "training_time_s": None,
        "timestamp": datetime.now().isoformat(),
    }

    if dry_run:
        result["status"] = "dry_run"
        print(f"[DRY RUN] {run_id}")
        print(f"  Command: {' '.join(cli_args[:10])} ... ({len(cli_args)} args total)")
        print(f"  Features: {condition.features}")
        print(f"  Conflicts: {condition.conflicts}, RP edges: {condition.add_rp_edges}")
        print(f"  no_tct: {condition.no_tct}, mid_in_edges: {condition.mid_in_edges}")
        print()
        return result

    print(f"[RUNNING] {run_id}")
    t0 = time.time()
    try:
        proc = subprocess.run(
            cli_args,
            capture_output=True,
            text=True,
            cwd=str(WHEATLEY_ROOT),
            timeout=3600 * 12,  # 12-hour timeout per run
        )
        elapsed = time.time() - t0
        result["training_time_s"] = elapsed

        # Save stdout/stderr for debugging
        with open(os.path.join(output_path, "stdout.log"), "w") as f:
            f.write(proc.stdout)
        with open(os.path.join(output_path, "stderr.log"), "w") as f:
            f.write(proc.stderr)

        if proc.returncode != 0:
            result["status"] = "error"
            result["error"] = proc.stderr[-2000:] if proc.stderr else "unknown error"
            print(f"  [ERROR] Return code {proc.returncode}")
        else:
            result["status"] = "completed"
            # Parse makespan from training output
            makespan = _parse_best_makespan(proc.stdout, output_path)
            if makespan is not None:
                result["makespan"] = makespan
                result["optimality_gap_pct"] = round(
                    100.0 * (makespan - inst["optimal"]) / inst["optimal"], 2
                )
                print(f"  Makespan: {makespan} (optimal: {inst['optimal']}, "
                      f"gap: {result['optimality_gap_pct']}%)")
            else:
                print(f"  [WARNING] Could not parse makespan from output")

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        print(f"  [TIMEOUT] after 12 hours")
    except Exception as e:
        result["status"] = "exception"
        result["error"] = str(e)
        print(f"  [EXCEPTION] {e}")

    return result


def _parse_best_makespan(stdout: str, output_path: str) -> Optional[float]:
    """Extract the best makespan achieved during training from stdout or saved model.

    Wheatley logs validation metrics during training. We look for the best
    makespan reported. Fallback: check if a result file was written.
    """
    import re
    best = None
    # Look for patterns like "best makespan: 1234" or "rl_makespan=1234"
    for pattern in [
        r"best.*?makespan.*?(\d+\.?\d*)",
        r"rl_makespan.*?(\d+\.?\d*)",
        r"ratio.*?(\d+\.\d+)",
        r"Makespan.*?(\d+\.?\d*)",
    ]:
        matches = re.findall(pattern, stdout, re.IGNORECASE)
        if matches:
            candidates = [float(m) for m in matches]
            # For ratio, smaller is better (RL/OR-Tools); for makespan, also smaller
            candidate = min(candidates)
            if best is None or candidate < best:
                best = candidate

    return best


def compute_summary(all_results: List[Dict]) -> Dict:
    """Aggregate results by condition and instance size for comparison."""
    from collections import defaultdict

    by_condition = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if r["status"] == "completed" and r["optimality_gap_pct"] is not None:
            key = f"{r['n_j']}x{r['n_m']}"
            by_condition[r["condition"]][key].append(r["optimality_gap_pct"])

    summary = {}
    for cond_name, size_gaps in by_condition.items():
        cond_summary = {}
        all_gaps = []
        for size, gaps in size_gaps.items():
            import statistics
            avg_gap = statistics.mean(gaps)
            cond_summary[size] = {
                "mean_gap_pct": round(avg_gap, 2),
                "std_gap_pct": round(statistics.stdev(gaps), 2) if len(gaps) > 1 else 0.0,
                "n_runs": len(gaps),
                "min_gap_pct": round(min(gaps), 2),
                "max_gap_pct": round(max(gaps), 2),
            }
            all_gaps.extend(gaps)

        if all_gaps:
            cond_summary["overall"] = {
                "mean_gap_pct": round(statistics.mean(all_gaps), 2),
                "std_gap_pct": round(statistics.stdev(all_gaps), 2) if len(all_gaps) > 1 else 0.0,
                "n_runs": len(all_gaps),
            }
        summary[cond_name] = cond_summary

    # Compute L2 impact: gap difference between B1 and DKAP
    if "B1_no_L2" in summary and "DKAP_full_L2" in summary:
        b1_overall = summary["B1_no_L2"].get("overall", {}).get("mean_gap_pct")
        dkap_overall = summary["DKAP_full_L2"].get("overall", {}).get("mean_gap_pct")
        if b1_overall is not None and dkap_overall is not None:
            summary["l2_impact_pp"] = round(b1_overall - dkap_overall, 2)
            summary["l2_impact_description"] = (
                f"Removing L2 (B1 vs DKAP) increases optimality gap by "
                f"{summary['l2_impact_pp']} percentage points"
            )

    return summary


def print_comparison_table(summary: Dict):
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 80)
    print("L2 ABLATION RESULTS -- Wheatley JSSP")
    print("=" * 80)
    print(f"{'Condition':<20} {'Size':<10} {'Mean Gap%':<12} {'Std':<10} {'N':<5}")
    print("-" * 60)

    for cond in ["B1_no_L2", "B2_partial_L2", "DKAP_full_L2"]:
        if cond not in summary:
            continue
        for size in sorted(summary[cond].keys()):
            if size == "overall":
                continue
            s = summary[cond][size]
            print(f"{cond:<20} {size:<10} {s['mean_gap_pct']:<12.2f} "
                  f"{s['std_gap_pct']:<10.2f} {s['n_runs']:<5}")
        if "overall" in summary[cond]:
            o = summary[cond]["overall"]
            print(f"{cond:<20} {'OVERALL':<10} {o['mean_gap_pct']:<12.2f} "
                  f"{o['std_gap_pct']:<10.2f} {o['n_runs']:<5}")
        print("-" * 60)

    if "l2_impact_pp" in summary:
        print(f"\nL2 Impact: {summary['l2_impact_pp']} pp "
              f"(B1_no_L2 gap - DKAP_full_L2 gap)")
        print(summary.get("l2_impact_description", ""))
    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI command generation for reference / manual execution
# ---------------------------------------------------------------------------
def print_all_commands(device: str = "cuda", seeds: List[int] = None):
    """Print all CLI commands for manual execution."""
    if seeds is None:
        seeds = [42]

    print("=" * 80)
    print("WHEATLEY L2 ABLATION -- CLI Commands")
    print("=" * 80)

    for condition in CONDITIONS:
        print(f"\n{'#' * 70}")
        print(f"# CONDITION: {condition.name}")
        print(f"# {condition.description}")
        print(f"{'#' * 70}")
        for inst_name, inst_info in BENCHMARK_INSTANCES.items():
            for seed in seeds:
                run_id = f"{condition.name}__{inst_name}__seed{seed}"
                output_path = str(RESULTS_DIR / run_id)
                cli_args = condition.to_cli_args(
                    instance_name=inst_name,
                    instance_path=str(INSTANCES_DIR / f"{inst_name}.txt"),
                    n_j=inst_info["n_j"],
                    n_m=inst_info["n_m"],
                    seed=seed,
                    output_path=output_path,
                    device=device,
                )
                print(f"\n# {run_id}")
                print(" ".join(cli_args))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="L2 Ablation Experiment for Wheatley JSSP"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print configurations without running training"
    )
    parser.add_argument(
        "--print-commands", action="store_true",
        help="Print all CLI commands for manual execution and exit"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cpu", "cuda", "cuda:0", "cuda:1"],
        help="Device for training"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456],
        help="Random seeds for repeated runs"
    )
    parser.add_argument(
        "--conditions", type=str, nargs="+",
        default=["B1_no_L2", "B2_partial_L2", "DKAP_full_L2"],
        choices=["B1_no_L2", "B2_partial_L2", "DKAP_full_L2"],
        help="Which conditions to run"
    )
    parser.add_argument(
        "--instances", type=str, nargs="+",
        default=list(BENCHMARK_INSTANCES.keys()),
        help="Which Taillard instances to benchmark"
    )
    args = parser.parse_args()

    if args.print_commands:
        print_all_commands(device=args.device, seeds=args.seeds)
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    cond_map = {c.name: c for c in CONDITIONS}
    selected_conditions = [cond_map[n] for n in args.conditions]

    # Log experiment configuration
    config = {
        "timestamp": datetime.now().isoformat(),
        "conditions": [asdict(c) for c in selected_conditions],
        "instances": {k: v for k, v in BENCHMARK_INSTANCES.items() if k in args.instances},
        "seeds": args.seeds,
        "training_budget": {
            "total_timesteps": TOTAL_TIMESTEPS,
            "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
        },
        "gnn_architecture": {
            "gconv_type": GCONV_TYPE,
            "n_layers": GNN_LAYERS,
            "hidden_dim": GNN_HIDDEN,
            "n_mlp_layers": GNN_MLP_LAYERS,
            "n_attention_heads": GNN_HEADS,
        },
    }
    config_path = RESULTS_DIR / "experiment_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Experiment config saved to {config_path}")

    # Run all experiments
    all_results = []
    total_runs = len(selected_conditions) * len(args.instances) * len(args.seeds)
    run_count = 0

    for condition in selected_conditions:
        for inst_name in args.instances:
            if inst_name not in BENCHMARK_INSTANCES:
                print(f"[SKIP] Unknown instance: {inst_name}")
                continue
            for seed in args.seeds:
                run_count += 1
                print(f"\n[{run_count}/{total_runs}] ", end="")
                result = run_single_experiment(
                    condition=condition,
                    instance_name=inst_name,
                    seed=seed,
                    device=args.device,
                    dry_run=args.dry_run,
                )
                all_results.append(result)

                # Save incremental results
                results_path = RESULTS_DIR / "all_results.json"
                with open(results_path, "w") as f:
                    json.dump(all_results, f, indent=2)

    # Compute and save summary
    if not args.dry_run:
        summary = compute_summary(all_results)
        summary_path = RESULTS_DIR / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_path}")
        print_comparison_table(summary)
    else:
        print(f"\n[DRY RUN] {len(all_results)} runs configured.")
        print(f"Results would be saved to {RESULTS_DIR}")

    # Also save the raw results
    results_path = RESULTS_DIR / "all_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
