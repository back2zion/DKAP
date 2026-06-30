#!/bin/bash
# Wheatley L2 Ablation - Final self-contained script
# Runs 3 conditions x 3 instances = 9 experiments on GPU
set -euo pipefail

WHEATLEY="/home/woodcross/projects/thesis/external_systems/wheatley"
RESULTS="/home/woodcross/projects/thesis/experiments/results/wheatley_l2_ablation"
PY="$WHEATLEY/.venv/bin/python"
export DGLBACKEND=pytorch
cd "$WHEATLEY"

# ===== SELF-TEST =====
echo "=== Self-test ==="
$PY -c "
import torch, dgl
assert torch.cuda.is_available(), 'CUDA not available'
g = dgl.graph(([0],[1])).to('cuda')
print(f'torch {torch.__version__}, DGL {dgl.__version__}+CUDA OK')
from jssp.models.agent import Agent
from jssp.dispatching_rules.solver import Solver
from jssp.utils.utils import load_problem
print('All imports OK')
" || { echo "SELF-TEST FAILED"; exit 1; }

mkdir -p "$RESULTS"

# ===== COMMON ARGS =====
COMMON="--gconv_type gatv2 --n_layers_features_extractor 4 --hidden_dim_features_extractor 32 --n_attention_heads 4 --graph_pooling learn --residual_gnn --normalize_gnn --n_j 15 --n_m 15 --total_timesteps 50000 --n_steps_episode 225 --batch_size 128 --n_epochs 5 --lr 0.0002 --n_workers 1 --n_validation_env 3 --validation_freq 10 --seed 42 --device cuda --transition_model_config simple --reward_model_config Sparse --duration_type deterministic --criterion makespan --fixed_problem --fixed_validation --first_machine_id_is_one --disable_visdom --ortools_strategy averagistic --max_time_ortools 10 --vecenv_type dummy --custom_heuristic_names SPT MWKR MOPNR"

# ===== CONDITIONS =====
declare -A COND_ARGS
COND_ARGS[B1]="--features duration --conflicts att --add_rp_edges none --no_tct"
COND_ARGS[B2]="--features duration mopnr mwkr --conflicts att --add_rp_edges none"
COND_ARGS[DKAP]="--features duration mopnr mwkr total_job_time total_machine_time job_completion_percentage machine_completion_percentage --conflicts clique --add_rp_edges frontier_strict --mid_in_edges --precompute_cliques"

# ===== RUN =====
for inst in ta01 ta02 ta03; do
    for cond in B1 B2 DKAP; do
        NAME="${cond}_${inst}"
        OUT="$RESULTS/$NAME"
        rm -rf "$OUT"
        echo ""
        echo "========================================"
        echo "[$(date +%H:%M:%S)] $NAME"
        echo "========================================"

        $PY -m jssp.train \
            --load_problem "instances/taillard/${inst}.txt" \
            $COMMON ${COND_ARGS[$cond]} \
            --path "$OUT" \
            2>&1 | tee "$RESULTS/${NAME}.log"

        echo "[$(date +%H:%M:%S)] $NAME DONE"
    done
done

# ===== COLLECT RESULTS =====
echo ""
echo "========================================"
echo "COLLECTING RESULTS"
echo "========================================"

$PY << 'PYEOF'
import json, os, re, glob
import numpy as np

base = "/home/woodcross/projects/thesis/experiments/results/wheatley_l2_ablation"
optima = {"ta01": 1231, "ta02": 1244, "ta03": 1218}
results = {}

for logf in sorted(glob.glob(f"{base}/*.log")):
    name = os.path.basename(logf).replace('.log','')
    cond, inst = name.split('_', 1)

    with open(logf) as f:
        log = f.read()

    # Extract all mean_criterion values (validation makespans)
    crits = re.findall(r'mean_criterion=\s*(\d+\.?\d*)', log)
    ratios = re.findall(r'Current ratio\s*:\s*(\d+\.?\d*)', log)

    best_ms = float(min(crits, key=float)) if crits else None
    best_ratio = float(min(ratios, key=float)) if ratios else None
    opt = optima.get(inst, None)
    gap = ((best_ms - opt) / opt * 100) if (best_ms and opt) else None

    results[name] = {
        "condition": cond, "instance": inst,
        "best_makespan": best_ms,
        "best_ratio": best_ratio,
        "optimal": opt,
        "gap_pct": round(gap, 1) if gap else None,
        "n_validations": len(crits),
    }
    print(f"  {name:15s}: makespan={best_ms}, ratio={best_ratio}, gap={gap and f'{gap:.1f}%'}")

# Summary by condition
print("\n===== CONDITION AVERAGES =====")
from collections import defaultdict
by_cond = defaultdict(list)
for r in results.values():
    if r['gap_pct'] is not None:
        by_cond[r['condition']].append(r['gap_pct'])

for cond in ['B1', 'B2', 'DKAP']:
    gaps = by_cond.get(cond, [])
    if gaps:
        avg = np.mean(gaps)
        print(f"  {cond:6s}: avg_gap = {avg:.1f}% (n={len(gaps)})")

with open(f"{base}/gnn_results.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {base}/gnn_results.json")
PYEOF
