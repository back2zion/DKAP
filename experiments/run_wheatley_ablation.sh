#!/bin/bash
# Wheatley L2 Ablation Experiment
# 3 conditions x 3 instances (ta01-ta03, 15x15) x 1 seed = 9 runs
set -e
cd /home/woodcross/projects/thesis/external_systems/wheatley
export DGLBACKEND=pytorch

RESULTS="/home/woodcross/projects/thesis/experiments/results/wheatley_l2_ablation"
mkdir -p "$RESULTS"

GNN="--gconv_type gatv2 --n_layers_features_extractor 4 --hidden_dim_features_extractor 32 --n_attention_heads 4 --graph_pooling learn --residual_gnn --normalize_gnn"
TRAIN="--n_j 15 --n_m 15 --total_timesteps 100000 --n_steps_episode 512 --batch_size 128 --n_epochs 10 --lr 0.0002 --n_workers 4 --n_validation_env 10 --validation_freq 5 --seed 42 --device cuda --transition_model_config simple --reward_model_config Sparse --duration_type deterministic --criterion makespan --fixed_problem --fixed_validation --first_machine_id_is_one --disable_visdom --custom_heuristic_names SPT MWKR MOPNR"

run() {
    local name=$1; shift
    echo "[$(date +%H:%M:%S)] START: $name"
    python3 -m jssp.train "$@" 2>&1 | tee "$RESULTS/${name}.log" | grep -E "best|Best|makespan|SPT|MWKR|MOPNR|Iteration|ratio" | tail -10
    echo "[$(date +%H:%M:%S)] DONE: $name"
    echo ""
}

for inst in ta01 ta02 ta03; do
    echo "===== B1: No L2 - $inst ====="
    run "B1_${inst}" --load_problem "instances/taillard/${inst}.txt" $GNN $TRAIN \
        --features duration --conflicts att --add_rp_edges none --no_tct \
        --path "$RESULTS/B1_${inst}"

    echo "===== B2: Partial L2 - $inst ====="
    run "B2_${inst}" --load_problem "instances/taillard/${inst}.txt" $GNN $TRAIN \
        --features duration mopnr mwkr --conflicts att --add_rp_edges none \
        --path "$RESULTS/B2_${inst}"

    echo "===== DKAP: Full L2 - $inst ====="
    run "DKAP_${inst}" --load_problem "instances/taillard/${inst}.txt" $GNN $TRAIN \
        --features duration mopnr mwkr total_job_time total_machine_time job_completion_percentage machine_completion_percentage \
        --conflicts clique --add_rp_edges frontier_strict --mid_in_edges --precompute_cliques \
        --path "$RESULTS/DKAP_${inst}"
done

echo "=========================================="
echo "COLLECTING RESULTS"
echo "=========================================="

python3 << 'PYEOF'
import re, glob, json, os
results = {}
for logf in sorted(glob.glob("/home/woodcross/projects/thesis/experiments/results/wheatley_l2_ablation/*.log")):
    name = os.path.basename(logf).replace('.log','')
    with open(logf) as f:
        log = f.read()
    # Find best validation makespan
    best = re.findall(r'[Bb]est.*?(\d+\.\d+)', log)
    ratios = re.findall(r'ratio.*?(\d+\.\d+)', log)
    spt = re.findall(r'SPT.*?(\d+\.\d+)', log)
    mwkr = re.findall(r'MWKR.*?(\d+\.\d+)', log)
    results[name] = {
        "best_makespan": float(best[-1]) if best else None,
        "best_ratio": float(ratios[-1]) if ratios else None,
        "spt_baseline": float(spt[-1]) if spt else None,
        "mwkr_baseline": float(mwkr[-1]) if mwkr else None,
    }
    print(f"{name:15s}: makespan={results[name]['best_makespan']}, ratio={results[name]['best_ratio']}")

with open("/home/woodcross/projects/thesis/experiments/results/wheatley_l2_ablation/summary.json", 'w') as f:
    json.dump(results, f, indent=2)

# Condition averages
from collections import defaultdict
by_cond = defaultdict(list)
for name, r in results.items():
    cond = name.split('_')[0]  # B1, B2, DKAP
    if r['best_makespan']:
        by_cond[cond].append(r['best_makespan'])

print("\n===== SUMMARY =====")
optima = {"ta01": 1231, "ta02": 1244, "ta03": 1218}
for cond in ['B1', 'B2', 'DKAP']:
    vals = by_cond.get(cond, [])
    if vals:
        avg = sum(vals) / len(vals)
        print(f"  {cond:6s}: avg_makespan = {avg:.1f} (n={len(vals)})")
PYEOF
