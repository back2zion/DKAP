#!/bin/bash
# Wheatley L2 Ablation Experiment - Quick Run
# 3 conditions x 3 instances(6x6) x 1 seed = 9 runs
# Uses small 6x6 instances for fast turnaround (~5-10 min per run on RTX 3090)

set -e
cd /home/woodcross/projects/thesis/external_systems/wheatley

RESULTS_BASE="/home/woodcross/projects/thesis/experiments/results/wheatley_l2_ablation"
mkdir -p "$RESULTS_BASE"

# Use 6x6 deterministic instances (fastest)
INSTANCE_DIR="instances/jssp/deterministic/6x6"
INSTANCES=$(ls $INSTANCE_DIR/*.txt 2>/dev/null | head -3)

# Reduced training budget for feasibility (6x6 converges fast)
TOTAL_TIMESTEPS=200000
N_STEPS=512
BATCH_SIZE=64
N_EPOCHS=10
LR=0.0002
N_WORKERS=4
N_VAL_ENV=10
VAL_FREQ=5

# Common GNN args (held constant across conditions)
GNN_ARGS="--gconv_type gatv2 --n_layers_features_extractor 4 --hidden_dim_features_extractor 32 --n_mlp_layers_features_extractor 2 --n_attention_heads 4 --graph_pooling learn --residual_gnn --normalize_gnn"

# Common training args
TRAIN_ARGS="--total_timesteps $TOTAL_TIMESTEPS --n_steps_episode $N_STEPS --batch_size $BATCH_SIZE --n_epochs $N_EPOCHS --lr $LR --n_workers $N_WORKERS --n_validation_env $N_VAL_ENV --validation_freq $VAL_FREQ --seed 42 --device cuda --transition_model_config simple --reward_model_config Sparse --duration_type deterministic --criterion makespan --fixed_problem --fixed_validation --first_machine_id_is_one --disable_visdom --custom_heuristic_names SPT MWKR MOPNR"

echo "=============================================="
echo "DKAP L2 Ablation Experiment - Wheatley JSSP"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Instances: 6x6 deterministic"
echo "Timesteps: $TOTAL_TIMESTEPS"
echo "=============================================="
echo ""

run_experiment() {
    local COND_NAME=$1
    local COND_ARGS=$2
    local INST_PATH=$3
    local INST_NAME=$(basename "$INST_PATH" .txt)
    local OUT_DIR="$RESULTS_BASE/${COND_NAME}__${INST_NAME}"

    echo "[$(date +%H:%M:%S)] Running: $COND_NAME / $INST_NAME"
    mkdir -p "$OUT_DIR"

    python3 -m jssp.train \
        --load_problem "$INST_PATH" \
        --path "$OUT_DIR" \
        $GNN_ARGS \
        $TRAIN_ARGS \
        $COND_ARGS \
        2>&1 | tee "$OUT_DIR/train.log" | tail -5

    echo "[$(date +%H:%M:%S)] Done: $COND_NAME / $INST_NAME"
    echo ""
}

# ===== B1: No L2 (Vanilla RL) =====
# Minimal features, no conflict cliques, no resource precedence, no TCT
B1_ARGS="--features duration --conflicts att --add_rp_edges none --no_tct"

# ===== B2: Partial L2 (Basic Domain) =====
# duration + mopnr + mwkr, but no conflict cliques, no RP edges
B2_ARGS="--features duration mopnr mwkr --conflicts att --add_rp_edges none"

# ===== DKAP: Full L2 =====
# All features, conflict cliques, resource precedence, machine_id in edges
DKAP_ARGS="--features duration mopnr mwkr total_job_time total_machine_time job_completion_percentage machine_completion_percentage --conflicts clique --add_rp_edges frontier_strict --mid_in_edges --precompute_cliques"

echo "===== CONDITION B1: No L2 (Vanilla RL) ====="
for inst in $INSTANCES; do
    run_experiment "B1_no_L2" "$B1_ARGS" "$inst"
done

echo "===== CONDITION B2: Partial L2 (Basic Domain) ====="
for inst in $INSTANCES; do
    run_experiment "B2_partial_L2" "$B2_ARGS" "$inst"
done

echo "===== CONDITION DKAP: Full L2 ====="
for inst in $INSTANCES; do
    run_experiment "DKAP_full_L2" "$DKAP_ARGS" "$inst"
done

echo "=============================================="
echo "All runs complete. Collecting results..."
echo "=============================================="

# Collect results
python3 << 'PYEOF'
import os, json, re, glob

results_base = "/home/woodcross/projects/thesis/experiments/results/wheatley_l2_ablation"
all_results = []

for run_dir in sorted(glob.glob(f"{results_base}/*__*")):
    run_name = os.path.basename(run_dir)
    log_path = os.path.join(run_dir, "train.log")

    if not os.path.exists(log_path):
        continue

    with open(log_path) as f:
        log = f.read()

    # Parse condition and instance
    parts = run_name.split("__")
    condition = parts[0]
    instance = parts[1] if len(parts) > 1 else "unknown"

    # Extract best makespan from log
    # Look for validation results
    makespans = re.findall(r'(?:best|makespan|agent)[:\s]+(\d+\.?\d*)', log, re.IGNORECASE)
    heuristic_makespans = re.findall(r'(SPT|MWKR|MOPNR)[:\s]+(\d+\.?\d*)', log, re.IGNORECASE)

    best_makespan = float(makespans[-1]) if makespans else None

    result = {
        "condition": condition,
        "instance": instance,
        "best_makespan": best_makespan,
        "heuristics": {name: float(val) for name, val in heuristic_makespans[-3:]} if heuristic_makespans else {},
        "log_lines": len(log.split('\n')),
    }
    all_results.append(result)
    print(f"  {condition:20s} | {instance:10s} | makespan={best_makespan}")

# Save results
output_path = os.path.join(results_base, "ablation_results.json")
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to {output_path}")

# Summary table
if all_results:
    print("\n===== SUMMARY =====")
    from collections import defaultdict
    by_cond = defaultdict(list)
    for r in all_results:
        if r["best_makespan"]:
            by_cond[r["condition"]].append(r["best_makespan"])
    for cond in ["B1_no_L2", "B2_partial_L2", "DKAP_full_L2"]:
        vals = by_cond.get(cond, [])
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  {cond:20s}: avg_makespan = {avg:.1f} (n={len(vals)})")
PYEOF
