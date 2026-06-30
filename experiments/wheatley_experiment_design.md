# L2 Ablation Experiment: Wheatley RL-Based Job Shop Scheduling

## 1. Motivation

The DKAP (Domain Knowledge-Aligned Pipeline) framework hypothesizes that L2 -- the structuring of domain knowledge into the computational pipeline -- is the critical layer determining system performance. In LLM-based Text-to-SQL, removing L2 drops accuracy by 26 percentage points. This experiment tests the same hypothesis in a fundamentally different domain: RL-based combinatorial optimization for Job Shop Scheduling (JSSP).

**Research Question:** Does removing domain knowledge structuring (L2) from an RL-based JSSP solver degrade makespan performance analogously to how removing L2 from LLM-based Text-to-SQL degrades accuracy?

## 2. System Under Study: Wheatley

Wheatley (Jolibrain, 2023) is a GNN-based RL system for JSSP that uses PPO to learn dispatching policies. Its architecture embeds several forms of domain knowledge:

| Component | Domain Knowledge (L2) | Role |
|---|---|---|
| Node features | TCT bounds, MWKR, MOPNR, completion percentages | Encode scheduling-specific state information |
| Edge structure | Conflict cliques (machine sharing), resource precedence | Encode constraint relationships between operations |
| Edge embeddings | Machine-ID-typed edges | Distinguish constraint sources |
| Feature normalization | Duration/time-aware normalization | Scale features relative to problem structure |

### 2.1 Feature Architecture (from `state.py`)

The feature vector for each operation node has the following layout:

| Offset | Feature | Size | L2 Category |
|--------|---------|------|-------------|
| 0 | `is_affected` (scheduled flag) | 1 | Basic |
| 1-4 | Task Completion Time bounds (TCT) | 4 | Advanced domain |
| 5 | `selectable` (eligible for scheduling) | 1 | Basic |
| 6 to 6+M | Machine ID encoding | M | Domain structure |
| variable | `duration` (processing times) | 4 | Basic domain |
| variable | `mopnr` (Most Operations Remaining) | 1 | Advanced domain |
| variable | `mwkr` (Most Work Remaining) | 4 | Advanced domain |
| variable | `total_job_time` | 4 | Advanced domain |
| variable | `total_machine_time` | 4 | Advanced domain |
| variable | `job_completion_percentage` | 4 | Advanced domain |
| variable | `machine_completion_percentage` | 4 | Advanced domain |

### 2.2 Graph Structure

- **Precedence edges**: Job ordering constraints (always present)
- **Conflict clique edges**: Connect operations sharing the same machine (domain L2)
- **Resource precedence edges**: Encode scheduling order on shared resources (domain L2)
- **Edge type embeddings**: Distinguish edge semantics via learned embeddings

## 3. Experimental Design

### 3.1 Conditions

#### B1: No L2 (Vanilla RL)

Strips all domain knowledge, retaining only what the environment minimally requires:

- **Features**: `duration` only (plus mandatory `is_affected`, `selectable`, `machine_id`)
- **Graph**: Precedence edges only; attention-based conflicts (`--conflicts att`), no cliques
- **Resource precedence**: None (`--add_rp_edges none`)
- **TCT**: Disabled (`--no_tct`)
- **Edge types**: No machine-ID encoding in edges

This condition represents what a domain-agnostic RL practitioner might build: a GNN operating on a basic graph with minimal features.

#### B2: Partial L2 (Basic Domain Knowledge)

Includes common dispatching-rule-inspired features but omits advanced graph structure:

- **Features**: `duration`, `mopnr`, `mwkr`
- **Graph**: Precedence edges only; attention-based conflicts, no cliques
- **Resource precedence**: None (`--add_rp_edges none`)
- **TCT**: Enabled (default)
- **Edge types**: No machine-ID encoding in edges

This condition represents a practitioner who knows standard scheduling heuristics (SPT, MWKR, MOPNR) and encodes them as features, but does not structure the graph to reflect resource constraints.

#### DKAP: Full L2 (Complete Domain Knowledge)

All domain knowledge integrated into features and graph structure:

- **Features**: `duration`, `mopnr`, `mwkr`, `total_job_time`, `total_machine_time`, `job_completion_percentage`, `machine_completion_percentage`
- **Graph**: Conflict clique edges (`--conflicts clique` with `--precompute_cliques`)
- **Resource precedence**: Frontier-strict (`--add_rp_edges frontier_strict`)
- **TCT**: Enabled
- **Edge types**: Machine-ID in edges (`--mid_in_edges`)

### 3.2 Controlled Variables

All conditions share the same:

| Parameter | Value | CLI Flag |
|-----------|-------|----------|
| GNN type | GATv2 | `--gconv_type gatv2` |
| GNN layers | 6 | `--n_layers_features_extractor 6` |
| Hidden dim | 64 | `--hidden_dim_features_extractor 64` |
| MLP layers per GNN layer | 3 | `--n_mlp_layers_features_extractor 3` |
| Attention heads | 4 | `--n_attention_heads 4` |
| Graph pooling | Learnable | `--graph_pooling learn` |
| Total timesteps | 1,000,000 | `--total_timesteps 1000000` |
| PPO epochs | 10 | `--n_epochs 10` |
| Batch size | 128 | `--batch_size 128` |
| Learning rate | 2e-4 | `--lr 2e-4` |
| Optimizer | RAdam | `--optimizer radam` |
| Reward | Sparse | `--reward_model_config Sparse` |
| Transition model | Simple | `--transition_model_config simple` |
| Discount factor | 1.0 | `--gamma 1.0` |
| Residual connections | Yes | `--residual_gnn` |
| Layer normalization | Yes | `--normalize_gnn` |
| Input normalization | Yes | (default) |

### 3.3 Benchmark Instances

We use Taillard (1993) benchmark instances, the standard JSSP benchmark:

| Group | Size (J x M) | Instances | Known Optima |
|-------|-------------|-----------|--------------|
| Small | 15 x 15 | ta01 (1231), ta02 (1244), ta03 (1218) | Proven optimal |
| Medium | 20 x 15 | ta11 (1361), ta12 (1367), ta13 (1342) | Proven optimal |
| Large | 20 x 20 | ta21 (1644), ta22 (1600), ta23 (1557) | Best known |

### 3.4 Seeds and Repetitions

Each (condition, instance) pair is run with 3 seeds: {42, 123, 456}, yielding 3 conditions x 9 instances x 3 seeds = 81 total training runs.

### 3.5 Heuristic Baselines

Each run also evaluates three dispatching rule heuristics for reference:
- **SPT** (Shortest Processing Time)
- **MWKR** (Most Work Remaining)
- **MOPNR** (Most Operations Remaining)

## 4. Metrics

### 4.1 Primary Metric: Optimality Gap

$$\text{Gap}(\%) = \frac{\text{Makespan}_{\text{RL}} - \text{Makespan}_{\text{optimal}}}{\text{Makespan}_{\text{optimal}}} \times 100$$

Lower is better. A gap of 0% means the RL agent found the optimal schedule.

### 4.2 Secondary Metrics

- **Makespan** (absolute): Direct comparison of schedule length
- **RL/Heuristic ratio**: How much RL improves over dispatching rules
- **Training convergence**: Validation makespan vs. training timesteps (from logs)

### 4.3 Statistical Analysis

- Report mean and standard deviation of optimality gap across seeds
- Paired t-test or Wilcoxon signed-rank test between conditions (paired by instance and seed)
- Effect size (Cohen's d) for B1 vs. DKAP comparison

## 5. Expected Results and Hypothesis

### H1: Domain knowledge structuring (L2) significantly impacts RL scheduling performance

We predict:

| Condition | Expected Mean Gap | Reasoning |
|-----------|-------------------|-----------|
| B1 (No L2) | 15-30% | GNN must learn all structure from scratch |
| B2 (Partial L2) | 8-18% | Heuristic features help but graph is impoverished |
| DKAP (Full L2) | 3-10% | Full domain alignment enables effective learning |

### H2: The L2 impact scales with problem complexity

Larger instances (20x20) should show a proportionally larger gap between B1 and DKAP, as the importance of domain-structured information grows with combinatorial complexity.

### H3: Cross-domain L2 impact analogy

If the gap between B1 and DKAP is in the range of 15-25 percentage points, this would be analogous to the 26pp drop observed in Text-to-SQL when removing L2 -- supporting the DKAP thesis that L2 is the universally critical layer.

## 6. Execution

### 6.1 Dry Run (Verify Configuration)

```bash
cd /home/woodcross/projects/thesis
python experiments/wheatley_l2_ablation.py --dry-run
```

### 6.2 Print All Commands (for Manual / SLURM Execution)

```bash
python experiments/wheatley_l2_ablation.py --print-commands --device cuda
```

### 6.3 Full Execution

```bash
python experiments/wheatley_l2_ablation.py --device cuda --seeds 42 123 456
```

### 6.4 Partial Execution (Single Condition)

```bash
python experiments/wheatley_l2_ablation.py --conditions DKAP_full_L2 --instances ta01 ta11 ta21
```

### 6.5 Example CLI Commands

**B1 (No L2) on ta01:**
```bash
python jssp/train.py \
  --load_problem instances/taillard/ta01.txt \
  --n_j 15 --n_m 15 --max_n_j 15 --max_n_m 15 \
  --features duration \
  --conflicts att --add_rp_edges none --no_tct \
  --gconv_type gatv2 --n_layers_features_extractor 6 \
  --hidden_dim_features_extractor 64 --n_attention_heads 4 \
  --graph_pooling learn --residual_gnn --normalize_gnn \
  --total_timesteps 1000000 --seed 42 --device cuda \
  --fixed_problem --fixed_validation --first_machine_id_is_one \
  --reward_model_config Sparse --disable_visdom \
  --custom_heuristic_names SPT MWKR MOPNR \
  --path results/B1_no_L2__ta01__seed42
```

**DKAP (Full L2) on ta01:**
```bash
python jssp/train.py \
  --load_problem instances/taillard/ta01.txt \
  --n_j 15 --n_m 15 --max_n_j 15 --max_n_m 15 \
  --features duration mopnr mwkr total_job_time total_machine_time \
    job_completion_percentage machine_completion_percentage \
  --conflicts clique --add_rp_edges frontier_strict --mid_in_edges \
  --precompute_cliques \
  --gconv_type gatv2 --n_layers_features_extractor 6 \
  --hidden_dim_features_extractor 64 --n_attention_heads 4 \
  --graph_pooling learn --residual_gnn --normalize_gnn \
  --total_timesteps 1000000 --seed 42 --device cuda \
  --fixed_problem --fixed_validation --first_machine_id_is_one \
  --reward_model_config Sparse --disable_visdom \
  --custom_heuristic_names SPT MWKR MOPNR \
  --path results/DKAP_full_L2__ta01__seed42
```

## 7. Analysis Plan

### 7.1 Primary Analysis

1. Compute mean optimality gap per condition, aggregated across all instances and seeds
2. Test H1 via paired statistical test (B1 vs. DKAP, B2 vs. DKAP)
3. Report the L2 impact in percentage points: `gap(B1) - gap(DKAP)`

### 7.2 Breakdown Analysis

1. Per-size breakdown (15x15, 20x15, 20x20) to test H2
2. Per-feature contribution: compare B2 (features without graph) vs. a hypothetical "graph without features" to disentangle feature vs. structure contributions

### 7.3 Cross-Domain Comparison Table

| System | Domain | L2 Present | L2 Absent | L2 Impact (pp) |
|--------|--------|-----------|-----------|----------------|
| Text-to-SQL | NL2SQL | ~85% acc | ~59% acc | ~26 |
| Wheatley JSSP | Scheduling | gap(DKAP) | gap(B1) | TBD |

## 8. Threats to Validity

1. **Training budget**: 1M timesteps may be insufficient for B1 to converge (the agent must learn more from scratch). Mitigation: also report learning curves, and consider extended runs for B1.
2. **Architecture confound**: B1 has fewer input features, hence a smaller input dimension to the GNN. This could affect capacity. Mitigation: the GNN hidden dim (64) and depth (6 layers) are held constant; only the input embedding layer changes size.
3. **Instance selection**: 9 Taillard instances may not generalize. Mitigation: instances span 3 size classes; future work can extend to all 80 Taillard instances.
4. **Optimal values**: For 20x20 instances, "optimal" values are best known upper bounds, not proven optima.
