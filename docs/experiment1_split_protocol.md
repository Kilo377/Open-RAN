# Experiment 1 Split Protocol

This protocol keeps Experiment 1 runnable in small batches while preserving a
clear path from development runs to a paper-facing transfer table.

## Exp.1-A Main Comparison

The frozen task plan is:

- `experiments/configs/exp1_main_a_bootstrap_task_plan.yaml`

It defines 18 transfer tasks, the seven main comparison methods, a 120-slot
rollout horizon, and seed-separated cumulative memory stores. Each task keeps
two warm-up scenarios and one transfer scenario so a failed task can be rerun
without rerunning the entire Experiment 1 matrix.

The split-config generator is:

```powershell
$python = 'C:/Users/Asus/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $python experiments/generate_exp1_task_split_configs.py `
  --plan experiments/configs/exp1_main_a_bootstrap_task_plan.yaml
```

The generator writes one config per `(task, seed)` and a `task_manifest.csv`
under `experiments/configs/exp1_main_a_bootstrap_split/`.

## Bootstrap Memory

The task plan points reflective variants at:

- `E:/oran_slot_sim_outputs/paper_eval/exp1_main_a_bootstrap_memory_v1`

For the first seed-1 continuation batch, initialize that directory from the
completed v17 cumulative memory snapshot:

```powershell
Copy-Item -Recurse -Force `
  'E:/oran_slot_sim_outputs/paper_eval/exp1_micro_v17_inherited_memory/*' `
  'E:/oran_slot_sim_outputs/paper_eval/exp1_main_a_bootstrap_memory_v1/'
```

This is useful for development because the full method starts with reflective
evidence instead of rebuilding every historical case. A paper-facing main table
must state how memory was initialized. Do not mix a warm-start seed with
cold-start seeds in one headline table without labeling the memory protocol.

Recommended final-table choices are:

1. Use only task-local warm-up memory per seed.
2. Initialize every seed from the same frozen calibration memory corpus, then
   write updates to separate seed-scoped stores.

The calibrated 5-seed Exp.1-A batch uses the second choice. Clone the same
frozen memory snapshot to `seed1.json` through `seed5.json` for each persistent
method before launching the new split configs. Then generate the split configs
with a fresh experiment prefix and memory directory:

```powershell
& $python experiments/generate_exp1_task_split_configs.py `
  --plan experiments/configs/exp1_main_a_bootstrap_task_plan.yaml `
  --output-dir experiments/configs/exp1_main_a_calibrated_split `
  --experiment-prefix exp1a_cal_v1 `
  --memory-dir E:/oran_slot_sim_outputs/paper_eval/exp1_main_a_calibrated_memory_v1
```

## Small Batches

Start with a memory-heavy seed-1 batch before committing to all 18 tasks:

```powershell
$python = 'C:/Users/Asus/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $python experiments/generate_exp1_task_split_configs.py `
  --plan experiments/configs/exp1_main_a_bootstrap_task_plan.yaml `
  --output-dir experiments/configs/exp1_main_a_seed1_pilot_split `
  --task-ids t05_mobility_to_blockage t15_stale_to_blockage t17_load_balance_to_scalability `
  --seeds 1
```

Validate one generated config before launching real rollouts:

```powershell
& $python experiments/run_experiment1_transfer.py `
  --config experiments/configs/exp1_main_a_seed1_pilot_split/exp1_main_a_bootstrap_v1_t05-mobility-to-blockage_seed1.yaml `
  --dry-run
```

With the current pilot settings, one `(task, seed)` config expands to 49 case
runs: seven methods over four warm-up repeats and three held-out transfer
repeats. Only transfer rows feed the headline table. For paired confidence
intervals and win rates, aggregate transfer repeats inside each `(task, seed,
method)` cell first; do not count the repeated transfer rollouts as independent
paper samples.

Run one generated config at a time:

```powershell
& $python experiments/run_experiment1_transfer.py `
  --config experiments/configs/exp1_main_a_seed1_pilot_split/<generated-config>.yaml `
  --skip-completed
```

After any completed batch, merge the available split outputs:

```powershell
& $python experiments/aggregate_experiment1_split_runs.py `
  --manifest experiments/configs/exp1_main_a_seed1_pilot_split/task_manifest.csv `
  --output-dir E:/oran_slot_sim_outputs/paper_eval/exp1_main_a_seed1_pilot_merged `
  --allow-missing
```

For a complete calibrated 5-seed batch, merge all 90 split outputs and write
paired task-seed statistics:

```powershell
& $python experiments/aggregate_experiment1_split_runs.py `
  --manifest experiments/configs/exp1_main_a_calibrated_split/task_manifest.csv `
  --output-dir E:/oran_slot_sim_outputs/paper_eval/exp1_main_a_calibrated_5seed_merged `
  --no-plot

& $python experiments/summarize_experiment1_paired_cells.py `
  --manifest experiments/configs/exp1_main_a_calibrated_split/task_manifest.csv `
  --output E:/oran_slot_sim_outputs/paper_eval/exp1_main_a_calibrated_5seed_merged/paired_cell_summary.csv
```

## Exp.1-B Memory Ablations

Exp.1-B reuses the same 18 task cells and seed protocol, but drops literature
baselines so the paired comparisons focus on the memory mechanism. For the
paper's main ablation table, use the compact core set:

- `full_method`
- `no_memory_planner`
- `success_only_memory`
- `no_transfer_guard`

Use `shared_memory_scope: task_seed` for formal Exp.1-B runs. Each task/seed
then starts from the same frozen bootstrap memory, while warm-up and transfer
updates are kept inside that task/seed cell. This prevents held-out transfer
outcomes from one task from entering another task's memory store. The runner
initializes missing task/seed memory files from `bootstrap_memory_source`.

Generate the split configs from the Exp.1-A task plan with explicit variant and
persistent-memory overrides:

```powershell
& $python experiments/generate_exp1_task_split_configs.py `
  --plan experiments/configs/exp1_main_a_bootstrap_task_plan.yaml `
  --output-dir experiments/configs/exp1_main_b_calibrated_split `
  --experiment-prefix exp1b_mem_v1 `
  --memory-dir E:/oran_slot_sim_outputs/paper_eval/exp1_main_b_calibrated_memory_v1 `
  --memory-scope task_seed `
  --variants full_method no_memory_planner success_only_memory no_transfer_guard `
  --persistent-variants full_method success_only_memory no_transfer_guard
```

Run the generated configs with `experiments/run_experiment1_transfer.py` and
aggregate them the same way as Exp.1-A. This keeps every Full-vs-ablation
comparison paired by task and seed.

For long split runs, suppress per-slot simulator logs while keeping case-level
progress:

```powershell
$env:ORAN_SLOT_METRICS = "0"
```

Keep `shuffled_memory` and `stale_memory_no_decay` as appendix robustness
ablations. They use the same protocol and can be generated by adding those
variants to both `--variants` and `--persistent-variants`.
