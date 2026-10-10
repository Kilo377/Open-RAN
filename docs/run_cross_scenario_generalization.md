# Cross-Scenario Generalization Run Guide

This guide prepares and runs the fixed cross-scenario generalization benchmark for the intent-driven O-RAN prototype.

## 1. What This Benchmark Runs

The benchmark answers one fixed question:

> Under different load, user-distribution, and xApp-availability conditions, does the full intent-driven orchestration method maintain lower unseen-scene intent loss and smaller transfer gap than the selected baselines?

The benchmark uses:

- Fixed intent: `Increase cluster throughput by 10% while keeping BLER stable`
- Fixed scenario manifest: [generalization_manifest.csv](/D:/研究�?oran_slot_sim/experiments/manifests/generalization_manifest.csv)
- Fixed variants:
  - `full_method`
  - `static_xapp_selection`
  - `no_rl_guidance`
  - `no_safety_shield`
- Default seeds:
  - `1`
  - `2`
  - `3`

The main experiment config is [cross_scenario_generalization.yaml](/D:/研究�?oran_slot_sim/experiments/configs/cross_scenario_generalization.yaml).

## 2. Dependencies

Recommended environment:

- Python `3.10+`
- The same package environment already used for the simulator and Phase 1 prototype

At minimum, make sure the environment includes:

- `pyyaml`
- `numpy`
- `tensorflow` / `sionna` if your existing simulator setup requires them

If you already run the simulator locally, reuse the same Python environment.

## 3. API Key Configuration

The current intent translation and xApp selection path reads the API key from:

- `DEEPSEEK_API_KEY`

### Windows PowerShell

```powershell
$env:DEEPSEEK_API_KEY="your_api_key_here"
```

### Windows persistent user variable

```powershell
[System.Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "your_api_key_here", "User")
```

### Linux / macOS

```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```

If the key is missing and you run the real benchmark, the runner will fail fast with a clear preflight error.

## 4. Dry-Run Validation

Use this first to validate config, manifest, output paths, and run planning without executing the full benchmark:

```powershell
python -m experiments.run_cross_scenario_generalization --config experiments/configs/cross_scenario_generalization.yaml --dry-run
```

This command will:

- load the manifest
- resolve all scenarios
- resolve all variants and seeds
- create the output root
- write `config_snapshot.yaml`
- copy `scenario_manifest.csv`
- write `run_plan.json`

It will not execute simulator cases.

## 5. Main Run Command

After the API key is configured, run:

```powershell
python -m experiments.run_cross_scenario_generalization --config experiments/configs/cross_scenario_generalization.yaml
```

This single command will:

- run all manifest scenarios
- run all configured variants
- run all configured seeds
- write per-case outputs
- build `case_results.csv`
- build `aggregated_results.csv`
- build `aggregated_results.json`
- build `orchestration_behavior.csv`
- populate `plots_input/`

## 6. Output Layout

Outputs are written under:

- [outputs/generalization/cross_scenario_generalization_v1](/D:/研究�?oran_slot_sim/outputs/generalization/cross_scenario_generalization_v1)

Expected structure:

```text
outputs/
  generalization/
    cross_scenario_generalization_v1/
      config_snapshot.yaml
      scenario_manifest.csv
      run_plan.json
      case_results.csv
      aggregated_results.csv
      aggregated_results.json
      orchestration_behavior.csv
      plots_input/
        aggregated_results.csv
        case_results.csv
        orchestration_behavior.csv
      cases/
        <scenario_id>__<variant>__seed<k>/
          final_summary.json
          orchestration_epoch_history.json
          slot_metrics.csv
          slot_snapshots.json
```

## 7. What To Inspect First

Start with these files:

1. [aggregated_results.csv](/D:/研究�?oran_slot_sim/outputs/generalization/cross_scenario_generalization_v1/aggregated_results.csv)
   Use this for the main table:
   - `seen_mean_intent_loss_normalized`
   - `unseen_mean_intent_loss_normalized`
   - `transfer_gap_intent_loss_normalized`
   - `all_mean_safety_violation_rate`
   - `all_mean_stability_penalty_mean`
   - `all_mean_xapp_switching_cost_mean`

2. [aggregated_results.json](/D:/研究�?oran_slot_sim/outputs/generalization/cross_scenario_generalization_v1/aggregated_results.json)
   Use this for machine-readable split summaries and per-variant metadata.

3. [orchestration_behavior.csv](/D:/研究�?oran_slot_sim/outputs/generalization/cross_scenario_generalization_v1/orchestration_behavior.csv)
   Use this to analyze how orchestration changes across scenes:
   - `selected_xapps`
   - `guidance_id`
   - `selected_rule_ids`
   - `hard_safety_envelope_fields`
   - `guided_action_region_fields`
   - `translation_refreshed_this_epoch`

4. Per-case files under `cases/`
   Important ones:
   - `final_summary.json`
   - `orchestration_epoch_history.json`
   - `slot_metrics.csv`

## 8. Rebuild Aggregated Results Only

If the per-case runs already exist and you only want to refresh the aggregated exports:

```powershell
python -m experiments.run_cross_scenario_generalization --config experiments/configs/cross_scenario_generalization.yaml --postprocess-only
```

You can also call the postprocessor directly:

```powershell
python -m experiments.postprocess_generalization --experiment-dir outputs/generalization/cross_scenario_generalization_v1
```

## 9. Common Errors

### Missing API key

Symptom:

- runner exits before simulation starts

Check:

- `DEEPSEEK_API_KEY` is set in the same shell session

### Manifest references a missing scenario

Symptom:

- error about unknown `source_case_id` or matrix case

Check:

- [generalization_manifest.csv](/D:/研究�?oran_slot_sim/experiments/manifests/generalization_manifest.csv)
- [scenario_matrix_templates.py](/D:/研究�?oran_slot_sim/scenarios/scenario_matrix_templates.py)

### Output directory confusion

Check:

- experiment root:
  [outputs/generalization/cross_scenario_generalization_v1](/D:/研究�?oran_slot_sim/outputs/generalization/cross_scenario_generalization_v1)
- per-case outputs:
  [outputs/generalization/cross_scenario_generalization_v1/cases](/D:/研究�?oran_slot_sim/outputs/generalization/cross_scenario_generalization_v1/cases)

### Need to inspect whether xApp availability changed the orchestration

Check:

- `scenario_manifest.csv`
- `orchestration_behavior.csv`
- per-case `orchestration_epoch_history.json`

Compare:

- `selected_xapps`
- `selected_rule_ids`
- `hard_safety_envelope_fields`
- `guided_action_region_fields`
