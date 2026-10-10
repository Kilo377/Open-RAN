# Paper Method Alignment

This document records the runtime switches used to align the simulator with the
paper's method components and ablations. The switches are intentionally explicit
so experiment outputs can be audited without inferring behavior from a variant
name.

## Method Switches

| Paper component | Runtime switch | Default | Current status |
|---|---|---|---|
| Reflective memory | `memory_mode` | `full` | Implemented for `full`, `none`, `success_only`, `failed_only`, `diagnostic_only`, `raw_reward` interface |
| xApp orchestration | `selection_mode` | `reflective` | Logged and propagated; detailed baselines are next-step work |
| Feasibility repair | `repair_mode` | `deterministic` | Candidate-level benchmark implemented for deterministic, reject-only, schema-only, and no-projection modes |
| Conflict handling | `conflict_mode` | `typed_taxonomy` | Candidate-level taxonomy benchmark implemented for no-check, binary, PACIFISTA, and typed modes |
| LLM planning safety | `llm_pipeline_mode` | `schema_bound` | Logged and propagated; validator/projector/fallback already exist |
| Near-RT adaptation | `near_rt_mode` | `projected_tuning` | Logged and propagated; current tuner uses projected/guided safe tuning |

These switches are exported through:

- `final_summary.json` under `method_config`
- `orchestration_epoch_history.json`
- `orchestration_behavior.csv`
- aggregated result method metadata

## Memory Modes

`memory_mode` is the first switch with full behavior in this alignment pass.

| Mode | Retrieval | Success storage | Failure storage | Feeds selection |
|---|---|---:|---:|---:|
| `full` | success + failure | yes | yes | yes |
| `none` | disabled | no | no | no |
| `success_only` | success only | yes | no | yes |
| `failed_only` | failure only | no | yes | yes |
| `diagnostic_only` | success + failure traces | yes | yes | no |
| `raw_reward` | success + failure cases | yes | yes | yes |

`raw_reward` is currently an interface-compatible placeholder for later
reward-only case summaries. It still uses the same local case object until the
candidate scorer has a separate reward-history baseline.

## Variant Mapping

The baseline variants in `experiments/experiment_variants.py` now expose both
legacy boolean flags and paper-facing method switches.

Examples:

- `full_method`: `memory_mode=full`, `selection_mode=reflective`,
  `repair_mode=deterministic`, `conflict_mode=typed_taxonomy`,
  `llm_pipeline_mode=schema_bound`, `near_rt_mode=projected_tuning`
- `no_memory`: `memory_mode=none`
- `success_only_memory`: `memory_mode=success_only`
- `static_xapp_selection`: `selection_mode=static`
- `no_validator`: `llm_pipeline_mode=minimal_parse`, `repair_mode=disabled`
- `no_safety_shield`: `near_rt_mode=unshielded_tuning`

## Candidate Repair Evaluation

The first standalone experiment for the paper evaluation is implemented in:

- `experiments/candidate_repair_eval.py`
- `experiments/run_candidate_repair_eval.py`
- `experiments/configs/candidate_repair_eval.yaml`

It evaluates 10 LLM-style candidate fixtures covering unknown xApps, missing
dependencies, hard actuator conflicts, duplicate categories, resource overflow,
QoS guard omissions, policy conflicts, and empty candidates.

Run:

```bash
python experiments/run_candidate_repair_eval.py
```

Outputs are written to `outputs/candidate_repair/candidate_repair_eval_v1/`:

- `case_results.csv`
- `summary.csv`
- `summary.json`
- `config_snapshot.json`

Current smoke result: deterministic repair reaches 90% feasible candidate rate,
0 hard-conflict escape, and 88.9% repair success over initially infeasible
inputs in this fixture suite. The remaining rejected candidate is schema-invalid
and empty. Reject-only remains safe but rejects 90% of candidates; schema-only
and no-projection let unsafe candidates escape in sandbox.

## Conflict Taxonomy Evaluation

Experiment 4 is implemented in:

- `experiments/conflict_taxonomy_eval.py`
- `experiments/run_conflict_taxonomy_eval.py`
- `experiments/configs/conflict_taxonomy_eval.yaml`

It evaluates hard actuator conflicts, soft KPI conflicts, scenario-dependent
risk conflicts, policy conflicts, and no-conflict plans under no-check, binary,
PACIFISTA, and typed-taxonomy modes.

Current smoke result: typed taxonomy has 0 hard-conflict escape, 0 policy escape,
0 risk realization, 0 protected-KPI violation, and 0 over-rejection in the
fixture suite; binary conflict checking is safe but over-rejects, and the
PACIFISTA profiler misses policy conflicts.

## Runtime Provider Setup

The default paper-oriented LLM provider is now DeepSeek through the
OpenAI-compatible client path:

- `llm_provider=deepseek`
- `llm_api_key_env_var=DEEPSEEK_API_KEY`
- `llm_base_url=https://api.deepseek.com`
- `llm_model=deepseek-v4-pro`

The base Sionna PHY default remains CPU in `configs/phy.yaml`, while the paper
intent-orchestration override uses `phy_type=custom_phy` for stable CPU sweeps
on native Windows. A reusable Sionna GPU override remains available in
`configs/phy_gpu.yaml`; it requires TensorFlow to list a physical GPU device on
the machine.

## Next Alignment Tasks

The remaining method-alignment work should add behavior behind the already
propagated switches:

1. `selection_mode`: add fixed-plan, single-best, resource-aware, and graph-like
   baselines.
2. `llm_pipeline_mode`: add sandbox LLM safety evaluation modes.
3. `near_rt_mode`: add static, unconstrained, frequent re-orchestration, and
   guarded-lightweight-RL modes.
