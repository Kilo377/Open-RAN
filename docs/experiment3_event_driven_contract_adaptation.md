# Experiment 3: Event-Driven Contract-Constrained Near-RT Adaptation

## Purpose

This experiment isolates the second paper challenge: after Non-RT has selected a
fixed xApp set and guidance contract, can the Near-RT layer adapt to sudden
traffic, mobility, interference, resource, and KPI-degradation events while
remaining inside the contract-level admissible action region?

The runner is a deterministic mechanism benchmark under the same artifact style
as `experiments/paper_sandbox_eval.py`.  The DRL methods train actual PyTorch
MLP actor/critic networks offline on CUDA when available, then evaluate the
frozen actor inference-only.  A NumPy neural fallback remains available when
PyTorch is not installed.  DDPG variants use one actor and one critic, TD3 uses
twin critics with delayed policy updates, and SAC uses a stochastic
tanh-Gaussian actor with twin critics and an entropy term.  The O-RAN-inspired
non-DRL baselines remain functional baselines under the same simulator, event
traces, xApp registry, and KPI pipeline.

## Runner

```bash
python experiments/run_experiment3_event_adaptation.py --evaluation-protocol heldout --method-suite paper --seeds 1 2 3 4 5 6 7 8 9 10
```

Quick smoke run:

```bash
python experiments/run_experiment3_event_adaptation.py --evaluation-protocol heldout --method-suite main --seeds 1 --training-seeds 1001 --validation-seeds 2001 --episode-length 80 --training-episodes 12
```

In environments without `matplotlib`, keep the CSV/JSON smoke test dependency
free by skipping figure generation:

```bash
python experiments/run_experiment3_event_adaptation.py --evaluation-protocol heldout --method-suite main --seeds 1 --training-seeds 1001 --validation-seeds 2001 --episode-length 80 --training-episodes 12 --no-time-series --no-figures
```

## Held-Out Protocol

The formal runner uses three disjoint seed sets:

- training seeds: neural-policy initialization and offline training only;
- validation seeds: `E1-E4` at medium severity for frozen reward/contract/hyperparameter selection;
- test seeds: inference-only evaluation and final reporting.

Training uses `E1-E4` at low/medium severity. The held-out test output is split into:

- `test_iid`: unseen test seeds on `E1-E4` low/medium;
- `test_ood_severity`: unseen test seeds on `E1-E4` high severity;
- `test_ood_event`: unseen test seeds on held-out `E5/E6` at all severities.

The runner rejects overlapping training, validation, and test seed sets.

## Scenario Design

Each episode has four phases:

- `pre_event_stable`: slots before event injection.
- `event_shock`: immediate event interval.
- `adaptation_or_recovery`: Near-RT tuner response and stabilization.
- final slots are included in the same recovery phase for stability checks.

Default settings:

- episode length `T = 300` slots
- event injection at `t_e = 0.4T`
- shock duration `Delta_s = 0.1T`
- recovery/stability window `W = 12` slots

Events:

- `E1 traffic_burst`
- `E2 hotspot_formation`
- `E3 mobility_spike`
- `E4 interference_perturbation`
- `E5 resource_perturbation`
- `E6 kpi_degradation`

Each event runs at `low`, `medium`, and `high` severity.

## Main Baselines

- `no_near_rt_tuning`
- `rule_based_bounded_tuner`
- `projected_greedy_search`
- `unconstrained_ddpg`
- `ddpg_projection_only`
- `full_contract_constrained_ddpg`

The `paper` method suite also includes neural wrapper ablations, neural
`CC-TD3`, neural `CC-SAC`, and functional O-RAN-inspired baselines:

- LLM-hRIC guidance + RL
- ORAN-GUIDE semantic RL
- TailO-RAN bounded tuner
- Kairos constrained energy control

## Metrics

The runner writes:

- `time_series.csv`: per-slot KPI, raw/executed action, projection, fallback,
  and latency trace.
- `episode_metrics.csv`: recovery time, AUV, peak degradation, guardrail,
  contract violation, fallback, projection, diversity, and latency metrics.
- `summary.csv`: aggregate table by method.
- `event_summary.csv`: aggregate table by event, severity, and method.
- `split_summary.csv`: aggregate table by held-out validation/IID/OOD split and method.
- `protocol_manifest.csv`: frozen split, seed, event, and severity assignments.
- `validation_episode_metrics.csv`, `test_iid_episode_metrics.csv`,
  `test_ood_severity_episode_metrics.csv`, and
  `test_ood_event_episode_metrics.csv`: split-specific episode tables.
- `training_diagnostics.csv`: projection-distance, raw-violation, boundary,
  fallback, diversity, reward variance, critic loss, actor loss, and Q-stability
  traces from the offline neural training loop.
- `run_plan.json`: fixed event definitions, contract dimensions, KPI specs, and
  method metadata.  It also records the neural backend, CUDA device, and training
  tuning values such as the quadratic KPI-violation shaping coefficient.

## Reward Shaping

The full contract-constrained reward keeps the paper's main terms: weighted KPI
violation, projection distance, raw-action violation, total variation, fallback,
and KPI guardrail penalties.  The current optimized runner adds a modest
quadratic KPI-violation shaping term during offline training:

```text
- lambda_quad * weighted_violation^2, where lambda_quad = 0.35
```

This does not change the evaluation metrics or contract projection/fallback
logic.  It makes the DDPG actor react more strongly to large post-event KPI
degradation while retaining zero raw/executed contract violations in the final
10-seed paper run.

Figures:

- `figure1_event_kpi_timeseries.png`
- `figure2_recovery_auv_boxplot.png`
- `figure3_safety_performance_pareto.png`
- `figure4_projection_diagnostics.png`
- `figure5_latency_cdf.png`
- `figure6_ablation_bar.png`

## Paper Interpretation

The expected conclusion is not that CC-DDPG is always best on every KPI.  The
intended reading is:

- static Non-RT guidance recovers slowly under event-driven dynamics;
- bounded rules are stable but less responsive under strong or mixed events;
- unconstrained DDPG can recover quickly but has high contract and guardrail
  risk;
- projection-only DDPG keeps executed actions admissible but relies more on
  clipping;
- full CC-DDPG approaches the recovery performance of unconstrained DDPG while
  reducing raw/executed contract violations, KPI guardrails, fallback use, and
  projection distance.
