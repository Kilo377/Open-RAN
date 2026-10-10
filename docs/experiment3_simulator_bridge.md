# Experiment 3-B: Full Simulator Bridge

This experiment evaluates the trace-trained contract-shaped Near-RT tuner in
the repository's slot-driven `core.simulator.Simulator`.

## Frozen Protocol

- Scenario: `oran_paper_eval_base`, custom PHY, 7 managed cells.
- Workload: 15 UEs, with five replicas each of eMBB, URLLC, and mMTC.
- Base arrival rates are below the no-event service capacity so transient
  recovery is measurable.
- Events: traffic burst, mobility spike, interference perturbation, and
  resource perturbation.
- Severities: low, medium, and high.
- Methods: No Near-RT tuning, Rule-based bounded, DDPG + projection only, and
  Full CC-DDPG.
- Default scale: 4 events x 3 severities x 5 seeds x 4 methods = 240 episodes.

The abstract six-dimensional action is translated to executable
`ConfigAction` fields. A deadband maps the reference action to neutral
controls, and event intensity scales all adaptive methods' responses.

## Safety Interpretation

`raw_action_violation_rate` measures the neural policy proposal before
deployment-side event scaling and projection. `executed_action_violation_rate`
measures the action after the contract wrapper. This distinction exposes
projection reliance instead of hiding unsafe proposals behind the wrapper.

Full CC-DDPG should be interpreted as a safety-performance Pareto method. It
does not need to rank first on every event-specific throughput or queue metric,
but it should avoid raw and executed contract violations while remaining
competitive on recovery and area under violation.

## Artifacts

- `time_series.csv`: actual simulator KPI, event intensity, and executed action.
- `episode_metrics.csv`: per-event, per-severity, per-seed outcomes.
- `summary.csv`: method-level aggregate.
- `event_summary.csv`: event and severity aggregate.
- `recovery_curve.csv`: relative-slot recovery traces.
- `training_diagnostics.csv`: trace-policy training diagnostics.
- `run_plan.json`: frozen protocol and workload manifest.
