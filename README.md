# oran_slot_sim

A slot-driven O-RAN-style RAN simulator prototype in Python.

This independent copy now defaults to Sionna 2.2 RT with actual NR-TB reference
feedback and SYS transport-budget adaptation (`sionna_rt_sys`,
`feedback_mode: nr_direct`, no online Monte Carlo probes). Geometry updates every 0.5 ms slot; Munich
uses dispersed legal roads for MU scheduling. See
[the current physics repair report](docs/rt_sys/TARGETED_PHYSICS_REPAIRS.md)
for execution scope, validation and Spark commands. The failed default
BLER/EESM abstraction is restricted to explicit diagnostic runs.

Latest system-level observation, packet delay, HARQ bookkeeping, low-rate NR
and fixed-baseline repairs: [2026-10-13 environment validation](docs/rt_sys/ENVIRONMENT_REPAIRS_20261013.md).
The RT entry defaults to fixed baseline control; algorithm execution requires
`--control-mode orchestrated`.

Shared initial conditions, warmup-separated load diagnostics, validated local
BLER caching/statistical feedback, same-codeword Chase, and scoped control
lifecycle checks are documented in [the research readiness repair report](docs/rt_sys/RESEARCH_READINESS_REPAIRS_20261013.md).
The xApp/rApp periods remain unchanged. Fast feedback is explicit
(`--feedback-mode nr_bler_bank`). The default `nr_direct` executes the MAC-selected
MCS with actual NR feedback; it does not promise a target BLER. Probe-based
reference modes remain explicit validation tools. See [no-probe operation](docs/rt_sys/NR_DIRECT_20261014.md).

## Current Status

The project has moved beyond the original minimum runnable skeleton and now
acts as a closed-loop RAN control simulator with:

- explicit grant-based multi-UE scheduling
- UE-internal logical channel / DRB / LCG multiplexing
- HARQ process / RV / NDI-aware retransmission state
- controller-visible measurement layering
- per-UE connection, link, antenna, beamforming, and resource-grid state
- scenario-package driven geometry and link generation
- lightweight serving-cell / mobility / traffic-steering semantics
- a backward-compatible `single_active_cell` mode and a `multi_cell_ready` cluster mode
- per-cell scheduler/resource-grid runtime state for managed cells
- system-level inter-cell interference approximation, including same-band overlap
  and small cross-band leakage
- RF-profile-aware candidate-cell steering based on `rf_band`, ACLR-like,
  ACS-like, and out-of-band isolation fields
- a dual-path Sionna PHY interface with a system-level path and a
  `sionna_cdl_su_mimo` path
- an intent-driven Non-RT -> Near-RT orchestration path with structured goal,
  xApp selection, reward shaping, and safe online tuning
- safety-constrained online tuning with hard parameter envelopes, forbidden
  action combinations, health-gated exploration, and baseline-centered updates
- structured orchestration logging for goal generation, xApp selection,
  reward shaping, failure-triggered reflection, and concise per-slot metrics

At a high level, the runtime loop is:

1. traffic arrives into UE queues
2. observations are built from the delayed measurement view
3. rApps and xApps produce policy, config, and command outputs
4. the executor applies the accepted control surface to the RAN state
5. the scheduler produces explicit multi-UE grants
6. the PHY executes link transmission for scheduled grants
7. ACK/NACK, queue, HARQ, link, beamforming, and resource-grid state are written back
8. snapshots, slot metrics, and final summaries are recorded

## Implemented Milestones

### V1-V3: Grant Scheduler, HARQ Resources, Grant-Driven PHY

- scheduler output upgraded from `scheduled_ue_ids` to explicit per-UE grants
- multiple UEs can be served in one slot
- HARQ retransmissions consume real slot resources
- PHY TB sizing and throughput depend on grant resources instead of a fixed per-UE budget

Key files:

- `models/scheduling.py`
- `ran/scheduler.py`
- `models/phy_input.py`
- `phy/custom_phy_adapter.py`
- `phy/sionna_engine.py`

### V4: Capability-Limited Control Surface

The control path supports both backward-compatible direct knobs and a more
realistic limited control surface:

- `scheduler_policy_hint`
- `latency_bias`
- `slice_weight_adjustment`
- `beam_candidate_preference`
- `power_cap_dbm`

Key files:

- `models/action.py`
- `ric/controller_manager.py`
- `ran/local_executor.py`

### V5: True vs Observable Metrics

The simulator now separates:

- true execution-side metrics
- controller-visible observable metrics
- measurement quality metadata
- control-decision trace

This separation is carried through slot metrics, snapshots, CSV output, and
final summary output.

Key files:

- `state/measurement_state.py`
- `core/measurement_updater.py`
- `core/metrics.py`
- `core/state_updater.py`
- `core/result_writer.py`

### V6.1-V6.4: Explicit Connection, Link, Antenna, Beamforming, and Resource Grid

The simulator now models several wireless engineering entities explicitly:

- connection state per UE
- per-UE link state
- explicit antenna capability state
- explicit beamforming state
- explicit OFDM slot resource-grid state

This means the system now distinguishes:

- not scheduled in the current slot
- connected but not granted
- out-of-sync connection state
- per-UE link quality and beamforming configuration
- slot-level resource-grid occupancy and overlap

Key files:

- `state/connection_state.py`
- `state/link_state.py`
- `state/antenna_state.py`
- `state/beamforming_state.py`
- `state/resource_grid_state.py`

### V7: Two-Layer Scheduler and More 3GPP-Like MAC Objects

The scheduler now works in layers:

- UE-level grant allocation
- UE-internal logical channel / DRB multiplexing
- LCG aggregation and BSR-like state
- demand-driven staged scheduling

The runtime state also carries more PHY/MAC-facing objects:

- HARQ process id / RV / NDI / retransmission round
- MCS table id / TBS selection method / DMRS pattern id
- CQI / RI / PMI validity-age style fields

Key files:

- `state/logical_channel_state.py`
- `state/logical_channel_group_state.py`
- `state/harq_state.py`
- `models/scheduling.py`
- `ran/scheduler.py`

### PHY/MAC Naming Alignment

The PHY/MAC stack now follows a more explicit 3GPP / O-RAN naming convention:

- the grant scheduler is treated as an abstract NR downlink MAC scheduler at
  the `O-DU High` side
- grant objects represent `DL-SCH` data scheduled onto a `PDSCH`-style OFDM
  resource grid
- the resource-grid state tracks PRB / OFDM-symbol occupancy, DM-RS overhead,
  HARQ process state, and CSI age in 3GPP-style terms
- the Sionna PHY adapter is treated as an abstract `O-DU Low / L1` executor,
  while explicit O-RU waveform processing remains out of scope
- numerology and slot PRB budgets are derived from carrier bandwidth and SCS
  using a 3GPP-style NRB lookup path with formula fallback rather than only a
  fixed manual PRB budget
- MCS selection uses more realistic 38.214-style tables for `qam64_low_se`,
  `qam64`, and `qam256`

### V8.1-V8.5: Scenario Package, Propagation Layering, Scenario Matrix, and Mobility

The simulator now has a reusable scenario package structure:

- scenario template
- scenario generator
- scenario registry

On top of that it now supports:

- geometry-driven link generation
- large-scale / small-scale propagation layering
- scenario matrices and scan scripts
- lightweight serving-cell / candidate-cell / handover state
- traffic-steering and mobility bias hooks

Key files:

- `models/scenario.py`
- `models/scenario_matrix.py`
- `scenarios/scenario_generator.py`
- `scenarios/scenario_registry.py`
- `scenarios/scenario_matrix_registry.py`
- `core/mobility_manager.py`

### M1-M4: Multi-Cell Runtime, Cell-Aware Control, Independent Scheduling, and Inter-Cell Interference

The simulator now supports a multi-cell-ready runtime that preserves the old
single-cell path while adding a cluster-aware execution mode:

- managed-cell runtime containers per cell
- cell-aware observation and cell-scoped control actions
- per-cell independent scheduler execution
- per-cell PHY input construction and cluster-level aggregation
- system-level inter-cell interference approximation
- cross-frequency leakage that remains non-zero even when cells are not
  co-channel
- RF-profile-aware coupling using `rf_band`, ACLR-like, ACS-like, and
  out-of-band isolation parameters

Key files:

- `state/managed_cell_runtime_state.py`
- `core/runtime_views.py`
- `core/rf_interference.py`
- `core/simulator.py`
- `core/mobility_manager.py`
- `models/observation.py`

### O1-O4: Intent-Driven Orchestration, Safe Online Tuning, and Reflection

The simulator now includes an intent-driven orchestration chain:

- user intent -> structured goal / KPI targets
- structured goal -> xApp selection plan
- goal + selected xApps -> reward specification and tuning ranges
- selected xApps + reward -> safe online parameter tuning at Near-RT timescales

The current orchestration path supports:

- a warmup stage before the first orchestration round so the intent is grounded
  on observed system state
- structured goal output with explicit KPI targets, risk posture, and
  constraints
- xApp portfolio selection with hard safety envelopes, forbidden combinations,
  and exploration health gates
- online tuning that operates within the hard/soft range intersection instead
  of unconstrained exploration
- baseline + clipped-delta style action updates rather than large direct jumps
- health-gated online exploration with freeze windows when BLER, queue, or
  throughput violate safety conditions
- failure-triggered Reflexion-style LLM self-critique for xApp reselection
- concise experiment logs that expose orchestration decisions without flooding
  later slot output

Current progress of this intent path:

- the full intent -> goal -> selection -> reward -> tuning loop is runnable
- online tuning is safety-constrained and suitable for simulator-side study
- xApp reselection already supports failure diagnosis and printed reflection
  traces
- repeated unchanged selection plans are suppressed from the log output
- the current focus is throughput, latency/reliability, mobility/steering,
  energy, and slicing-oriented intents rather than arbitrary natural-language
  intents

Key files:

- `experiments/run_intent_orchestration.py`
- `ric/non_rt/intent_goal_translator_rapp.py`
- `ric/non_rt/xapp_selection_rapp.py`
- `ric/non_rt/reward_shaping_rapp.py`
- `ric/near_rt/goal_parameter_tuning_xapp.py`
- `ric/controller_manager.py`
- `models/intent_orchestration.py`

## Current PHY Control Surface

### Already Controllable at Runtime

These parameters can currently be changed through the control path and affect
PHY behavior:

- transmit power via `tx_power_dbm`
- transmit power ceiling via `power_cap_dbm`
- beam/codebook behavior via `beam_candidate_preference`, `codebook_id`, and `beam_profile_id`
- scheduler-side resource allocation via grant PRBs, symbols, and resource positions
- per-UE rank/layer and beamforming state as part of the PHY input path
- per-cell steering and mobility hooks via `traffic_steering_preference` and
  `mobility_bias_db`
- cell-scoped application of config actions via `target_cell_id` and cluster
  scope semantics

Relevant files:

- `models/action.py`
- `ran/local_executor.py`
- `core/simulator.py`
- `phy/custom_phy_adapter.py`
- `phy/sionna_engine.py`

### Present as State or Config, But Not Yet a Runtime Control Knob

These values exist today, but they are not yet exposed as a runtime control
surface that xApps/rApps can change slot by slot:

- carrier frequency
- subcarrier spacing
- bandwidth

At the moment:

- `carrier_frequency_hz` and `subcarrier_spacing_hz` live in antenna state and
  are exposed in observation/snapshot output
- `bandwidth_hz` is a PHY backend configuration item used by the Sionna noise model
- none of the above are currently changed by `ConfigAction` or `LocalExecutor`

## What The Simulator Can Do Today

- grant-based multi-UE scheduling
- HARQ-aware resource accounting
- demand-driven staged scheduling
- logical channel / DRB / LCG-aware UE-internal multiplexing
- control-loop experimentation with delayed/filtered measurements
- explicit connected vs out-of-sync UE state
- lightweight serving-cell / handover-pending / traffic-steering behavior
- multi-cell-ready runtime with per-cell state containers
- per-cell independent scheduling in cluster mode
- cell-aware observation and cell-scoped control actions
- system-level inter-cell interference approximation for same-frequency and
  cross-frequency cases
- explicit per-UE link and beamforming state
- explicit slot resource-grid occupancy
- scenario-template and scenario-matrix driven experiments
- Sionna-backed PHY execution with both system-level and `cdl_su_mimo`-style paths
- intent-driven orchestration experiments with structured goals, xApp
  selection, reward shaping, and safe online tuning

## Intent Types Best Supported Today

The currently deployed xApps are strongest on intents that can be expressed as
cluster-level or cell-level KPI optimization problems.

Best-supported intent families:

- throughput / capacity intents such as "increase throughput by 10%",
  hotspot-capacity relief, and eMBB/video throughput improvement
- latency / URLLC / reliability intents such as reducing BLER, protecting
  URLLC latency, or containing queue buildup for delay-sensitive traffic
- mobility / handover / steering intents such as reducing handover risk,
  improving edge-user service continuity, and offloading overloaded cells
- energy / power-saving intents such as reducing power cost, trimming power
  caps, and using sleep-oriented policies under light load
- slicing / SLA intents such as biasing resources toward URLLC or eMBB and
  maintaining slice fairness or slice-specific queue balance
- compound intents such as "increase throughput while keeping BLER stable" or
  "reduce latency without aggressive power increase"

These intent families are mainly backed by the following deployed xApp
categories:

- `massive_mimo_beamformer_xapp` for throughput, SINR, coverage, and beam
  tradeoffs
- `spectrum_optimizer_xapp` for scheduler-policy and slice-efficiency tuning
- `latency_scheduler_xapp` and `urllc_guard_xapp` for latency and reliability
  protection
- `traffic_steering_xapp` and `mobility_predictor_xapp` for steering,
  handover-risk, and edge-performance improvement
- `power_saving_xapp` and `cell_sleep_xapp` for energy-oriented control
- `ran_slicing_manager_xapp` for slice-aware SLA and fairness control
- `goal_parameter_tuning_xapp` for constrained online fine tuning of the
  selected xApp portfolio

Less suitable today:

- arbitrary open-domain intents that cannot be mapped to RAN KPIs
- intents requiring full waveform-accurate multi-cell PHY optimization
- operational intents outside the current RAN control surface, such as alarm
  handling or core-network policy orchestration

## What Is Not Yet Implemented

- full RRC attach / random access / handover procedures
- fully concurrent waveform-level multi-cell PHY with exact inter-cell
  interference coupling; the current implementation uses per-cell scheduling
  plus sequential PHY execution with a system-level cross-cell interference
  approximation
- full waveform-level Sionna OFDM MIMO pipeline with explicit `ResourceGrid`,
  CDL channel objects, estimation, and detection
- full MU-MIMO joint detection rather than system-level overlap/interference approximation

## Main Entry Points

The main runnable entry points now focus on the paper-aligned orchestration
workflow and scenario-level evaluation.

### `experiments/run_intent_orchestration.py`

Use this when you want one end-to-end intent-driven orchestration run with:

- active intent lifecycle
- validated translation
- reflective xApp selection
- RL guidance generation
- Near-RT safe online tuning

### `experiments/run_cross_scenario_generalization.py`

Use this when you want the paper-style cross-scenario generalization batch:

- fixed intent
- manifest-based seen/unseen split
- `Full`, `StaticSelection`, `NoGuidance`, and `NoShield` variants
- aggregated results and orchestration-behavior export

### `experiments/run_scenario_matrix.py`

Use this when you want a broader scenario-matrix sweep outside the fixed
generalization manifest.

## Run

Single intent-driven orchestration run:

```bash
C:\Users\Asus\miniconda3\python.exe -m experiments.run_intent_orchestration
```

Cross-scenario generalization batch:

```bash
C:\Users\Asus\miniconda3\python.exe -m experiments.run_cross_scenario_generalization --config experiments/configs/cross_scenario_generalization.yaml
```

Scenario matrix batch:

```bash
C:\Users\Asus\miniconda3\python.exe -m experiments.run_scenario_matrix
```

## Dependencies

Minimal dependencies:

```bash
python -m pip install pyyaml numpy
```

For the Sionna backend you also need the relevant TensorFlow and Sionna
packages installed in your environment.

## Verification

The project is currently checked with:

```bash
npx pyright
C:\Users\Asus\miniconda3\python.exe -m unittest discover -s tests
```

Recent local verification status:

- `pyright`: 0 errors, 0 warnings, 0 informations
- `unittest`: 76 tests passing

无探针 DU 的发送资源契约、分组字节预留、固定 OLLA 和 xApp KPI/动作接口见
[接入说明](docs/NO_PROBE_DU_CONTRACTS_20261014.md)。公共字段与后端能力目录见
[生成的 KPI/xApp 目录](docs/rt_sys_kpi_and_xapp_catalog.json)。完整有限验收入口为
`scripts/validation/run_contract_acceptance.py`；测试默认只收集 `tests/`，避免历史结果里的源码副本被重复收集。

Spark 完整接入验收已完成（冻结源码 `4f28cb753220`）：
[验收报告与承载限制](docs/NO_PROBE_ACCEPTANCE_20261014.md)。
18/18 矩阵用例完成不代表所有负载 QoS 达标；24 UE 正常负载仍有明显溢出与超期。
原始数据在 `outputs/rt_contract_20261014/final_v7/`，源码归档与两端校验回执在
`outputs/rt_contract_20261014/local_validation_v7/`。

四篇论文控制/观测接口使用专用配置显式启用，见
[接口与运行说明](docs/PAPER_INTERFACES.md)、
[逐项文献追踪和适配差异](docs/PAPER_FIELD_TRACEABILITY.md)、
[机器可读动作/KPI目录](docs/paper_interface_catalog.json)。
固定测试动作入口为 `--control-timeline`，不包含论文决策算法。
