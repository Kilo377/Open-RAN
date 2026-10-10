# Nine-xApp Algorithm and Tuning Alignment

The paper evaluation keeps nine Near-RT xApp categories.  Each xApp exposes
bounded, typed tuning fields so Non-RT guidance can constrain the Near-RT
control surface without placing an LLM in the fast path.

## 1. Traffic Steering xApp

- Role: load steering, hotspot relief, cell-edge rescue.
- Inputs: per-cell load, cell-edge UE ratio, interference penalty, mobility risk,
  Non-RT intent weights.
- Algorithm: multi-factor steering pressure score instead of a single overload
  threshold.
- Tunable fields:
  - `steering_aggressiveness`
  - `load_threshold`
  - `cio_bias_db`
  - `mobility_bias_db`
  - `measurement_event_offset_db`

## 2. Mobility Predictor / Robustness xApp

- Role: mobility robustness, handover risk control, ping-pong mitigation.
- Inputs: candidate-cell gain, measurement age, handover pending state, beam
  risk, future CQI, channel confidence.
- Algorithm: predictive robustness score that trades early handover against
  stable retention under low confidence.
- Tunable fields:
  - `steering_hysteresis_db`
  - `time_to_trigger_ms`
  - `mobility_margin_db`
  - `mobility_bias_db`
  - `measurement_event_offset_db`

## 3. Power Saving xApp

- Role: energy saving without breaking latency or queue constraints.
- Inputs: least-loaded cell, queue headroom, energy intent, latency guard.
- Algorithm: energy headroom score with mild/aggressive power-saving modes.
- Tunable fields:
  - `power_cap_dbm`
  - `beam_candidate_preference`
  - `sleep_threshold`
  - `wake_up_threshold`
  - `dwell_time_slots`

## 4. Spectrum Optimizer / Interference Coordination xApp

- Role: PRB efficiency, interference coordination, resource packing.
- Inputs: cell utilization, inter-cell interference, dominant slice pressure.
- Algorithm: coordination pressure score selects throughput-dense,
  moderate-pressure, or interference-aware mode.
- Tunable fields:
  - `scheduler_policy_hint`
  - `slice_weight_adjustment`
  - `power_offset_db`
  - `muting_ratio`
  - `coordination_threshold`

## 5. Latency Scheduler xApp

- Role: QoS scheduling for URLLC and tail-latency protection.
- Inputs: total queue, URLLC UE set, HOL delay, HARQ urgency, BLER pressure,
  slice weights.
- Algorithm: QoS pressure score tunes scheduling bias while preserving explicit
  Non-RT slice weights when they are already supplied.
- Tunable fields:
  - `latency_bias`
  - `slice_weight_adjustment`
  - `prb_weight`
  - `priority_coefficient`
  - `delay_threshold_ms`

## 6. Massive-MIMO Beamformer xApp

- Role: coverage/capacity beam control and spatial gain tuning.
- Inputs: beam misalignment risk, post-equalization SINR, precoder match, cell
  beam stress.
- Algorithm: switches between coverage rescue, capacity push, and balanced beam
  modes with explicit power and coordination hints.
- Tunable fields:
  - `beam_candidate_preference`
  - `power_cap_dbm`
  - `power_offset_db`
  - `coordination_threshold`

## 7. RAN Slicing Manager xApp

- Role: slice SLA protection for eMBB, URLLC, and mMTC traffic.
- Inputs: per-slice pressure, dominant slice share, latency/reliability intent.
- Algorithm: pressure-ratio slice reweighting with URLLC guard bonuses.
- Tunable fields:
  - `scheduler_policy_hint`
  - `slice_weight_adjustment`
  - `prb_weight`
  - `priority_coefficient`

## 8. URLLC Guard xApp

- Role: hard guardrail for reliability and low-latency traffic.
- Inputs: URLLC queue, worst BLER, latency pressure, target cell.
- Algorithm: guard activation based on queue/Bler/pressure; emits explicit
  reservation and priority controls.
- Tunable fields:
  - `latency_bias`
  - `power_cap_dbm`
  - `prb_weight`
  - `priority_coefficient`
  - `delay_threshold_ms`

## 9. Cell Sleep xApp

- Role: cell sleep and load consolidation under safe low-load conditions.
- Inputs: non-primary cell load, queue, interference, energy intent, latency
  guard.
- Algorithm: guarded sleep score selects active, micro, light, or deep sleep.
- Tunable fields:
  - `cell_sleep_mode`
  - `power_cap_dbm`
  - `sleep_threshold`
  - `wake_up_threshold`
  - `dwell_time_slots`

## Validation

The following checks passed after the update:

```text
python -m unittest tests.test_sensitivity_xapps tests.test_controller_control_surface tests.test_control_surface_executor tests.test_safe_goal_parameter_tuning tests.test_xapp_selection_variants
Ran 33 tests OK
```
