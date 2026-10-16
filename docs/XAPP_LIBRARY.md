# Three-level training-free xApp library

The opt-in `xapp_library` format separates **12 functional categories**, **28
versioned implementations** and independently named **runtime instances**. The
old `rule_based_v1` format remains a compatibility path, not 12 additional
algorithms. A read-only KPI monitor is not counted as a control implementation.

## Implementations

Each category retains `legacy_v1`, the prior measured-feedback implementation.
New methods are local deterministic functional baselines, not claims of an
upstream algorithm reproduction or of published performance. There is no model
training, pretrained predictor, BLER table, extra reference decode or online
Monte Carlo probe. See `xapp_library_catalog.json` for goals, sources, actions,
required KPIs, defaults and the version-specific evidence requirement.

| Category | Additional implementations | Total |
|---|---|---:|
| throughput_power | cqi_proportional, backlog_cqi | 3 |
| rf_energy | sinr_pi | 2 |
| cell_sleep | empty_only | 2 |
| load_balancing | load_pi | 2 |
| traffic_steering | a3_rsrp, sla_max_sinr | 3 |
| mobility_robustness | window_search | 2 |
| ue_fairness | equal_share, deficit_airtime | 3 |
| slice_sla | goodput_pi, service_deficit | 3 |
| latency_guard | delay_pi | 2 |
| interference_coordination | adaptive_sfr | 2 |
| ret_coverage | coverage_grid | 2 |
| beam_management | max_min_reference | 2 |

IDs are `category:algorithm`, with an independently recorded implementation
version. Instance IDs are deployment identities; they are never inferred by
splitting the name. The example presets use `category__algorithm` as instance
IDs simply for readability.

The new frequency-power methods normalize CQI or backlog×CQI scores to the
executed scalar/cap budget. PI uses executed actuator values, velocity-form
proportional/integral updates, elapsed report time, bounded changes and saturation
protection. Configured defaults are kp=0.2 and ki=0.1/s; RF/CIO changes retain the
existing step bounds, slice/latency changes are bounded by 0.25 per update. These
are explicit engineering defaults, not tuned performance claims. Equal-share
caps rotate the integer remainder. Airtime deficits debit measured **actual
transmitted PRB×slot**, while service deficits debit actual delivered bits.
Caps are not scheduling guarantees, and slice weights are not hard isolation.

A3-like selection uses delivered RSRP, configured CIO/hysteresis and continuous
produced-report time for TTT. Changes to CIO/HYS/TTT/neighbors restart that
continuity timer, and invalid/missing reports break continuity; command cooldown
history is retained. SINR steering retains SLA checks and repeated-report
guards. Neither predicts unobserved target throughput. Empty-only sleep never
migrates UEs; coverage checks, dwell and wake protection still apply.

MRO window search scores completed events from an executed candidate window,
including recently departed UEs; it does not interpret every failure as an
early/late handover. RET evaluates the grid 0/3/6/9/12/15 degrees, accumulating
delivered reference samples for at least 400 ms after execution. Other relevant
configuration changes invalidate comparison windows. No search asks the PHY to
simulate an extra transmission. Search methods may be inconclusive under moving
or changing conditions and do not guarantee an optimum.

## Instances and selection

`XAppInstance(instance_id, implementation_id, cell_ids, parameters, priority)`
owns isolated per-cell decision engines. `run(observation, policy_snapshot)`
is unchanged. Unselected instances do not execute decisions or shadow policies.
Read-only managed-cluster context is separate from action authority: a source
cell instance can observe neighboring cells and request handover to an allowed
neighbor without gaining power/PRB control of that neighbor.

Example configuration section (inside a normal RT/SYS JSON config):

```json
{
  "xapp_library": {
    "enabled": true,
    "monitor": true,
    "instances": [
      {"instance_id": "power_step", "implementation_id": "rf_energy:legacy_v1"},
      {"instance_id": "power_pi", "implementation_id": "rf_energy:sinr_pi", "parameters": {"pi_kp": 0.2}}
    ],
    "initial_selection": {"rf_energy": "power_step"}
  }
}
```

`cell_ids` on an instance restricts its source-cell authority; omission means
all cells in the selected manifest. `initial_selection` expands across those
managed cells. Alternatively provide an explicit `bindings` map. The full-pool
preset registers all 28 but selects only one legacy implementation per category
per cell. Registering overlapping same-category instances without an explicit
selection fails; there is no arbitrary priority-based choice within a category.

Non-RT uses `PolicyUpdate.xapp_bindings`:

```python
PolicyUpdate(xapp_bindings={
    'cell_0': {'rf_energy': 'power_pi'},
    'cell_1': {'rf_energy': 'power_step'},
})
```

This is a **complete replacement** for the new-library selection. Missing
categories/cells are disabled, `{}` disables all new-library control, and `None`
means no replacement. Unknown instances, category/scope mismatches, unavailable
capabilities or invalid instance parameters reject the entire update before
mutating the snapshot. Parameters follow defaults → instance → non-RT instance
override; non-RT targets instance IDs through `rule_based_parameters`.
Using legacy enabled/disabled lists to address a new instance is explicitly
rejected; it cannot silently override or bypass the binding table.

Binding changes take effect atomically at the next 5-slot near-RT invocation.
Per-cell/category generation tokens invalidate old queued actions, including an
old incarnation of an instance later re-enabled with the same ID. A partial
switch leaves other cells' history/tokens untouched. Replacement holds executed
configuration until the next valid decision, rather than restoring startup
values. Explicit disable uses existing lease withdrawal and other-category
re-arbitration. Already started handovers and HARQ/packet lifecycles continue.
Different categories can still conflict and use the existing field arbitration
and physical transaction checks. There is no automatic selection rApp here.

Category, algorithm and registered-instance descriptors are exposed to non-RT
alongside the existing capability surface. Selection admission/application and
rejection records go to `xapp_selection.jsonl`; decisions and executed receipts
include instance/implementation/category/version/generation identity, with field
ownership for merged configurations. No private channel state is published.

## Executed-resource observations

The opt-in public producer adds cumulative actual transmitted PRB-slots, an
epoch that changes with the serving cell, the epoch start, rolling-window
PRB-slots, window bounds/sample count and total delivered bits. They traverse
the normal report queue and independent projection. Cumulative counters are not
mislabelled as per-window samples. Deficit methods reset their per-UE accounting
on an epoch/cohort change and never debit an unexecuted requested PRB cap.

## Validation and operation

Use the individual/combination examples in `configs/rt_sys/xapp_library/` and
`--control-mode orchestrated`. `--policy-timeline` loads fixed non-RT stimuli
aligned with the existing 10-slot rApp period; this is acceptance tooling, not
a KPI-based selector. Config/command stimuli still use `--control-timeline`.

`scripts/validation/validate_xapp_library.py` performs paired physical response,
legacy migration, combination, resource-pressure and selection lifecycle gates.
Missing triggers or an issued-but-not-executed request are not passes. Most
response fixtures use 128 slots; sleep, mobility and MRO fixtures are explicitly
longer because their unchanged windows cannot execute within 64 ms. The MRO
fixture generates real completed handover/link-loss events; it increases target
confirmation time to exercise the failure path and may pause ordinary data
resources after initial real transmissions, without fabricating NACKs or
injecting event records. It is not a deployment/capacity result.

All scientific data/cache/logs stay on Spark. A fresh 8-slot cold-cache timing
gate precedes parallel functional tests; concurrent case times are not isolated
performance benchmarks. Long estimates separate compute, unique cold channel
keys and disk cost. The long driver refuses an incomplete/failed response gate
or a different source fingerprint, stops on severe accounting failures and
archives exact sources and SHA256 inventories. After a successful short gate,
launch the long script detached, record PID/log/exit status, report the estimate,
and stop agent polling until the user requests result inspection.

Version-specific Spark acceptance is recorded separately; presence in this
catalog or a completed process alone never asserts every implementation passed.
