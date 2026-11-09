# Chongyu xApp pool

Develop new xApps in this directory. `catalog.json` registers `ue_energy_rf`
(X-01), `latency_scheduler_rf` (X-12), and `mec_offload_budget_qgbdt`
(MEC expected-workload guard), `throughput_scheduler_mlp` (X-08), and
`latency_urllc_rf` (X-12 packet-latency adaptation).
Compact KPI/RCP metadata:
[X-01](x01.json), [X-12](x12.json), [MEC budget](mec_budget.json), [X-08](x08.json),
[X-12 URLLC](x12_urllc.json). Registration leaves them OFF in native baseline runs. It never falls
back to a Yisong implementation. Pools are selected at initialization;
restart the simulation after changing the catalog or switching pools.

The [scheduler conflict study](../../../../docs/rt_sys/SCHEDULER_CONFLICT_STUDY.md)
compares frozen X-08 and the packet-level X-12 adaptation with complete-action
random arbitration.

An xApp should subclass `ric.near_rt.pools.base.PoolXApp`. The shared base supplies
cell-scoped selection, identity, contract and descriptor plumbing. Implement the
algorithm's `run(observation, policy_snapshot)` and validate its parameters.
Consume the normal delivered observation, not simulator state or PHY truth.
Return `ControlDecision` with `ConfigAction` entries; each action must use
`source_name=self.name`, an allowed `target_cell_id`, and
`action._library_provenance = self.identity(cell_id)`. The normal arbitration,
delayed actuation and withdrawal path remains responsible for execution.

For example, create `my_xapp.py` with this interface (fill in the algorithm
before registering it):

```python
from ric.near_rt.pools.base import PoolXApp

class MyXApp(PoolXApp):
    implementation_id = 'my_xapp'
    category_id = 'my_scheduler'
    description = 'Describe the implemented controller'
    required_kpis = ('queue_bytes',)
    writable_actions = ('scheduler_mode',)

    def run(self, observation, policy_snapshot):
        raise NotImplementedError('Implement the control algorithm here')
```

After implementation, add its explicit entry to `catalog.json`:

```json
{
  "my_xapp": {
    "factory": "ric.near_rt.pools.chongyu.my_xapp:MyXApp",
    "category_id": "my_scheduler",
    "description": "Describe the implemented controller",
    "commands": []
  }
}
```

The class's identity/category and command declaration must match the catalog.
Factories must live in the selected pool's namespace. Unknown pools or IDs fail
explicitly. Registration is not a capability certificate: the existing runtime
KPI/action checks still determine whether the UI can enable an xApp.

```bash
# Start the native baseline and discover only this pool's registered xApps.
python experiments/run_rt_sys.py --platform dgx --scene munich --cells 3 --ues 12 \
  --slots 400 --xapp-pool chongyu --live-xapp-test \
  --output outputs/chongyu_live --live-state outputs/chongyu_live/live.json

# After registering an implementation, optionally start with it ON.
# Add: --xapp-test-app my_xapp --xapp-test-start on
```

On AGX, use its activation environment and `--platform agx`. The launcher
`bash scripts/orin/live.sh --xapp-pool chongyu` passes the same selection through.
The normal default is OFF. An empty pool can run OFF but cannot start ON or run
a matched ON/OFF comparison until a compatible implementation is registered.

For programmatic initialization, use `ric_config['xapp_pool'] = 'chongyu'` with
`type: pool_xapp` entries containing `implementation_id`, `instance_id`, and
`cell_ids`. No process-global registry mutation is needed.

## X-01 online RF

See `docs/rt_sys/ONLINE_UE_ENERGY.md`. Use `--energy-xapp-mode online` to collect
executed-action outcomes and refit small RF batches; use `--energy-xapp-mode frozen
--energy-xapp-model outputs/.../model.json` for deployment without learning.
Training requires the UE service models and 100 ms task-cohort telemetry,
automatically configured by the finite runner with the six-UE services preset.
The accepted algorithm controls UE0/UE4; it never changes UE2/UE6 DRX.

## X-12 online RF

`--latency-xapp-mode online|frozen --latency-xapp-model outputs/.../model.json`
selects X-12. Combine it with the X-01 flags to activate both disjoint control
categories. Paper `w5G` maps to `ue_scheduler_weights`, a priority multiplier
for each target UE, preserved through NR TB resource packing. The action set
also includes native round-robin with weight one. See
`docs/rt_sys/LATENCY_ENERGY_STUDY.md` for measured online fitting and the
OFF/X01/X12/BOTH indirect-conflict experiment.

## MEC budget QGBDT

`mec_offload_budget_qgbdt` controls only UE4's `ue_offload_fraction`. It is
inspired by paper X-13's lower-quantile utility model; routing a task locally
is different from paper `alpha_adm` server-side rejection. The known workload
filter is analytic: 20 tasks/s times .05 GFLOP/task times p must be at most
.75 GFLOP/s. Thus p=0 and p=.5 are feasible. Actual random route counts are
reported separately and do not define a hard per-task admission quota.

Parameters are `mode` (`online`, `frozen`, or the analytic `rules` ablation),
`model_path`, `log_path`, `window_ms=100`, `seed=137`, `budget_gflops=.75`,
and `task_profile` copied from validated manifest UE4 metadata with exactly
`interval_ms=50`, `compute_gflop=.05`, `deadline_ms=100`. Missing profile data
leaves the app not evaluable. Artifacts/logs must be under project outputs.

Online collection balances real executed p=0/.5 outcomes until at least 12
rows with six per action. It fits a fixed 32-tree depth-two quantile GBDT
(alpha=.2, learning rate=.05, leaf minimum three), then refits every four
new rows, capped at 24. Untrained rows are also saved for bounded continuation.
Only future mature cohorts with confirmed executed offload policy and UID
ownership yield labels. Utility includes all local/offloaded generated tasks,
including failed and overdue pending tasks. DRX ownership is not required.
Frozen inference never adds rows, fits, or rewrites its artifact.

Rules-only cold start/ablation is explicitly marked. Invalid input emits no
new action and reports not evaluable; a persistent lease is not rolled back
by an empty decision. The lower quantile is a ranking score, not a calibrated
safety bound. Compare the analytic largest-feasible-p ablation before
claiming any learning benefit. Model persistence stores actual fit settings,
measured rows, initial prediction, and additive tree arrays without pickle.
See `docs/rt_sys/cases/mec_offload_budget_qgbdt.json` for the approved case.

## X-08 throughput MLP

`throughput_scheduler_mlp` is a measured-data `MLPRegressor`, not an RF alias.
A single global decision controls the six-UE network. It uses the existing
`latency_aware` base score and NR weighted packing as the native `w5G` adapter;
this is not a new throughput MAC and changes no capacity, PHY, or MCS model.
Four fixed arms are native RR/all-one weights, weighted-family/all-one weights,
and boosts of 1.5 or 2 for each cell's backlogged eMBB/non-URLLC UE with the
highest public SINR (fall back to any backlogged UE). Others remain at one.
A requested weighted arm can be effectively neutral when no UE is backlogged;
this is logged separately. MU members may still share one rectangle.

Parameters: `mode=online|frozen|fixed`, `fixed_action=0|1|2|3`,
`policy=throughput|guarded` (default `throughput`), `window_ms=100|200`,
`seed=173`, `model_path`, and `log_path`. Fixed arm 1
is the scheduler-family ablation; fixed arm 3 is the efficiency-rule ablation.
The native baseline leaves the app OFF. All artifacts stay under outputs.

Labels are whole-network cumulative delivered-bit differences over an actual
common 100 ms interval after every cell/UID executes the held snapshot, with
20 ms wash-in. Six per-UE goodputs and UE2/UE6 conservative failure-event rates
are independent outputs. Failure-event rates sum actual deadline misses, late
completions, and overflow per second; they can double-count and are not unique
packet probabilities. Original 10 ms URLLC deadlines are never relaxed.
Matched OFF delivery and unique matured-packet deadline checks report the
QoS consequences independently from the throughput objective. Optional
`guarded` candidate filtering uses measured RR training references as heuristics,
not a measured deployment counterfactual or calibrated safety bound. Bad
outcomes remain training rows; missing, mixed, reset, or spliced inputs do not.

The fixed small MLP has one eight-unit tanh hidden layer, feature and target
standardization, L2 regularization and a bounded deterministic LBFGS fit. It
needs 24 actual network windows, six per arm, refits every four new rows, and
stops adding data at 1024. The capacity was expanded from the pilot's 48-row
limit to support longer measured datasets, including the released 254-row
checkpoint. Untrained measured rows persist for bounded resume.
Frozen mode never adds rows, fits, or rewrites its artifact. Plain JSON records
actual weights, biases, scalers, fit diagnostics, schemas, and measured rows.
A weak native-performance verdict must not be turned into parameter tuning.

Training checks the installed multi-output loss and gradient with finite
differences. A project-local correction preserves the mean-output objective
and L2 term when required; installed packages stay unchanged. V2 artifacts
record the numerical proof. Legacy fits require a verified refit before frozen
deployment; their measured rows and original files remain available for audit.

The optional guarded-policy proxy allows a fixed numerical tolerance of 0.001 event per
measurement window: 0.01 event/s at 100 ms, or 0.005 event/s at 200 ms. This
is one-thousandth of the observed one-event resolution, not a risk allowance
or a calibrated SLA probability. Raw forecasts, reference values, tolerance
and rejected arms are logged; solver nonconvergence is also disclosed.
The cumulative failure-count input grows with run length. The model is an
experimental short-horizon controller; a longer deployment can be outside
its observed training distribution and has no performance guarantee.

With optional `guarded` policy, during online collection before six actual
windows per arm, a bad weighted
window is retained and requires one actually measured RR cooldown. Collection
then resumes balanced exploration even if that RR window is also bad: a bad
baseline cannot establish that untried arms caused it. Cooldown and release
events are logged. This bootstrap rule does not change frozen or post-coverage
guards, fixed policies, SLA thresholds, labels, or the MLP optimizer.

X-08's default `policy` is `throughput`, following the throughput-first
conflict experiment objective: the same frozen nine-head MLP
selects the highest predicted network goodput with the same 1% RR-abstention
margin, while per-UE delivery and URLLC constraints are inactive. All packet
and QoS outcomes remain recorded, including poor outcomes; this mode makes
no safety or SLA-compliance claim. Its incomplete online bootstrap balances
measured arms without a QoS cooldown. Frozen learning remains disabled.
Fixed-policy ablations are unchanged and apply no learned constraints. Every
decision and summary records `policy` and `constraints_active`; bypassed
guard warnings remain visible. The model, labels, PHY and deadlines are shared
unchanged between policies, so no retraining is needed to compare them.

Select `--throughput-xapp-policy guarded` for the optional protected control.
The default `throughput` policy keeps collateral KPI losses visible for intent
orchestration and conflict evaluation.


## URLLC QoS RF scenarios

The dedicated `scripts/orin/run-urllc-scenario.sh` entry point defaults to
equal-mean burst pressure, `traffic-load=2`, and the opt-in `qos_v2` policy
version. Controllers remain off unless explicitly enabled. The general
`run-six-ue-cuda-nr.sh` / `run_rt_sys.py` entry points keep native traffic.

```bash
bash scripts/orin/run-urllc-scenario.sh --urllc-scenario burst --output outputs/urllc-pressure
bash scripts/orin/run-urllc-scenario.sh --urllc-scenario native --output outputs/urllc-native
bash scripts/orin/run-urllc-scenario.sh --urllc-scenario alternating --output outputs/urllc-alternating
```

`burst` changes UE2/UE6 arrival timing to 20 ms ON / 60 ms OFF, fourfold
peak. Every complete 80 ms cycle preserves native integer byte means,
packet counts and payload sizes (175 / 137 B). Both original 10 ms deadlines
and radio resources stay unchanged. Named pressure presets reject a different
native mean load rather than silently changing its average offered traffic.

`alternating` repeats 0.8 s native CBR / 0.8 s burst inside one run.
Queues, packet IDs, HARQ, mobility, channel and controller state continue
across phase boundaries. Slot 1 is phase-local slot 0; each burst phase
starts with a complete ON segment. The phase schedule is experiment metadata,
not an RF feature. The 3.2 s default covers two full alternating cycles.

Add `--urllc-xapp-mode frozen --urllc-xapp-model outputs/.../model.json`
for the trained RF; fixed ablations use `--urllc-xapp-mode fixed
--urllc-xapp-fixed-action 1|2|3`. The complete study defaults to evaluating
burst, native and alternating scenarios. All arms, including RR, use the same
scenario, physical seed and per-slot external arrivals. Use
`--eval-scenarios burst native --eval-seconds 1.6` for the original two-scenario
evaluation. The automatic HTML and raw packet-cohort audits remain enabled.
No result from the previous static scenarios establishes performance in the
new alternating scenario.
