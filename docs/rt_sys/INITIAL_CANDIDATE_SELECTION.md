# Initial xApp combination selection

`semantic_influence_coverage_v2` adds **xApp capability-direction coverage** to
the first-stage influence screen. It does not read the current proposed action,
infer policy outputs, use parameter-gain signs as policy directions, or call
the conflict scorer. V1 and exhaustive enumeration remain selectable.
The active service then applies the recorded-action exclusion rule documented
below; graph capability direction and proposed parameter-step direction remain
separate stages.

For one-sided intent ranges, a lower bound requires an increasing capability;
an upper bound requires a decreasing/upper-limiting capability. This direction
does not flip just because the latest observation already satisfies the range.
A bounded band admits either control direction provisionally; checking actual
values within the band belongs to the later evaluation. All requested KPIs
must have an unlocked graph path from at least one supporting xApp.

Primary capability directions come from canonical released `target_kpis`, not
the app name, the KPI's generic preferred direction, or every affected-KPI edge.
Thus X-02 supports decreasing BS power, not increasing throughput merely
because sleep has a large throughput effect. An opposing extra member does
not veto a combination that already has support for all directions. Shared
and distinct meaningful control strategies remain protected for later conflict
analysis, including potential costs and compensating policies.

Strength coverage/scales in V2 use only direction-supporting sources for each
KPI. Raw graph magnitudes still describe control-strategy diversity. Unknown
directions go to reserve; a globally unsupported intent does **not** restore
all groups as primary through an exhaustive fallback. Missing gain magnitudes
on an otherwise declared supported capability remain explicitly unquantified.
The graph, trained controllers, gain estimates and conflict method are unchanged.

Validated indirect/secondary capabilities can be supplied by the operator with
`capability_catalog` (absolute JSON path; CLI `--capability-catalog`). This is
an extension interface, not a hardcoded list of six apps. The schema is
`oran.xapp-capabilities.v1` with `estimates` rows containing:

- `app_id`, canonical `kpi`, `direction` (`increase` or `decrease`);
- matching `model_sha256`, `runtime_parameters`, `metric_scope`, `scenario_id`;
- `kind: matched_frozen_policy_on_off`, at least two unique paired seed samples
  (`seed`, `before`, `after`) and a direction-resolving `confidence_interval`;
- `source: {path: absolute_path, sha256: audited_file_hash}`.

Mismatched/uncertain evidence does not add a capability. Parameter perturbation
rows cannot be substituted for frozen-policy ON/OFF evidence. Conflicting
capability evidence becomes unknown rather than a claim of bidirectional
optimization. No secondary records have been promoted from the current batch
of validation labels; the shipping default uses released objective directions.
This is not a guarantee of per-window or population-wide improvement.

Enable with `--candidate-selection semantic_influence_coverage_v2`, or set that
method in the scenario manifest's `candidate_selection` object. Threshold 0.05
and marginal tolerance 0.05 remain the conservative defaults. A new catalog
does not rewrite a previously frozen batch or its measurements.

---

`smo/candidate_selection.py` is the replaceable first stage. It consumes a
resolved intent and the existing graph, and returns candidates for later
conflict quantification/evaluation. It does not infer model actions, change
graph coefficients, score conflicts, select a winner or launch simulations.

`reverse_graph` preserves the existing exhaustive catalog exactly and remains
the service default. `influence_coverage_v1` splits it into `groups` (primary)
and `reserve_groups`. Together they contain every original group, with stable
IDs and unchanged conflict annotations. Only the primary pool proceeds through
normal review and explicit evaluation when this method is enabled. Returning
to `reverse_graph` restores exhaustive review/evaluation.

For each xApp and requested KPI, take the maximum absolute normalized influence
across its writable graph paths and tested contexts within the selected scene.
Normalize this capability by the strongest candidate for that same KPI. The
combination's coverage is the maximum capability of its members, independently
for every requested KPI. Paths are not summed or interpreted as action outputs.

A group with coverage below the threshold for any requested KPI enters reserve.
For the others, a proper subset can be its primary-pool witness when every KPI
loses at most `marginal_tolerance` and removing members loses no meaningful
distinct control or shared-control trained policy. Shared RCPs do not make
different trained controllers equivalent. Meaningful strategies are those with
a path at or above the coverage threshold. Unquantified paths/zero-capability
reference KPIs are retained conservatively; uncertain finite estimates remain
quantified. An empty primary pool restores the exhaustive catalog.

This is a heuristic about statistical influence capability, not feasibility.
It does not use gain signs, intent direction or numerical bound distance to
forecast attainment. A small required improvement can be achieved by a weak
path; conversely, a large absolute influence can move a KPI the wrong way.
All reserve groups remain recoverable for that reason.

The conservative experimental defaults are threshold **0.05** and marginal
tolerance **0.05**. A 0.05 threshold means 5% of the strongest candidate's
influence on each KPI, not a 5% KPI improvement or confidence level. On the
retrospective trial this retained all 28 known passing records, while 0.10
moved four of those records to reserve. These short historical records do not
establish generalization to unseen scenes or controller versions.

Python:

```python
from smo.candidate_selection import select_combinations

catalog = select_combinations(graph, frame, resolved_intent, settings={
    "method": "influence_coverage_v1",
    "coverage_threshold": 0.05,
    "marginal_tolerance": 0.05,
})
primary, reserve = catalog["groups"], catalog["reserve_groups"]
```

Service CLI: add `--candidate-selection influence_coverage_v1` to `python -m smo`.
To compare settings use `--coverage-threshold 0.10 --marginal-tolerance 0.05`.
`GraphService(candidate_selection={...})` supports the same settings.
For `smo.scenario_service`, an operator manifest accepts a `candidate_selection`
object either globally or per scenario. Existing manifests keep the old method.
No preset, memo tag, intent anchor or historical evaluation needs rewriting.

Read-only retrospective benchmark:

```sh
python -m experiments.benchmark_candidate_selection \
  --analysis /path/to/context-study/analysis/final \
  --batch-root /path/to/evaluation/jobs/batches \
  --output /path/to/candidate-selection-trial
```

It evaluates seven KPI sets across three graphs at thresholds 0.05/0.10/0.20,
then scores retained historical outcomes. Outcomes never enter selection.
Missing before/after measurements are reported separately. The report includes
pool sizes, passing-record recall, whether a known passing option survives per
batch, best observed interval gap, reasons/witnesses and source hashes.

## Confirmed opposed direct actions

After the graph screen, the active service applies `opposed_direct_actions_v1`
from `smo/direct_action_filter.py`. This is a hard admission rule requested by
the operator, separate from the conflict score. Recorded, valid shadow actions
from two distinct xApps must write the same parameter on the same UE/cell and
have strictly opposite signed steps from the same observed current value.
Every combination containing such a pair is excluded from primary and reserve.
There is no empty-primary fallback and no comparison to predicted KPI outcomes.

Numeric offload fractions and UE scheduler weights are checked directly. DRX
cycle/on-duration/inactivity/wakeup timers are checked by scalar component.
Unchanged writes, different entities/parameters, or two fields of the same app
do not trigger the rule. Scheduler/sleep strategy names, Boolean switches and
cyclic DRX offsets have no declared numeric order and are not assigned invented
opposite directions. Unknown/invalid proposals, mismatched current values,
inconsistent observation slots or unsupported fields remain unconfirmed.

The service uses an exact captured decision and its own frame, or the nearest
earlier recorded decision with the lag disclosed. Future decisions are never
borrowed. An unavailable decision leaves the graph catalog unfiltered and marks
that fact explicitly. Language parsing freezes the decision context before the
LLM request. No inference, control, arbitration draw or model mutation occurs.

The same review path handles intent analysis, refresh, conflict inputs and new
single/batch deployment admission. Client-supplied actions cannot bypass it.
`excluded_groups` retains group IDs and opposing-action evidence outside the
candidate pools; the UI shows counts/reasons and list exports retain the audit.
The filter is recomputed per decision, not a permanent xApp blacklist. Existing
frozen batches, their measurements, presets and frozen intent thresholds are not
rewritten or retroactively removed.
