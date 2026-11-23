# Lightweight SMO: intent graph

This module displays the Chongyu `xApp → RCP → KPI` graph and reads network
state. Matched fixed-RCP experiments provide optional local response evidence;
intent-to-policy actuation is separate work.

The [intent review entry](SMO_INTENT_REVIEW.md) uses DeepSeek to turn natural
language into KPI intervals, then matches read-only xApp capability previews
and highlights their control/effect paths. It does not deploy controllers.

Python's standard library is sufficient. No GPU, Sionna, external font, CDN or
additional service is required.

```bash
# Contract topology without starting a simulation.
python -m smo --port 8891

# Actual recorded state, with a simulation-time playback slider.
python -m smo --replay outputs/YOUR_COMPLETED_RUN --port 8891

# Actual ongoing state. Future live runs publish compact graph telemetry.
python -m smo --state outputs/YOUR_RUN/live.json --port 8891

# A self-contained offline HTML, including recorded frames.
python -m smo --replay outputs/YOUR_COMPLETED_RUN \
  --export outputs/smo/intent-graph.html --export-only
```

Open `http://127.0.0.1:8891`. For a remote AGX, run the same command inside the
project and forward this loopback port with SSH. The service has no simulator
write/control path. Intent review accepts text and optional session-only LLM
credentials; original records and models are never modified.

## Graph semantics

Six app nodes are built from `artifacts/xapps/chongyu/20261117/release.json`
and the corresponding files under `ric/near_rt/pools/chongyu/`. Public writable
RCPs are checked against class declarations with AST parsing, without importing
or initializing the learners. The QoS-v2 contract is used for X-12-URLLC;
X-01 DRX is controllable in release 20261117; the former locked release is retained separately.

The current registry has 6 xApps, 5 RCP types and 18 KPI types. Seven target
KPIs are visible by default; **All KPIs** reveals the side effects and guards.
An app's target/affected lists provide **candidate** RCP-to-KPI edges, not
identified causal effects. Input features such as reference SINR do not
automatically become output-effect edges. Contract files and SHA-256 values
are retained in the JSON graph. Normalization is explicit in `smo/contracts.py`.
App objective direction is distinct from the unknown sign of a parameter effect.

Click a node or link to inspect its scope, actual values and evidence. Focus
an xApp to highlight just its declared paths. **Export graph** downloads the
currently selected graph snapshot as JSON.

xApp names state their objective with Reduce, Increase or Limit. KPI cards
show a reference target direction (lower, higher or bounded); a parsed intent
replaces this hint with its requested direction or interval. These arrows are
not observed time trends. BS means base station. BS power consumption is the
sum of base-station power averaged over the measurement window, in W; energy
consumed over a fixed interval is in J. Network DL throughput remains delivered
user-data goodput. Display names do not change metric keys, units, population
scopes, controllers or the measured influence matrix.

## Time and measurement semantics

- App nodes distinguish enabled state, current-generation suggestions,
  historical suggestions and last actual applied receipts. A suggestion that
  lost arbitration is never displayed as an executed parameter.
- The default view treats UEs as one population. Total energy includes every
  UE; task metrics include the complete configured MEC workload; URLLC metrics
  cover all configured URLLC traffic. Task/packet latency and success ratios
  pool event counts, rather than averaging user means. Mean UE goodput includes
  zero-goodput users. Expected/realized MEC demand sums all task sources.
- RCPs show a common value, range or policy distribution. DRX summarizes the
  enabled population and its timer profiles. Individual UE tables and IDs are
  omitted from the UI; original entity maps stay in exported evidence.
- Throughput, AP power and URLLC event rates use actual counter deltas over a
  rolling 100 ms simulation window. The first window can be shorter; actual
  boundaries are exported. An unavailable rate stays unavailable.
- Completion latency is weighted by completion counts, not averaged across
  per-UE means. URLLC failure events include disjoint overflows and carryover;
  they are not mature arrival-cohort SLA probabilities.
- UE energy and task performance are cumulative, clearly labelled in details.
  Expected MEC demand sums configured task rate × task compute × **executed**
  offload probability across the workload. Realized demand uses task counters.
- The graph displays **execution telemetry**, which can include simulator
  counters. It is not the delayed/noisy public E2 observation used by xApps.
  The backend never feeds these values into a controller.
- Old live files without `graph_telemetry` still show their available radio,
  packet and control state. Missing service fields remain unavailable.

Completed traces are streamed once and reduced to bounded UI frames. Receipt
and suggestion streams are merged in timestamp order without loading all raw
records into memory. Live views read atomic `live.json` without rescanning
large traces. All sampling is for visualization; source data are kept intact.

## Parameter influence estimates

`--influences estimates.json` imports measured results using
`schema: "oran.intervention-edges.v1"` and an `estimates` list. No fitted
strengths are shipped by default. Each result requires:

| Field | Meaning |
|---|---|
| `parameter`, `kpi` | Canonical graph keys |
| `estimate`, `unit` | Finite local effect and its units |
| `kind` | `local_derivative`, `range_linear_fit` or `finite_contrast` |
| `sample_count`, `horizon_ms` | Number of samples and response horizon |
| `scope`, `context` | Affected entities and operating conditions |
| `method`, `data_sha256` | Estimation procedure and source-data hash |
| `design` | `controlled_perturbation` |
| `from`, `to` | Required for a finite contrast |
| `confidence_interval` | Optional finite interval containing the estimate |

Multiple `contrast_id` values retain separate contexts. The overview considers
all measured changes: a solid P–K link retains finite calibrated responses,
including variable gains whose intervals include zero and observed zero
responses. Declared objective paths also remain visible without fabricated
numerical weights. Line existence does not certify a nonzero or stable effect.
Selecting the RCP or link exposes the per-change results and a shared
parameter context selector; no effect is averaged across transitions. A
specific action preview uses only its corresponding contrast, so a confident
QoS-deadline result cannot establish a Latency-aware effect. The default dashed
filter hides only uncalibrated candidate side-effect paths. Gain confidence is
metadata, not a topology deletion rule. Diagram values are unsigned normalized
magnitudes; signed weights and intervals remain in Details and the risk engine.
`graph_completeness` in the service bundle audits all declared xApp objectives
through real xApp→RCP→KPI paths and reports any isolated visible nodes.

Population estimates
must carry `metric_scope: "ue_population.v1"` and
`population_control_eligible: true`. The default service rejects legacy subset
coefficients and single-member probes. A single-member intervention cannot be
renamed into a population-wide parameter derivative.

Recompute population outcomes from existing preserved traces with:

```bash
python -m smo.identify --view population --study outputs/rcp-pilot \
  --output outputs/population-identification
python -m smo --replay outputs/rcp-pilot/cases/s20261116_native \
  --influences outputs/population-identification/influences.json
```

Only the group-wide scheduler and AP-policy contrasts from the historical
pilot are promoted to population-control evidence. Single-member probes stay
in the complete matrix and raw records. The dedicated
[group-wide study](POPULATION_RCP_PERTURBATION.md) fills offload, DRX and
service-class priority responses without relabelling the original probes.
The [multi-point extension](MULTIPOINT_RCP_IDENTIFICATION.md) covers declared
numerical domains with 7–8 levels and three independent seeds. A range fit
adds sampled/valid domains, the measured grid, seed slopes and residual
diagnostics. The inspector plots actual points against the first-order line.
Direction uncertainty and the adequacy of a straight-line approximation are
reported separately.
The legacy view, original subset KPIs and
per-UE vector evidence remain available explicitly with `--view entity`.
Population monitoring does not retrain xApps or change their deployed targets.

The [small controlled-perturbation study](RCP_PERTURBATION.md) produces this
format from matched native network runs, with raw-data and execution audits.

Categorical scheduler/sleep policies and general structured DRX actions require
named finite contrasts. An explicitly identified scalar DRX inactivity timer
can instead have a range fit while its other fields remain fixed. Coupled scheduler-mode/weight actions do not establish
independent partial derivatives. The importer checks structure/provenance
fields; experimental design and identification must be validated in the study
that produces the file. A local estimate is not a global causal law.

## Validation

```bash
python -m pytest tests/test_smo_graph.py tests/test_smo_population.py tests/test_rt_live.py -q
```

Tests cover six-release contract alignment, DRX action-space alignment and QoS versions,
candidate-vs-measured evidence, finite contrasts, counter-delta units,
completion-weighted latency, missing data, rejected actions, same-tick OFF/ON,
nonmutating compact telemetry, streaming replay, live updates, read-only HTTP
and self-contained HTML export. Analytic effect estimates in tests are labelled
fixtures and are not added to the product graph.

## Significant operational P–K projection

The current scenario manifest sets `influence_selection: significant_only`.
`smo/statistical_filter.py` retains scalar, population-eligible estimates whose
confidence interval is valid, meets `influence_confidence_level` (0.90 or 0.95,
default 0.95), and excludes zero. The current exploratory UI uses 0.90 uniformly
across scenarios and KPIs. Lowering the confidence level narrows the interval
and relaxes admission; it does not increase the estimated statistical effect.
This is a pointwise interval screen, with no multiple-comparison correction or
new fit. Observed-zero estimates, intervals
containing zero, uncalibrated paths and unverified confidence are excluded.

For 90% screening, a recorded 95% Student interval is recomputed as estimate ±
t(0.95, n−1) × standard error, using independent seed blocks/pairs (not slots).
The source interval must agree with the recorded standard error, seed count
and 95% critical value. Otherwise the original stricter interval remains in
force. Derived rows retain the source confidence level/interval and method;
raw calibration files and point estimates remain unchanged. Both the inspector
and canonical-unit projection carry the effective interval and actual level.

Filtering is per action/context estimate. A mixed P–K edge retains only its
significant estimates; a path with none is removed from the active graph.
Control-contract edges and KPI nodes remain intact. The identical projected
graph supplies visualization, reverse paths, candidate selection and current
risk scoring. Declared objectives and Show dashed cannot restore excluded
relations. Original calibration files, frozen intents and simulation results
are not edited. Read-only exclusion metadata is exported and shown separately.

Risk is evaluated over this explicitly limited graph model. An excluded
relation supplies no term and is not claimed to have zero physical influence.
Unsupported actions on a retained conditional relation remain unavailable;
another sleep profile cannot silently substitute for an excluded contrast.
No candidate due to missing significant coverage means insufficient statistical
support, not a proof of physical infeasibility. Quantification remains independent
of actual deployment evaluations.
