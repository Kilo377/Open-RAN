# SMO intent review

The active SMO discovers trained xApps and reports conflict risks from the
knowledge graph. It does not choose hypothetical actions, invoke xApp model
inference, tune a controller, select a combination or deploy network controls.

The pipeline is: natural-language intent → resolved KPI intervals → intersection
of active constraints → related xApps → complete KPI-covering combinations →
structural conflict risks from graph paths.

The existing DeepSeek client returns only `ranges` and clarification `questions`.
Each range contains `kpi`, `lower`, `upper`, `lower_inclusive` and
`upper_inclusive`. Bounds accept a native-unit number, `minimum`, `maximum`,
`current`, null for an unbounded endpoint, or `{"current_multiplier": 1.2}`.
Increase uses an open lower endpoint at the captured current value; decrease
uses an open upper endpoint. Do-not-decrease / do-not-increase include current.
Ratios have domain [0,1]. Other registered KPIs have a nonnegative domain without
an invented finite maximum. Ambiguity and unsupported KPIs require clarification.
The LLM never selects an xApp or control value.

UE and base-station saving objectives use power in W. `ue_power_w` is the sum
of all UE energy counter differences divided by elapsed seconds over the same
observation window; `ap_power_w` already uses the corresponding BS definition.
This is total population mean power, not a per-UE mean and not cumulative energy
divided by the full run duration. Missing members, zero elapsed time and windows
straddling an energy-counter reset remain unavailable. Cumulative `ue_energy_j`
is retained in raw telemetry and All KPIs as history, not an actionable saving
target. The native language catalog excludes that historical joule key and
explicitly interprets generic saving requests as window-average power. An
explicit joule budget is not silently reinterpreted as watts.

Old saved energy intents keep their original J bounds and receive
`invalidated_kpis=['ue_energy_j']` plus a blocking `constraint_issues` entry tied
to the original intent IDs. `questions` contains only clarification of the new
language input. Empty joint intervals use the same record-specific issue channel.
The UI marks the affected records and pauses joint planning separately from
current-input feedback. Old energy constraints cannot produce
a new combination plan until deactivated/reparsed; old numerical baselines are
never relabelled as W. New power intents freeze their actual power reading and
its measurement-window metadata. Released controller models are unchanged:
X-01 already trained against energy-counter delta divided by window duration.

Power influence coefficients are generated into a separate retained artifact
directory by `smo.power_influences.derive_power_influences`. The original study's
archived analysis source/hash and dataset prove a 200–700 ms energy-counter
difference. New power rows divide all energy-valued statistics, CIs, seed/grid
observations and fit residuals by 0.5 s. The response horizon or full run duration
is not used as a substitute. Original energy rows and files remain unchanged;
canonical source/dataset/analysis fingerprints are recorded in `derived_from`.
The 500 ms calibration context remains explicit and is not treated as an exact
100 ms live-window response.

The language catalog includes aggregation and task population definitions.
`mec_offloaded_latency_ms` is E2E completion delay of offloaded MEC tasks only;
`task_latency_ms` includes local plus offloaded completions. Both are E2E metrics.
The UI alias `MEC task delay` (including `Reduce MEC task delay` and
`降低MEC task delay`) refers to the offloaded-only KPI. An explicit request for
all tasks including local execution takes precedence and retains the all-task
KPI. The LLM receives these aliases and examples; its returned keys are not
rewritten by keyword rules. Saved intent records retain their original parsing
version, population and frozen anchors.
The legacy loopback language relay provides the same versioned semantic glossary
for every request without rewriting returned KPI keys. It preserves the original
text and semantic-context hashes. If the original text plus glossary exceeds
the legacy broker's 2,000-character limit, it asks for a shorter intent rather
than truncating either part.

Each accepted intent gets an immutable record containing text, its original
symbolic request, resolved numeric bounds, captured KPI baseline and replay
slot. New intents join the active set. Numeric ranges are intersected separately
for each KPI, preserving open/closed endpoints. Increasing from 20 Mbit/s and
later decreasing from 30 Mbit/s permits (20,30) while both constraints are active.
Opposite direction words alone do not establish a contradiction. Strictly
coincident bounds give an empty interval; two inclusive bounds can leave a point.

The UI shows a KPI / desired-range / captured-baseline table for each parsed
intent and a separate common-range table. Update observation reassesses the
same fixed bounds at another snapshot. Removing one intent recomputes the rest;
clear empties the active set. Browser storage retains active record IDs. The
optional `--intent-store` JSON ledger retains all records across service
restarts, including inactive or incorrectly parsed records, for audit. It is
network-context-specific and never stores the model API credential. Correcting
an interpretation creates a new record; it does not rewrite an old baseline.

Every requested KPI is reverse-traced through KPI←RCP←xApp paths with unlocked
controls. A KPI need not be the xApp's own optimization target. Merely declaring
a target without a control/influence path cannot establish coverage. Default
coverage retains measured parameter–KPI weights even with uncertain directions;
showing dashed paths additionally admits uncalibrated candidates. Measured
constant-zero side effects do not establish unrelated coverage, while an app's
declared objective path remains structural coverage without claiming attainment.
Uncalibrated declared-target paths are retained as contract relations, with
explicit evidence metadata, rather than numerical response estimates.
Maximize/minimize labels, current parameter equality and an empty/no-new-action
policy result do not remove a structurally related xApp. All nonempty subsets are
enumerated. Every displayed group must relate to every requested KPI, and every
member must relate to at least one. Single-app groups and larger covering supersets
are both retained. `owns_kpis` retains only actual declared objectives;
`related_kpis` and `relation_paths` separately explain graph contributions.
Missing relations and empty intent intersections remain explicit.
`kpi_backtrace` records the reverse index for each requested KPI. Each group
contains `coverage_paths` with witnesses for every KPI, plus required/covered
counts. The UI exposes these under “KPI 覆盖 N/N”. Every contributor in this
table belongs to that group, and each witness references real graph edges.

No action profiles, variants, hypothetical parameter values, learned-policy
predictions or action assignments participate in this active path. The current
GraphService returns `oran.intent-review.v2`, `analysis_basis=knowledge_graph`,
`hypothetical_actions_used=false`, `model_inference_invoked=false`,
`runtime_proposals_used=false`, `automatic_selection=false` and
`deployment_enabled=false`.

Direct graph risk means two controllers share an unlocked RCP and their write
scopes can overlap. Explicit disjoint member scopes are excluded, but a policy's
target UE subset cannot prove it never resets other priorities. Without actual
proposals the values are unknown: the status is `potential_unknown_values`,
not confirmed incompatible writes. Equal current RCP values do not prove that
future controller proposals agree.

Indirect graph risk means two distinct xApps have paths through different
parameters to the same requested KPI. This includes side-effect contributors,
not only declared objective owners. This is `structural_coupling`:
a graph relationship can identify a possible interaction, but cannot decide
whether an unknown action helps or harms, or establish a net joint KPI response.
Coefficient signs are not treated as chosen action signs. Tests assert that
sign reversal does not change structural coupling.

The overview uses all measured contexts without selecting or averaging a
contrast. Eligible finite population influence estimates remain visible even
when their intervals include zero. Declared objective paths also stay visible,
with uncalibrated numerical effects left unknown. Zero and variable responses
remain explicitly marked. The default view hides only uncalibrated candidate
side-effect paths and their indirect statistics. Control edges are unaffected.
The service supplies default and with-uncertain catalogs separately. Both retain
raw risks for their groups for audit. Additional uncalibrated relations can add groups.

The main list has two columns: xApp combination and graph conflict risks.
Viewing a row only highlights its xApps and graph paths; it does not select a
policy action, winner or deployment. Conflict-bearing groups are retained.
Declared objectives, graph relations and nonempty intervals do not prove physical reachability
or simultaneous KPI attainment. Real runtime proposals and joint-network
validation belong to a later stage.
There is no combination-selection algorithm or KPI-based ranking in this version:
`selection_algorithm=null`, `ranking_performed=false`, and no default winner.

Current hot-started learned xApps use frozen mode, with learning and exploration
disabled. Their own models, guards, internal windows/pending state and delivered
observations determine their proposals. SMO does not edit that policy or replace
its action. Runtime proposal and actual execution are distinct: the existing
execution path can apply the user-requested uniform-random direct-conflict
arbitration. None of that execution is invoked by this graph review.

The legacy `smo.intent_match`, `smo.conflict_catalog` and
`smo/web/legacy_action_review.js` remain only for auditing historical action-space
experiments. The browser does not load the legacy action catalog. Node-only
exports preserve old experiment tests. The legacy catalog generator targets
`legacy_action_review.js`, so it cannot repopulate the active UI with action
choices.

Read-only planning APIs:

- `GET /api/graph`: source graph and telemetry, with redacted language status.
- `GET /api/intent/status`: redacted model configuration.
- `POST /api/intent`: new text, optional slot and `active_intent_ids`.
- `POST /api/intent/review`: saved IDs and observation slot, without an LLM call.
- `POST /api/intent/session-key`: session-only provider key/model configuration.
- `POST /api/intent/clear-key`: clear that credential.

Only saved server records can supply old anchors; clients cannot replace them
with submitted numeric ranges. The service binds loopback and rejects foreign
origins. Models, enable states, control values and retained simulation data are
unchanged by review. Keys stay in service memory or its environment and are
excluded from graph/HTML/JSON exports and the intent ledger.

Each graph **xApp block** displays frozen-model shadow decisions retrieved via
`POST /api/xapps/preview` (snapshot slot only). `--live-xapps` records them
automatically; baseline runs can opt in with `--xapp-decision-preview`.
Every 100 ms, the RIC uses its ordinary public E2 projection and executed-control
ownership receipt. Each frozen xApp runs on an independent deep copy before
arbitration. Active instances keep their copied lifecycle/pending/RNG state; OFF
instances preview their first activation, explicitly labelled `cold_activation`.
Logs, training and model saves are disabled on these disposable copies. No
selection, policy-store update, arbitration or executor call is made by preview.
The graph loads the selected snapshot's preview automatically. Its toolbar's
**更新拟动作** button explicitly refreshes it. Each xApp block shows the control,
exact scope, delivered current value and proposed value, retaining its ON/OFF
badge. Identical parameter/current/proposed triples share a scope list; differing
values are never averaged together. Block heights and graph ports follow the
contents so proposals do not overlap or disappear. Detailed original values and
cell/UE membership remain in SVG titles. There is no separate preview panel.
Identical proposals, no new action and missing/invalid inputs are distinct states.
Reading a replay retrieves the captured shadow inference at exactly that slot;
it does not infer decisions from reduced KPI graph values. Older runs without
`xapp_decision_previews.jsonl` report unavailability rather than fake actions.
Preview results do not filter, rank, select or numerically validate the graph's
candidate combinations. They are retained separately from executed receipts.

Run in the existing cached environment:

```bash
uv run --offline --with numpy python -m smo --view population \
  --replay /path/to/retained/run --influences /path/to/influences.json \
  --intent-store /path/to/frozen-intents.json --port 53505
```

The general UI recording must include each relative target's measurement.
For example, BS power needs enabled AP services and recorded energy counters;
the retained native AP-enabled replay supplies them. A missing current value
cannot be fabricated. The six editable preset sentences pass through the same
LLM boundary; selecting a preset alone does not analyze, enable or deploy an
xApp. The offline HTML is a standalone graph replay; natural-language analysis
requires the local HTTP service.

Primary validation covers graph-only combinations, service non-use of legacy
action-space functions, interval/ledger semantics, origin boundaries, language
scope definitions and browser rendering. Analytic fixtures are never presented
as recorded network performance or provider responses.


Intent achievement comparison uses a separate immutable before-snapshot archive
(`frozen-intents.comparisons.json`, scoped to its intent-ledger filename). The UI displays
only the KPIs explicitly requested by the active saved intents. Targets remain
frozen; neither new observations nor a new comparison baseline rewrite them.
`POST /api/intent/comparison/start` accepts saved intent IDs, a recorded slot,
and `observe_slots` (default **500 slots**, 250 ms in this network).
`POST /api/intent/comparison/review` accepts its saved comparison ID and slot.
Neither endpoint loads a learner, selects apps or deploys controls.

Comparison v2 derives **two exact, equal-length windows** from the existing
per-slot scalar trace, independent of the UI replay sampling stride. For a
baseline at B, the before interval is `(B−W, B]`. If A is the first genuine,
known xApp `applied` slot after B, the after interval is `(A−1, A+W−1]`.
It includes exactly W executed slots, including A. ON badges and shadow
proposals are insufficient. There is no extra fixed settling period; the
initial transition is included. Missing history or a reset in any member's
counter makes the affected metric unavailable, never a point-value fallback.
The original point snapshot and frozen intent ranges remain intact for audit.

Mean UE/BS power is interval energy divided by elapsed seconds. Goodput uses
interval delivered bytes; URLLC failures use interval deadline-failure plus
disjoint overflow events. Packet delay is completion-weighted within the
interval. Expected MEC demand reports its time mean and full-interval extrema;
limits are checked against the extrema so an average cannot hide a transient
configuration violation.

MEC latency uses routed task birth/terminal events, excluding pre-window
births. Both arms admit births during `[start, end − max(task deadline))`,
with the maturity guard entirely inside W slots. For this network, W=500
slots leaves a 150 ms birth exposure followed by the existing 100 ms task
observation allowance. Completed, failed and unresolved mature task counts
are displayed. No mature completions, missing event logs, or unresolved
mature tasks yield unavailable latency. Event totals are audited against the
independent cumulative task counters. Future terminal events cannot leak
into an earlier review. This completed-task latency is not a delivery-success
SLA; its sample composition and sample counts remain visible.

For target interval I=[L,U] and measured value y, absolute shortfall is
`max(0, L−y, y−U)` (omit an unbounded side). For a full-interval configuration
constraint, use its minimum/maximum for the respective sides. Normalize by
the absolute before-window reading. If that is zero/unavailable, use a
nonzero requested boundary, then a known measurement resolution; otherwise
normalization is unavailable. The scale remains fixed between arms. Strict
endpoint equality can have zero numeric distance while still failing its
open-bound requirement; the boolean result is authoritative.

Every requested KPI must pass. Report the worst normalized shortfall across
all requested KPIs and its before-to-after decrease in **percentage points**.
There is no weighted compensation between KPIs. Both feasible outcomes have
zero shortfall; overshooting a feasible range earns no extra reward. Missing
measurements prevent an overall quality claim. No non-requested KPI enters
the score. Each saved intent also shows its own conjunctive result.

A 500-slot result is a short-term screening result, not independent trials,
a long-term SLA estimate, or causal proof of xApp benefit. Sessions persist
across UI refresh and reject a changed live run identity. Live mode tails the
existing plain/gzip traces incrementally without duplicating simulation
recording. No simulation branches, xApp selection, ranking or deployment are
started by these endpoints. `oran.intent-comparison.v2` exports window bounds,
measurement provenance, scale basis and gaps alongside raw readings.

The graph, inspector and combination table now expand to their full contents.
There are no nested vertical/horizontal graph or table scroll areas. The SVG
fits the available width, and the complete combination list follows the graph;
only the document uses normal page navigation as its content grows.

Exhaustive empirical evaluation is now the primary workflow. After graph
backtracing and structural conflict detection, **评估全部 N 个组合** freezes
all candidates in the selected solid/dashed catalog, all active intent ranges,
the snapshot identity, trained models, worker source archive hash, seed, and W-slot budget. The backend
recomputes the entire catalog; the batch-start API rejects a group ID, group
subset, actions, parameter values, model paths and commands. Every catalog
member runs, even when it conflicts or an earlier member already satisfies the
intent. Catalog order is enumeration order, not an optimization ranking.

Configure an isolated AGX worker with `--deployment-config`. The existing
`scripts/orin/prepare-smo-worker.py` copies source/models to an owned worker
folder and reuses the installed virtual environment. It stores no password.
The current target is AGX at 100.72.45.61, using a dedicated memory-backed
`/dev/shm/oran_smo_batch_eval_20261119/source` copy. An authenticated SSH
ControlMaster is required. No global packages or settings are changed.

Each combination starts a **fresh independent run**, using the same native
six-UE fast scenario, seed 20261118 and frozen release 20261117. All xApps are
OFF during baseline collection. The selected member set then follows normal
hot activation, RIC arbitration and delayed actuation. No shadow action is
deployed and no learned action is manually adjusted. For W=500, the baseline
ends at slot 520 and its scored interval is slots 21–520. Selection is admitted
after that boundary; the post window starts at its first actual application A
and ends at A+499. A bounded two-controller-period action-start allowance is
included in the total run plan, not added as a settling period in the score.
The remote source marker must match the batch's frozen source hash before every launch. Each arm reuses only propagation caches, never the preceding arm's network,
learner, lease or control state. Direct conflicts retain uniform-random
mitigation. Frozen intent limits are never rebased to a new run's baseline.

`POST /api/deployment/batch/start` accepts saved intent IDs, snapshot slot,
`observe_slots` and `show_dashed`. `pause` and `resume` accept only the batch ID.
The queue runs one combination per configured AGX concurrently. Successes, execution failures,
measurement unavailability and unmet KPI targets remain distinct. A failed
execution does not end the remaining queue and never yields fabricated zero
KPIs. A checkpoint is committed on job admission and every completion. Pausing
stops the owned current run, retains partial data and leaves pending members
for resume. Completed members are never rerun on resume; retrying a failed or
cancelled member creates a new attempt while retaining previous recordings.
A lost SSH link blocks further starts until the active run is reconciled.
Service restart recovers the same frozen catalog and current child job.

The combination table shows **every member**, its previously detected conflict
risks, execution status, requested KPI before/after values, raw/percentage
changes, samples and conjunctive target attainment. No non-requested KPI is
scored. Baseline differences across arms are flagged, not filtered out.
The batch does not quantify conflict severity, assign conflict weights, rank
combinations, select a winner or stop at the first successful combination.
Existing interval-quality fields are retained in raw comparison JSON for
compatibility; they are not used by the batch scheduler or as conflict scores.

`GET /api/deployment/batch/results?batch_id=...&format=json|csv` exports all
members, including pending/failed rows with missing measurements left blank.
JSON retains intent ranges, catalog hash, original conflict records, seed and
all attempt IDs. Each child keeps its trained-model checkpoints/hashes, raw
telemetry, selection/actuation receipts, exact-window comparison and standalone
replay HTML. Raw archives are copied to the UI computer before marking a
member complete. The current local output is beneath
`Desktop/ORAN/SMO/all-combination-evaluation-20261119/jobs/`.

The graph remains the reference source while the batch runs; the final member
is **not automatically selected or opened**. Clicking **查看实测回放** explicitly
loads that completed member's real graph/telemetry and comparison. Merely
highlighting a graph combination never launches or chooses it. The legacy
single-run backend remains available for isolated diagnostics, but the main
UI submits only whole-catalog batches.


Dual-AGX execution uses an operator-controlled pool configuration with global
`local_output`, `seed`, `label`, and a `workers` list. Each worker has a unique
`id`, `host`, `socket`, `project_root`, `python`, and `label`. Only one unresolved
job can occupy a host. Idle workers take the next pending catalog member,
letting the faster device finish more members without changing catalog order,
intent constraints, source/model hashes, or simulation seeds. Existing single
worker configurations remain supported.

On pool migration/restart, every in-flight child is reattached to its original
host. Completed member results and IDs remain unchanged; only pending members
are assigned to newly available workers. A disconnected worker's unresolved
job is retained without duplication while healthy hosts continue the queue.
Pause sends scoped stop markers to all running children and retains partial
recordings. Resume/retry preserves successful members and previous attempts.
The UI shows each host, combination, stage and slot progress; CSV/JSON include
worker provenance. Neither dispatch nor speed measurements rank xApp effects.

The current pool is AGX2 (100.72.45.61) plus AGX1 (100.116.105.126). Both use the
**identical frozen archive** used by the already running batch; no source or
model changes are applied to either simulation copy. Separate virtual
environments are reused (Torch 2.12 on AGX2, 2.9.1 on AGX1; Sionna 2.2.0 on
both). A same-combination bridge matched baseline/post UE power, throughput,
URLLC failures and expected MEC load; the AP power difference was numerical
roundoff. Runtime environments and per-run baseline flags remain recorded.
This bridge checks consistency for one case, not statistical equivalence of
all possible scenarios. Current 200-slot windows remain unchanged; with the
100ms task maturity guard they cannot estimate mature MEC task latency, which
remains explicitly unavailable rather than zero.

Unified UI entry and user-owned memo persistence

The interactive system has one canonical entry: `http://127.0.0.1:53505/`.
Opening the legacy 53504 UI automatically routes to 53505 before issuing
client API requests. The legacy language broker's process/key are retained;
it is not a separate user-facing workflow. Intent parsing, frozen ranges,
knowledge graph, all-candidate enumeration, dual-AGX dispatch, progress,
per-combination measurements and result viewing use the same main page and
same-origin API. No topology-pruning policy has been enabled.

Intent satisfaction quantification uses only the requested KPI intervals and
actual equal-window deployment-before/after measurements. Every comparison row
now exposes a `quantification` object, also included in batch JSON and CSV.
Pure increase/decrease requests report signed favorable change relative to the
actual deployment baseline, without inventing a finite completion percentage.
Numeric targets report progress from that baseline to the violated boundary;
progress may exceed 100%, but actual membership in the entire frozen interval
still determines feasibility. A two-sided overshoot can therefore remain unmet.
An already-feasible baseline reports maintenance and interval margin. Zero
baselines retain absolute change without a fabricated relative percentage;
strict endpoints and missing samples keep their existing feasibility semantics.
Metrics constrained by window extrema use those extrema for target progress and
interval margins. No averaging across objectives, ranking or statistical
significance claim is introduced. Archived comparisons are enriched on read;
their original measurements, intent anchors and simulation records are retained.

Preset text, freeform tags and deleted preset IDs now use an atomic server
store (`--preset-store`, currently `Desktop/ORAN/SMO/intent-presets.json`).
The browser cache is imported only when the server store is empty; a fresh or
empty browser does not overwrite existing tags. Subsequent writes require the
current revision; stale tabs receive a conflict while their editor text remains
intact. Prior revisions are retained in `intent-presets.history.jsonl`.
Browser copies remain a cache/offline fallback, not the sole durable record.
Exports embed the saved profile for offline display. Tags remain arbitrary
user memos and are never inserted into the LLM request, parsed KPI bounds,
conflict classification, or deployment decisions.

The original eight user-tagged presets and the deletion of `lc02` were recovered
from the 53505 browser profile. They had appeared missing because the 53504
origin has a separate browser store and an older API context. Original text,
punctuation, tags and deletion choices were retained rather than reset to
built-in defaults. Existing frozen intents and the completed 56-member batch
are unchanged by profile recovery.
