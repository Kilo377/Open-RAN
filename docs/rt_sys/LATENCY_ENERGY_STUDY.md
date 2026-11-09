# X12 latency control and X01 energy: small factorial pilot

The native network still has six moving UEs, three Munich cells, 10 MHz and
depth-2 ray tracing. X01 is the accepted frozen UE-energy RF controller. X12
learns a separate small random forest online from actually executed scheduler
settings and delayed task feedback. The refinement has four candidates: native
round-robin with target weight 1 (abstention), or `latency_aware` with target-UE
scheduler weights 0.5, 1 or 2. It does not learn from predictions used
as labels or alter the traffic generator to manufacture a favorable outcome.
The current primary latency objective is explicitly **offloaded AR-task
end-to-end delay**, aligned with the offloaded-task delay in paper Eq. 14.
All-task delay, success and delivered traffic remain visible safeguards.
The receipt identifies this objective as `offloaded_ar_task_e2e_v2`; it is a
declared change of measured population, not a lower improvement threshold.

The two declared RCP sets are different:

| Controller | Writable network attributes |
|---|---|
| X01 energy | `ue_drx_configs`, `ue_offload_fraction` |
| X12 latency | `scheduler_mode`, `ue_scheduler_weights` |

The physical coupling is through radio service availability, shared scheduling,
task-result delivery and UE energy. Distinct writable fields establish a
candidate indirect interaction, not proof of an actual harmful conflict.
Offloading may reduce both UE energy and task delay, so synergy is possible.

## First pilot and reason for refinement

The original frozen source `f90467754f8988e73240e87d24c3d4f94a652055502ad4416a7dd0788e21f6ee`
trained on seed 66 and evaluated seed 55. Its three actions all forced
`latency_aware` and scaled the cell-wide `latency_bias`. Although actual online
training, execution and frozen-model checks passed, X12 did not improve pooled
task delay: OFF 15.781 ms versus X12 16.906 ms. UE4 improved, while UE0 worsened
from 16.906 to 20.281 ms. The old driver flagged a 2.796% energy cost when adding
X12 to X01, but this did not establish conflict between two individually
effective controllers. Energy interaction D was slightly beneficial and delay
D was only +0.1875 ms. The original outputs and receipt remain immutable.

The refinement addresses the action mapping rather than changing the network:
target-UE scheduling weights replace the cell-wide priority multiplier, and
native round-robin/weight1 is available when changing scheduler family is not
useful. A valid native abstention is reported as such; it is not called an
effective latency optimization. Individual efficacy is now a prerequisite for
claiming a successful indirect-conflict case.

The second pilot, source
`8e2903f7bb5f54672a9a5843bb6d14afe4d7b0f759f34ed7e19e76d737cbb369`,
used the new per-UE controls, 4 s of online training on seed 88 and 26 measured
rows. On validation seed 55, X01 saved 12.60% target energy, but X12's pooled
all-task mean was 16.0469 ms versus OFF 15.7813 ms, a 1.68% deterioration.
UE0 worsened while UE4 improved. Its receipt correctly says
`controllers_not_individually_effective` and skips seed 77. A separate fixed
weight-2 diagnostic improved pooled all-task mean by approximately 4.55%,
below the original 5% requirement; this was a fixed policy, not an RF result.
These failed overall-delay pilots and their receipts remain unchanged.

Inspection showed that local tasks often contribute a fixed 25 ms component
that a radio scheduler cannot control, mixed with a small and variable set of
offloaded outcomes in each training cohort. The next iteration measures the
actual offloaded route directly, retains the same 5% improvement requirement,
and protects all-task outcomes. Historical traces lacking route counters are
rejected by this evaluator rather than retrospectively declared successful.
The learner uses measured pending-offloaded-task state and measured routed
outcomes; its feature schema rejects earlier artifacts.

The first AR-scope run, source
`b6a87c4745196b79896663a54f0dd2e8c03adfac6cdebc4f41351bdee431bef1`,
collected 22 real route labels on seed 99. Validation seed 55 improved pooled
offloaded E2E from 10.25 to 9.85 ms (3.90%), with both target UEs improving and
all global guards passing. This still missed the unchanged 5% criterion;
the receipt remains `controllers_not_individually_effective`, and seed 77
was not run. BOTH showed no operational conflict in that run.

## Run budget and interface

```sh
python experiments/run_latency_energy_study.py \
  --output outputs/latency_energy_pilot_YYYYMMDD \
  --energy-model outputs/models/ue_energy_rf/model.json
```

The default online X12 training duration is 4 simulated seconds at seed 99,
with eight exploration windows (two repeats of four candidates).
It uses the original demand and no X01 controller. The trained X12 model is
then frozen for four matched 1 s runs at seed 55: OFF, X01, X12 and BOTH.
The default is 8 simulated seconds across five runs. The initial experiment
has one validation seed. Seed 55 informed the refinement after the failed
first pilot, so it is explicitly validation, not an untouched test seed.
`--eval-seeds 55 77` requests a second complete factorial at a predeclared
independent seed, in reversed execution order, only if the first validation
passes execution, disjoint-RCP and individual-efficacy checks. On failure the
driver records `needs_refinement` and skips additional seeds. X01 may already
have been evaluated on one of these network seeds in another study; that prior
exposure must also be disclosed rather than claiming wholly unseen conditions.

`--train-seconds`, `--eval-seconds`, `--train-seed`, `--window-ms` (currently
100), `--explore-windows` and `--latency-model` expose the cost and artifact
choices. `--train-only` collects online training without evaluating. Use
`--skip-training --latency-model outputs/.../model.json` to evaluate a trained
X12 artifact without fitting it. All study/model paths stay under the project
`outputs/` directory, and each study output directory must be fresh.

For one final bounded regularization attempt, training-only leave-one-out CV
on those 22 measured rows compares `max_features` in {1, 0.7, 0.5, 0.33}.
The selected 0.5 lowers the reported CV MAE by 5.71% versus 1. Validation
outcomes are not CV samples or hyperparameter-selection scores; seed 55 did
motivate further work and remains refinement validation. The objective,
action set, network demand, 5% efficacy target and all guards remain unchanged.
X01 retains its original fitting default of 1.

Resume a separate copy of the seed-99 X12 artifact, preserving the original:

```sh
mkdir -p outputs/models/latency_rf_resume
cp outputs/native_factorial_ar/latency_rf_model.json outputs/models/latency_rf_resume/model.json
python experiments/run_latency_energy_study.py \
  --output outputs/latency_energy_resume_YYYYMMDD \
  --latency-model outputs/models/latency_rf_resume/model.json \
  --resume-parent-model outputs/native_factorial_ar/latency_rf_model.json \
  --training-cv-report outputs/latency_rf_training_cv.json \
  --resume-online --train-seconds 1.2 --train-seed 101 --eval-seeds 55 77
```

The CV JSON must already exist at the supplied path. The driver records its
path/hash before training and verifies that it stays unchanged. The copied
model must exist, be trained and initially match the separately supplied
immutable parent. `--resume-online` is mutually exclusive with
`--skip-training`. A trained online model begins with exploitation, collects
new real delayed cohorts and refits from its retained measured history.
The driver requires more rows, more trained samples and more fits before any
frozen evaluation; new rows must match new `measured_sample` event counts,
and the original feature/label prefix must be unchanged. The receipt records
the before/after artifact stamps and growth proof under `training_lineage`.
The original parent artifact is checked after every case.

V1 trained artifacts record their actual historical `max_features=1` by
inference; V2 explicitly stores `fit_parameters`. Loading a V1 forest under
the new X12 configuration does not relabel it as a 0.5 fit. Only a successful
fit after new measured data may produce that V2 claim, and the driver checks
actual fitted metadata against the online configuration. Frozen runs must
preserve the actual fitted metadata, sample/fit counts and artifact bytes.
This adds only 1.2 simulated seconds of new training physics before the same
four-arm validation. Seed 77 runs only after validation passes. If this
bounded attempt misses the criterion, stop and report it honestly rather
than continuing hyperparameter search against validation outcomes.

An optional `--traffic-load` multiplier is explicit, defaults to 1, and applies
to every training and evaluation arm. The driver never silently increases
load, narrows bandwidth, moves UEs or changes deadlines to obtain a conflict.
Stress experiments must be labelled separately from the native-demand pilot.

## Measurements and evidence

Every case exports and audits its usual HTML. `acceptance.json` is atomically
updated after each phase and retains run commands, source fingerprints,
simulation/wall times, artifact hashes and all four-arm comparisons. Geometry
is reused within a seed to reduce cost; these sequential wall times are not a
controlled measure of inference overhead. Source/configuration must remain
unchanged throughout one study.

All four arms must have identical radio/movement/seed manifests and original
business configurations. Task cohorts are accumulated from the full recorded
trace, with identical intervals and **all-task exogenous generated counts**
across the four arms. Offloaded generated/completed counts are deliberately
not required to match: X01 controls routing, so those conditional populations
may differ. Their counts are reported for every target UE and case.
Only creation cohorts starting at or after 100 ms and observed through their
whole deadline horizon enter task scoring. Failed and overdue pending tasks
stay in the denominator. Completed-task mean delay is reported separately;
it does not turn dropped or pending tasks into successful completions. Energy,
packet goodput and packet-deadline counters cover the entire run, including
startup, and are not silently trimmed to improve results.

Route aggregation requires actual `offloaded_generated`,
`offloaded_completed`, `offloaded_on_time_completed`, `offloaded_failed`,
`offloaded_pending` and `offloaded_completion_latency_ms_sum` counters in each
scored cohort. Route conservation and subset counts are checked. No route is
inferred by multiplying total tasks by a requested offload fraction. The
offloaded mean is weighted by actual completed routed tasks; success rates
retain failed and pending routed tasks in their generated denominator.

The study reports UE0/UE4 individually and pooled, all-six-UE energy/goodput,
and non-target UE1/UE2/UE6/UE8 changes. A pooled mean is weighted by completed
task counts. Non-target harm is visible even if the target controllers meet
their own objectives; low-latency UE2/UE6 protection must still be inspected.

Declared RCPs are read from the actual active xApp descriptors. BOTH must have
an empty RCP intersection. X01 must show RF-selected actions, while X12 must
show new RF prediction batches beyond the cumulative count stored in its
artifact (including genuine RF decisions to abstain). Each must also show
executed-policy confirmation events, ordinary successful actuation receipts,
matching runtime effects and no recorded rejection. Frozen sample counts,
trained-sample counts and fit counts must match the artifact; artifact files
must remain byte-for-byte unchanged. Otherwise the result is labelled
`insufficient_control_evidence`, not used to claim a verified interaction.

For X12, a receipt must contain the requested weight map for current target
UEs, and the produced runtime map must match it with values in {0.5,1,2} and
mode in {`round_robin`,`latency_aware`}. Both target UEs must be covered across
receipts. Native round-robin/weight1 can pass execution checks, but an entire
run of such abstentions is labelled `native_abstention_only` and cannot satisfy
the separate individual-efficacy criterion.

OFF and X01 retain native round-robin scheduling. X12 and BOTH can retain it
or switch target-containing cells to `latency_aware` after normal delayed control.
When a switch occurs, a benefit versus OFF includes the scheduler-family transition.
It does not isolate the necessity of RF versus a fixed `latency_aware` policy.
A fixed-weight ablation can test that question without calling a fixed policy
an RF controller.

## Operational conflict classification

Before calling either a successful conflict case or compatible coexistence
between effective controllers, standalone X01 must reduce pooled UE0/UE4 energy
by at least 5%, and standalone X12 must reduce their completed **offloaded
AR-task** pooled mean E2E delay by at least 5%, both versus OFF. Each also needs
no more than five-point loss of target **all-task** on-time rate, no target
all-task mean-delay increase beyond `max(1 ms, 5%)`, at least 95% of OFF
delivered bytes for every UE, and no more than five-point non-target packet
deadline-rate increase. Mean-delay safeguards are checked separately for UE0
and UE4; the route objective cannot hide degradation of local or overall tasks.
Pooled efficacy does not guarantee every target UE's mean delay improves;
the per-UE results remain visible. A controller that merely selects native
abstention throughout is not counted as an effective latency controller.
If either standalone criterion fails, the classification is
`controllers_not_individually_effective`, even if a raw cross-cost flag exists.

For each seed the driver reports the factorial interaction

`D = BOTH - X01 - X12 + OFF`.

All-task and offloaded-task mean-delay D are both retained, with their scopes
labelled explicitly. Positive energy/delay D and negative all-task on-time-rate
D are adverse directions.
They describe nonadditivity; a positive D alone does not establish a conflict
when BOTH is still better than either isolated controller.

The predeclared operational comparisons are:

- Energy intent: BOTH raises pooled UE0/UE4 energy by more than 2% versus X01.
- Latency intent: for either target UE, BOTH raises completed **offloaded
  AR-task** mean E2E delay versus X12 by more than `max(1 ms, 5%)`.
- Separate all-task safeguards: BOTH raises either target's all-task mean
  delay versus X12 by more than `max(1 ms, 5%)`, or lowers its all-task on-time
  rate by more than five percentage points.

The BOTH/X12 routed comparison is conditional on tasks actually offloaded in
each arm, not a matched individual-task treatment effect. X01 may change the
membership and number of routed tasks; all generated demand remains matched
and all-task safeguards prevent quietly ignoring the rest. Offloaded counts,
completion rates and on-time rates are reported alongside the primary mean.

With valid executed-control evidence, disjoint RCPs and both standalone
efficacy criteria satisfied, a threshold crossing is
labelled `indirect_operational_conflict_observed`. Without such harm the label
is `no_detected_conflict`, including compatible or beneficial coexistence.
These thresholds describe an engineering tradeoff in this paired run; they
are not confidence intervals, statistical significance, standard-mandated
SLAs or proof that no alternative assignment could satisfy both objectives.
The energy >2% flag is retained as `operational_tradeoff_flag` even for an
ineffective controller, and is separate from observed task deadline
non-success counts. It is not relabelled a hard energy KPI violation without
an independently declared energy contract. Factorial D is always reported,
but nonadditivity alone is not used as proof of conflict.

The same distinction applies to non-target diagnostics: a greater than 5%
goodput loss, five-point packet deadline-rate increase, or mean packet-delay
increase beyond `max(1 ms, 5%)` is highlighted for review. Those diagnostics do
not silently modify the definitions of the two target xApp intents.

## Scope

The RF is genuinely fit from a small online dataset, then deployed frozen.
The first few seconds do not establish minute-long mobility, prolonged outage,
MEC congestion or broad traffic generalization. UE power parameters remain
uncalibrated scenario assumptions; uploads use the shared logical UL model
and returned task data uses the existing NR/SYS DL/HARQ path. A no-conflict
result is a valid result of the current native network, not a failed test to
be repaired by artificially forcing a conflict.

## Final bounded result (2026-11-04)

The final frozen candidate used source
`f317fd2b10ae5546fd9654e87debe7067e3542766de30e7a03505961d61fb28e`.
Its original 22 measured rows were retained, six new online rows were collected
on network seed 101, and fits increased from 11 to 15. The resulting 28-row RF
uses actual `max_features=0.5`; the model SHA-256 is
`507fab1eaf3e5b93150938f751cff005940b4a3a2f4c27692acc4c9bcbbfe1f9`.
The training-only CV improvement is not a network-performance improvement.

The matched 1 s validation on seed 55 gave:

| Arm | UE0+UE4 energy, J (whole run) | Offloaded-task E2E, ms (mature cohorts) | Routed tasks scored |
|---|---:|---:|---:|
| OFF | 1.888050 | 10.250000 | 20 |
| X01 | 1.650145 | 10.390625 | 32 |
| X12 | 1.883720 | 9.850000 | 20 |
| BOTH | 1.659565 | 9.828125 | 32 |

X12 improves latency by **3.90%, below the unchanged 5% target**. Its per-UE
delays improve from 10.6111 to 10.2222 ms (UE0) and from 9.9545 to 9.5455 ms
(UE4). All-task delay/success, per-UE delivery and non-target guards pass.
Actual RF inference, executed scheduler settings and frozen artifact checks
pass, but these do not replace the failed efficacy criterion. Seed 77 was
skipped; no independent confirmation or paper-level gain is claimed. The
bounded tuning attempt is finished.

The candidate is available at `outputs/models/latency_scheduler_rf/model.json`,
with `deployment.json` explicitly setting `status=experimental` and
`accepted=false`. Native xApp defaults stay OFF. The combined launcher is
`scripts/orin/run-x01-x12-rf.sh`; it explicitly enables the experimental pair.
Compact KPI/RCP descriptors are `ric/near_rt/pools/chongyu/x01.json` and
`x12.json`.

The same-task diagnostic fixes membership before outcomes and validates the
simulator's actual route counters and policy receipts. All 20 common offloaded
task terminal outcomes are exactly identical in X12 and BOTH, with 9.85 ms
pooled delay and 100% on-time completion in both. BOTH's additional 12 routed
tasks explain its different conditional mean. Target energy increases 0.57%
over X01 for the whole run, or 0.97% over the common steady interval [0.1,1] s
under identical executed DRX/offload policies. Neither crosses the declared
operational threshold. There is **no convincing indirect-conflict case in
this native-load pilot**, and X12 also lacks standalone efficacy acceptance.

Separately, the original accepted X01 model was frozen for a 10 s AGX1
OFF/ON comparison using source `d3540db48286097294b33709e50d92158e52b541c029676ebaa6b36fd4dda453`.
Target UE0+UE4 energy decreases 16.261595 to 14.027480 J (13.74%); all-six-UE
energy decreases 51.786225 to 49.559315 J (4.30%). Network goodput is 26.3335
versus 26.3541 Mbit/s. Both target UEs pass the relative task/delivery guards.
UE0 nevertheless suffers a native late-run no-path outage: its mature-cohort
on-time rate is 137/196 (69.90%) OFF versus 135/196 (68.88%) ON. This is not
an absolute SLA success. UE4 is 196/196 on time in both. Main simulation wall
times are 983.88 s OFF and 3102.84 s ON; these are not controlled RF-overhead
measurements.

The final focused suite passed 215 tests and four subtests on AGX2, including
actual scikit-learn fitting and artifact compatibility. The earlier actual-NR
integration checks also passed. All exported HTML integrity/self-contained
audits passed; no new browser visual inspection was performed. Failed pilots,
parent artifacts, CV data, raw traces and source archives remain retained.

## Runtime bookkeeping note

A read-only audit during the long X01 test found growing E2 lifecycle work.
Held target states receive fresh request IDs every slot; status updates scan
request history, and per-slot metrics recount its statuses. This creates
quadratic bookkeeping as the run grows. It is a plausible source of increased
wall time with active xApps, separate from RF fitting; no controlled profiler
attribution or speedup has been established. The frozen studies retain the
original path. A future change should preserve request IDs, actuation/RNG
semantics and full audit rows, using indexed status lookup and exact incremental
counters, then verify per-slot equivalence on a fixed-seed 400-slot run.


## MEC stage diagnostics

Runs with `--mec-stage-metrics` record additive task-stage residence times for upload, backhaul to
edge, edge compute, return backhaul, downlink result delivery, and local
compute. Actual UL/CPU service is work divided by the configured service rate;
remaining residence is waiting or slot quantization. Downlink residence is
observed as a whole and is not labelled a measured PHY service time.

Terminal events contain `task_stage_timing`. Public per-UE service reports
contain `ue_task_stage_metrics`, keeping completed, failed and pending
populations separate. The automatic HTML shows completed-offloaded stage means,
current backlog by stage, and cumulative terminal failures by stage. These
statistics do not enter xApp features or labels.
Timing diagnostics are opt-in; the general fast-model default does not perform
the additional per-task aggregation or add diagnostic fields to its reports.

The recorded deadline policy remains keep-until-completion: an overdue task
continues consuming resources, and the existing per-UE admission cap applies
before route selection. No application cancellation or packet/HARQ discard is
implied. Cancelling an already-reserved result requires a separate coordinated
PacketLedger/HARQ lifecycle change.

For a diagnostic-only fixed action, use `--latency-xapp-mode fixed` and
`--latency-xapp-fixed-action 0|1|2|3`: native RR, or latency-aware target
weights 0.5, 1 and 2. Fixed mode uses normal observations, ownership and delayed
actuation, records measured outcomes, and never fits or predicts.
`experiments/run_mec_stage_ablation.py` compares OFF, the three fixed settings
and the frozen RF using matched physical inputs and arrival cohorts. Its short
results diagnose the writable control surface; they are not a new training or
long-horizon acceptance claim.
