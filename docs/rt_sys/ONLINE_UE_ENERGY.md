# Online RF UE-energy pilot

The first Chongyu energy xApp controls UE0 and UE4 in the existing six-UE
Munich network. The radio, traffic, UE positions, RT depth and fast model stay
fixed. UE2 and UE6 retain their low-latency DRX protection. This is an initial
system-model experiment, not a reproduced trained model from the paper.

## Learning loop

The xApp reads delivered KPI reports, chooses one bounded DRX/offload candidate,
and sends an ordinary scoped control action. Only the chosen action's actual
subsequent outcome becomes a training row. Predictions of unexecuted candidates
are never treated as labels. Delayed telemetry must show the requested policy
was executed before crediting its outcome.

Random forests do not use `partial_fit` here. Online means that new labels are
collected during the simulation and the small forest is refitted in batches
as those observations arrive. The saved JSON forest contains the learned tree
structure and its feature schema. Frozen evaluation loads this artifact and
does not collect new training labels, fit, or modify the artifact.

Task feedback is grouped by creation-time cohorts, not by which action happens
to be running when an old task finishes. A cohort becomes eligible for scoring
only after its whole generation interval plus its maximum task deadline is
observed. Pending and failed tasks remain in the generated-task denominator;
they do not disappear from the success rate. Mean delay describes completed
tasks and is not substituted for the success-rate guard.

The existing traffic can keep DRX active for much of the time: its eMBB OFF gap
is 8 ms, equal to the original inactivity timer. Offload selection can therefore
be the main measured source of energy savings. DRX and offload actions and their
outcomes must be logged; savings must not automatically be attributed to DRX.

## Small-budget experiment

From the accepted AGX GPU environment:

```sh
python experiments/run_ue_energy_study.py \
  --output outputs/ue_energy_pilot_YYYYMMDD
```

The default budget is one 2.8 s online run with seed 11, followed by a matched
1 s native-baseline/frozen-policy pair at held-out seed 22. If this first pair
passes, a second 1 s pair runs at held-out seed 33, in reversed execution order.
There are at most 6.8 simulated seconds across five complete runs. Each run
exports and audits its ordinary self-contained HTML replay.

With `--fixed-reference`, a separate seed-22 run sets UE0/UE4 offload probability
to 1 at timeline slot 5 and retains the baseline DRX parameters. It executes
only after the first learned-policy pair passes. This reference is explicitly
labelled constant control, not RF; its outcome is saved under `fixed_reference`
and does not determine RF acceptance. With the default seeds it adds one
simulated second and reuses the already recorded seed-22 OFF run.

The predeclared acceptance contract is at least 5% aggregate modelled energy
saving on UE0/UE4, with no greater than five percentage points of on-time task
rate loss and no greater than 5% delivered-byte loss for either target UE.
OFF and frozen runs must have exactly matching deployment/movement/radio and
original business configurations. They use identical mature task-cohort
intervals and generated task counts. Task scoring excludes cohorts starting
before 100 ms, so the first control's startup transition is not attributed to
the deployed policy. Energy and goodput totals still include the entire run,
including startup; they do not silently discard its costs. The driver accumulates the bounded public
cohort snapshots from the recorded trace, so it does not silently score only
the last few cohorts retained in the final snapshot. Mixed-policy cohorts are
valid evaluation demand and remain in scoring; training must not use them as
if they were outcomes of one isolated action. The study also reports all-six-UE energy
and network goodput, so targeted savings are not presented as network-wide
savings. Returned MEC result bytes are included in delivered payload.

If the first pair fails, the driver stops with `status: needs_refinement` and
preserves the model, data and failed comparison. It does not loosen the
thresholds, drop unsuccessful tasks, silently change traffic or continue a long
search until a favorable seed appears. The next engineering step is to inspect
the online rows and action coverage, then explicitly run a small refinement
budget. A passing short pilot does not establish performance over a full minute
of mobility, prolonged outage, high edge load or unseen traffic patterns.

`acceptance.json` is atomically updated after every phase. It records commands'
outputs, source fingerprints, run times, HTML audits, per-UE guards and the
frozen artifact hash. Geometry is shared within a seed to save RT generation;
sequential case wall times are not a controlled xApp-inference-overhead test.
Use the xApp's own fit and prediction timers to measure algorithm cost.

Useful options:

```sh
# Collect a pilot without spending an evaluation budget yet.
python experiments/run_ue_energy_study.py \
  --output outputs/ue_energy_train_only --train-only

# Evaluate an existing artifact without modifying or retraining it.
python experiments/run_ue_energy_study.py \
  --output outputs/ue_energy_eval_existing --skip-training \
  --model outputs/ue_energy_train_only/model.json

# Resume real online sample collection and RF updates from an existing model.
python experiments/run_ue_energy_study.py \
  --output outputs/ue_energy_refinement --resume-online \
  --model outputs/ue_energy_train_only/model.json --train-seconds 1.5
```

`--train-seconds`, `--eval-seconds`, `--train-seed`, `--eval-seeds`,
`--window-ms` and `--explore-windows` make the experiment budget explicit.
`--resume-online` and `--skip-training` are mutually exclusive: the former may
update the existing model using newly executed outcomes; the latter never does.
The study directory must be fresh and stay under this project's `outputs/`.

## Interpretation

Energy is the configurable UE radio plus application-compute system model.
The watt values are uncalibrated scenario assumptions, not measured handset
battery consumption. MEC upload is a shared logical uplink service with RT
path availability; it is not an NR UL waveform/interference simulator. Results
return through the existing actual NR/SYS downlink queue and HARQ path. These
limits apply to both training and evaluation; the xApp does not remove them.

See [UE_SERVICES.md](UE_SERVICES.md) for the underlying state, energy and task
models. A modest online dataset can establish a useful deployed policy on this
preset; it is not evidence of broad generalization or of optimal control.

## Conservative deployment after the first pilot

The initial broad-DRX deployment saved energy on validation seed 22 but reduced
UE0 delivered bytes by 5.62%, exceeding the predeclared 5% budget. It is retained
as a failed pilot. Default exploitation/frozen operation therefore keeps the
original DRX profile and lets RF select among offload probabilities 0/0.5/1.
Balanced initial online exploration still measures all nine candidates;
`deploy_drx: true` is an explicit research opt-in requiring further validation.
A short resumed online phase gathers outcomes under the conservative policy.
Seed 22 has now informed refinement and is a validation seed; seed 33 remains
independent of those changes. Thresholds, traffic and radio are unchanged.
The fourth RF target uses the observed completed-task mean when available and
a 100 ms deadline penalty when no tasks complete; it is a training cost proxy,
not an invented latency observation. Reported actual task means remain
unavailable if no tasks completed, and the separate non-success-rate guard
retains all pending/failed tasks.

## Frozen deployment

After acceptance, copy the exact accepted JSON model to
`outputs/models/ue_energy_rf/model.json` in the independent project checkout.
The launcher `bash scripts/orin/run-ue-energy-rf.sh --duration-s 1 --output
outputs/energy_demo` uses the original six UEs, current fast model and frozen
X-01. It does not train or alter that artifact. Normal native launchers keep
xApps OFF. Models/logs stay under outputs; only online fitting needs
`requirements-ue-energy.txt`. Inference uses the checked JSON trees without
scikit-learn.

## Measured acceptance, 2026-11-04

AGX1 completed 2.8 s of initial online training (18 measured samples) and a
1.2 s resumed online phase (6 additional samples). The final forest contains
24 samples and seven batch fits. Training simulations took 506.04 s in total;
RF fitting itself took 0.2701 s. These timings exclude evaluation and setup.
The resumed phase changed the deployment action constraint, not the acceptance
thresholds. The failed pilot and its model remain available separately.

Each evaluation case simulated 1 s with the same six UEs and matched demand.
Energy and delivered-byte totals include startup. Task statistics use the
same eight mature 100 ms arrival cohorts per target UE (16 tasks per UE).

| Evaluation | UE0 + UE4 energy reduction | All-six-UE energy reduction | Network goodput, OFF → RF | Target on-time tasks, OFF → RF |
| --- | ---: | ---: | ---: | ---: |
| Seed 22, validation after refinement | 16.24% | 5.75% | 30.199 → 30.227 Mbps | 100% → 100% |
| Seed 33, independent confirmation | 15.57% | 5.46% | 30.200 → 30.227 Mbps | 100% → 100% |

Both target UEs passed their individual delivered-byte and task-rate guards.
Non-target UE delivered bytes were unchanged, with no queue overflow or packet
deadline misses in either pair. Mean completed-task latency on independent
seed 33 fell from 17.94 to 11.38 ms for UE0, and from 16.41 to 10.06 ms for UE4.
The small goodput increase includes additional MEC result bytes; it is not
evidence of a radio throughput improvement.

The frozen RF selected the native DRX profile with offload probability 1 for
both target UEs. A separate constant-offload-1 reference produced identical
seed-22 energy and KPI results. This validates a useful online learning and
control pipeline on this lightly loaded MEC preset; it does not establish
that RF outperforms a simple fixed policy. Longer mobility, outages and edge
congestion remain outside this short acceptance test. Energy numbers refer to
the modelled UE radio and CPU, with uncalibrated power assumptions.

The original regression suite passed 124 tests; the final conservative-policy
revision passed 52 related tests (overlapping coverage, not 176 distinct tests).
The deployment launcher also passed a dry run with frozen inference and the
correct six-UE configuration. See
[online_ue_energy_acceptance.json](online_ue_energy_acceptance.json) for hashes,
scope and compact results. Full commands, failed-pilot results, action records,
cohort audits and self-contained HTMLs remain in the study output directories.


## DRX control unlocked, 2026-11-17

The current launchers expose `--energy-xapp-deploy-drx` (default enabled) and
`--no-energy-xapp-deploy-drx` for the historical native-DRX ablation. The
controller's low-level constructor retains its conservative default for direct
legacy instantiations; finite and warm/live launchers pass the explicit setting.
The 221-row frozen RF checkpoint is unchanged. Release 20261117 records the
new action availability rather than claiming additional training. The historical
study drivers explicitly retain their original native-DRX protocol.

`experiments/run_drx_unlock_study.py` compares Base, frozen X-01 with native DRX,
and the same frozen X-01 with DRX control. Runs use matched seeds, traffic,
movement, radio, initial state and observation/control delays. Raw records,
model checkpoints and self-contained HTMLs are retained. The former five-percent
throughput guard is reported diagnostically; it does not stop this tradeoff
experiment or remove an unfavorable seed. The analysis checks executed DRX
profiles as well as selected actions and reports whole-network and target-group
energy separately, goodput, completed-packet mean/P95 latency, and mature MEC
cohort completion latency and on-time rate.

The graph includes all structured DRX profiles, and no longer marks X-01's
DRX interface locked. Joint-profile policy effects must not be substituted for
a scalar inactivity-timer derivative.
