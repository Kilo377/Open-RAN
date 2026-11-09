# Scheduler conflict pilot

This is a paper C-06-inspired M1 direct-conflict case. Its UE6 packet objective
is an explicit adaptation of X-12's AR task-latency objective, not an exact
reproduction; the legacy MEC/AR `latency_scheduler_rf` remains available.

This case pairs the existing frozen X08 throughput MLP with a separate packet-outcome
RF, `latency_urllc_rf`, targeting UE6's original 10 ms deadline. It preserves the
six moving UEs, three cells, PHY, channel and explicit 2× traffic profile. The
X08 policy is `throughput`: QoS impacts are measured, not constrained away.
The existing X08 model is immutable; this case makes no ML-versus-fixed claim.

Run the bounded isolated training phase:

```bash
python3 -m experiments.run_scheduler_conflict_study \
  --phase train --output outputs/scheduler_conflict_train_20261105 \
  --throughput-model outputs/x08_load2x_recovery_20261105/throughput_mlp_model.json \
  --train-seconds 6 --max-train-seconds 10 --train-seed 179
```

The one optional extension uses only the remaining predeclared training budget
and is permitted solely for insufficient real fitted action coverage. The RF
needs 24 actual windows, six per action, with a maximum of 48 retained rows.
All bad and zero-completion windows remain observations. A missing completed
mean is `null`, not zero. Insufficient coverage stops as `not_trained`.
Training labels are five actual packet-counter rates: UE6 unique first failure
plus overflow, UE6 completion-latency sum, UE6 completions, network DL goodput,
and UE2 unique first failure plus overflow. They are outcome-window rates
including carryover, not arrival-cohort failure probabilities.

After training, run a fresh matched frozen pilot:

```bash
python3 -m experiments.run_scheduler_conflict_study \
  --phase evaluate --output outputs/scheduler_conflict_case_20261105 \
  --throughput-model outputs/x08_load2x_recovery_20261105/throughput_mlp_model.json \
  --urllc-model outputs/scheduler_conflict_train_20261105/urllc_rf_model.json \
  --eval-seconds 1 --eval-seed 55 --mitigation-seed 1701 --eval-workers 3
```

`--eval-workers` accepts 1 (default), 2 or 3. OFF always finishes first and warms
the shared geometry cache; only the three frozen controller cases run in
parallel. Training stays serial. Progress writes use a shared lock and atomic
replacement, and final case order is deterministic. A worker failure keeps the
study incomplete and waits for all submitted cases before reporting failure.
This changes orchestration only: model bytes, inputs, duration and audits stay
the same.

`--dry-run` prints commands and starts nothing. `--phase all` is available when
the same frozen source can collect and evaluate sequentially. Separate phases
re-audit the original train receipt, exact measured rows, applied ACKs, packet
events, model and recording hashes. Collector and evaluation source fingerprints
are explicitly separate; all four evaluation arms must share one source.
Training/evaluation seed and duration differences are allowed, while configured
business/radio/deployment and mobility-model parameters must match.

The four arms are OFF, X08, URLLC RF, and BOTH with `uniform_random` mitigation.
Each cell selects one whole scheduler request for 100 ms from canonical source
order. Mode and complete served-UE weight map travel together. Per-cell random
streams use seed 1701, independent of the network seeds. Epochs and individual
draws are counted separately from repeated actuation receipts. Empty-cell mode
competition is reported but cannot alone establish occupied network conflict.

The pilot thresholds are declared before evaluation:

* X08: at least 5% network DL goodput gain relative to matched OFF.
* URLLC RF: UE6 mature unique deadline/overflow failure fraction at most 1%.
  Native UE6 is already at a zero-failure floor in the known seed55 case;
  satisfying this target is preservation, not improvement over OFF.
* Useful protection: at least 20 percentage points fewer UE6 failures than X08.
* Post-arbitration target harm: BOTH loses more than 2% goodput versus X08, or
  increases UE6 failures by more than 2 percentage points versus URLLC RF.

All-six delivery ratios, UE2 outcomes, original 10 ms mature packet-ID failure
unions, pending/overflow and completed-only mean/p50/p95 remain visible. Offline
QoS diagnostics use 95% native delivery and at most 5 percentage points failure
degradation; they are reported separately from throughput-first objective gains.
Packet-byte conservation, every trace slot and identical exogenous arrivals are
audited. Endogenous MEC-result bytes may legitimately shift.

The evaluator replays both strict JSON models on the actual logged feature
contexts and verifies all forecasts and declared selection rules. Prediction
counters alone cannot establish learned control. Training additionally requires
actual applied source-provenance ACKs no later than the public confirmation
timestamp, complete held maps, and 20 ms wash-in. Raw packet events independently
reconstruct all five RF labels over `(start_slot, end_slot]`.

Every completed simulation exports its own `network-replay.html`. The fresh
study directory contains atomic `acceptance.json`, immutable case logs and
`comparison-data.json` with KPI comparisons, per-UE metrics, selection epochs,
model/effect proof and hashes. Seed55 is calibration, not independent validation.
One short run supports no significance claim. Structural direct competition,
individual target satisfaction and observed target harm are separate findings;
this experiment does not claim an indirect conflict. Longer or additional seed
runs require an explicit new budget; the driver never schedules them itself.

## Recorded pilot

The [recorded acceptance](scheduler_conflict_acceptance.json) passes the
predeclared direct-conflict case gates. Each case simulated 1 second at 2× load.

| Frozen case | Network goodput (Mbit/s) | All-UE own-deadline/drop failures | URLLC 10 ms failures | URLLC completed-packet p95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| OFF | 48.35 | 49.49% | 0% | 0 |
| X-08 | 57.16 | 18.34% | 46.74% | 133 |
| X-12 URLLC | 48.35 | 49.49% | 0% | 0 |
| BOTH, random | 54.98 | 26.53% | 30.62% | 100.5 |

Network statistics cover all six UEs with their original 10/50/100 ms budgets.
URLLC statistics pool UE2 and UE6: failures divide summed packet counts, while
latency quantiles use the union of original completed-packet samples. Neither
UE failure percentages nor UE p95 values are averaged. The network aggregate
shows that X-08 improves overall timeliness while sacrificing URLLC. Traffic
delivered/offered and demand-normalized byte-delivery Jain fairness are secondary
network metrics; neither measures deadline satisfaction.

The RF genuinely selected native RR, preserving the baseline floor. BOTH lost
3.83% goodput relative to X-08 and increased UE6 failures by 61.24 percentage
points relative to X-12 URLLC (30.62 points after pooling both URLLC UEs).
UE6 remains the controller's target and a per-UE diagnostic, not the full-network
evaluation population. Twenty occupied-cell conflict epochs actually
executed, including ten in UE6's cell. Repeated held-action receipts are not
independent trials. A recorded 0 ms completion means delivery in the same
simulation slot, not zero physical end-to-end latency.
