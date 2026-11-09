# MEC budget control and random direct-conflict mitigation

`mec_offload_budget_qgbdt` is a Chongyu-pool xApp inspired by paper X-13.
It uses actual quantile gradient boosting on measured, executed task cohorts.
Its UE routing control is an adaptation of the paper's server admission control.
Compact KPI/RCP metadata is in `ric/near_rt/pools/chongyu/mec_budget.json`.

The expected UE4 offload budget is 0.75 GFLOP/s. With 20 tasks/s and
0.05 GFLOP/task, the analytic filter permits routing probabilities 0 or 0.5.
QGBDT ranks their lower service-utility quantile. Cold start and the explicit
`rules` mode use the analytic rule and do not claim learned inference.
Actual randomized route counts are reported separately from expected demand.

Run isolated online collection and the matched OFF/X01/BUDGET/BOTH study:

```sh
bash scripts/orin/run-mec-budget-study.sh --output outputs/my_mec_study
```

Training starts with four simulated seconds and is capped at six total.
A learned deployment requires 12 measured rows, six per feasible action,
and a genuine fit. Insufficient coverage is labelled `rules_only`.

After a trained model has been placed at
`outputs/models/mec_offload_budget_qgbdt/model.json`, run the frozen pair:

```sh
bash scripts/orin/run-mec-budget-random.sh --duration-s 1 --seed 55 \
  --output outputs/my_random_pair
```

Random mitigation is opt-in. It selects one conflicting UE offload request
uniformly every 100 ms with independent RNG streams, then preserves that
request within the window. It does not blend values or discard other UEs'
requests. Logs link raw candidates, random draws, submitted requests and actual
execution. Expired, withdrawn or obsolete queued selections cannot take effect.
This implementation explicitly rejects unsupported conflicting fields.

Structural M1 evidence and post-mitigation KPI compliance are separate outcomes.
Random mitigation may satisfy both intent contracts. Finite win counts and
actual execution durations need not be exactly 50/50. No statistical or
paper-level performance claim follows from one short calibration run.

## Initial native-case validation

The frozen source is `ad49f113051fb568795375648a2c4b30b0bc8ac5b5d51a2ae263d64167ae66a4`.
Four simulated seconds produced 13 genuine online rows; the fitted 12-row
prefix has six rows per action and one actual 32-tree quantile fit. Frozen
BUDGET/BOTH runs made four/three new inference batches without changing the
artifact. The model selected 0.5 throughout, matching the analytic rule; this
does not demonstrate added value from learning.

Matched 1 s runs used network seed 55 and predeclared mitigation seed 1701:

| Arm | UE4 energy, J | Expected offload demand, GFLOP/s |
|---|---:|---:|
| OFF | 0.836245 | 0.500 |
| X01 | 0.721560 | 1.000 |
| BUDGET | 0.836245 | 0.500 |
| BOTH_RANDOM | 0.818085 | 0.625 |

Structural M1 and independently effective isolated intents were verified.
The random pair met the 0.75 GFLOP/s budget and task/delivery guards, but
saved only 2.17% UE4 energy versus the 5% contract, compared with X01's 13.71%.
Energy covers the whole run; workload/task scoring covers mature arrival
cohorts [0.1,0.9) s. All 16 scored UE4 tasks finished on time. Realized routing
was 12 tasks, or 0.75 GFLOP/s, distinct from the expected 0.625 value.

There were 11 random selections including an undeployed terminal selection.
The ten deployed windows selected BUDGET eight times and X01 twice; the
per-draw probability remained 0.5. All 1,950 UE4 applied acknowledgements
matched their selected value. Ten old-window messages were rejected;
four changes between different parameter values showed no stale overwrite.
OFF and standalone X01 packet metrics, UE service reports and energy
components exactly match the preceding source's seed-55 results.

See `docs/rt_sys/mec_budget_random_acceptance.json` for the compact result.
Raw traces, online events, model hashes, five HTML audit results and the full
receipt remain in the AGX2 private workspace's `outputs/mec_random_pilot/`.
The model is experimental; this short calibration is not an independent
long-run generalization test. Native defaults keep xApps OFF and arbitration
at priority until the random policy is explicitly selected.
