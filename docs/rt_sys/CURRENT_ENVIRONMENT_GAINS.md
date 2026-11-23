# Current-environment P–K calibration

`experiments/current_environment_calibration.py` runs a fixed experimental
instrument over the sealed current simulator. It uses normal RIC admission,
actuation delay and executed-state recording. No trained xApp selects an action,
and no combination evaluation result or intent Metric enters fitting.

The protocol pins all four network profiles, physical source, model catalogue
and stimulus driver. Three independent seeds share a 2,040-slot native prefix;
measurement covers slots 2,441–4,440, followed through slot 4,840. Pairing stays
on the same AGX. The 1 s observation contains a complete URLLC traffic cycle.
MEC latency uses mature task-birth cohorts rather than a cumulative run mean.

| RCP | Fixed stimuli |
|---|---|
| MEC offload probability | 0, 0.25, 0.5, 0.75, 1 across the MEC population; native vector retained as reference |
| DRX timers | 20/40/80 ms cycles; 1/8/16 ms inactivity; exact native→20/4/1 and native→80/4/1 MEC-population profiles |
| Downlink scheduler | Round-robin, latency-aware, QoS-deadline categorical contrasts |
| Scheduling priority | MEC 0.5/1/2 under latency-aware; URLLC 0.5/1/2/4 under QoS-deadline; efficiency group 0.5/1/2 under latency-aware |
| BS sleep mode | Active, micro-sleep, light-sleep, deep-sleep contrasts |

Each seed contributes one slope or paired contrast. Slots and individual UEs
are not independent statistical replications. Scalar fits retain sampled levels,
residuals and confidence intervals; discrete settings use named contrasts.
An unobserved completion latency remains missing, and a measured zero remains
an observed zero within this horizon. Raw signs and physical units are retained;
the graph displays unsigned, normalized influence strength.

The one-shot stimulus has a finite admission lease. The manager can subsequently
reject duplicate refreshes as expired while leaving the already executed
configuration in place. The auditor accepts only those late refresh rejections,
records their number, and separately verifies the entire requested vector on
every scored slot. Other rejections or state mismatches invalidate the run.

`experiments/fit_current_environment_gains.py` accepts only sealed, SHA-verified
local mirrors. It checks paired physical prefixes, exogenous traffic, task
births, complete recorded slots and held parameter states. The gain matrix is
admitted by `smo/current_calibration.py` only when its certificate matches the
runtime recipe, raw measurement file, source and profiles. Historical matrices
remain separate. Scheduler-conditioned weight gains are selected using the
proposed scheduler context; gains from incompatible scheduler modes are not
added together.

Run on the Mac from the repository:

```sh
uv run --offline --with numpy python -m experiments.fit_current_environment_gains \
  --study /absolute/path/to/current-calibration-study
```

Raw compressed traces, receipts, manifests, checkpoints and fit diagnostics stay
under the study directory in `Desktop/ORAN/SMO/`. Remote results and all caches
use owned RAM directories. The mirroring command never removes remote raw data.

Numerical evidence is copied and SHA-verified first (the complete per-slot trace, all control receipts, configuration, stimulus, initial conditions and final counters). This is sufficient to run the full auditor; no slots, UEs or metric samples are omitted. Large redundant PHY diagnostic logs remain intact remotely and continue their full backup after each host finishes. The local proof distinguishes a complete numerical mirror from a complete raw mirror.
