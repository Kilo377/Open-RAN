# Serial 30-second xApp validation

`experiments/run_xapp_long_validation.py` runs eight simulations, one child at a
time: X-01 OFF/frozen, X-02 OFF/frozen, X-08 OFF/frozen, and X-12 MEC OFF/frozen.
Each child simulates 30 seconds (60,000 slots at 0.5 ms). It keeps the accepted
six moving UEs, three cells, fast PHY, and existing algorithms. X-01 and X-12
use native 1× traffic; X-02 and X-08 use the prior explicit 2× profile and the
same enabled AP energy model. Each pair has one declared seed and matched
deployment, channel, traffic and physical capacity. This extends evaluation;
it does not collect new training data or tune actions after seeing results.

`--reuse-matched-baselines` optionally avoids repeating identical deterministic
OFF simulations: X-08 uses the audited X-02 OFF recording, and X-12 MEC uses
the audited X-01 OFF recording. This gives four paired comparisons from six
physical runs and eight explicitly identified logical cases. Reuse requires
identical command parameters after removing only the output path; it copies
neither raw files nor simulation state and is not an independent repetition.

The four trained JSON artifacts must be copied to a private directory under
this checkout's `outputs/`. Its `models.json` declares their exact hashes:

```json
{
  "models": {
    "x01": {"path": "x01.json", "sha256": "the-file-sha256"},
    "x02": {"path": "x02.json", "sha256": "the-file-sha256"},
    "x08": {"path": "x08.json", "sha256": "the-file-sha256"},
    "x12_mec": {"path": "x12_mec.json", "sha256": "the-file-sha256"}
  }
}
```

Preview the exact commands without creating a queue or starting simulations:

```bash
.venv-orin/bin/python -m experiments.run_xapp_long_validation \
  --output outputs/xapp_30s_validation --model-dir outputs/xapp_frozen_models \
  --duration-s 30 --seed 55 --reuse-matched-baselines --dry-run
```

Run in the private GPU-enabled checkout. An outer `nohup`/`setsid` session keeps
the queue alive after SSH disconnects; the Python queue PID is recorded in
`queue-status.json`. The queue calls the existing activation/run shell script
for each simulation, so it does not modify system packages or global settings.

```bash
nohup setsid .venv-orin/bin/python -m experiments.run_xapp_long_validation \
  --output outputs/xapp_30s_validation --model-dir outputs/xapp_frozen_models \
  --duration-s 30 --seed 55 --min-free-gb 1 --estimated-case-gb 0.8 \
  --reuse-matched-baselines \
  > outputs/xapp_30s_validation.queue.log 2>&1 &
```

The immutable `queue-plan.json` contains the source/config fingerprint, model
manifest, model hashes, AP profile hash and all eight commands. A nonblocking
file lock prevents two queue processes from sharing an output directory. The
status file is atomically updated every ten seconds with the active PID,
elapsed wall time and the child's actual progress fields. Simulation progress
is taken from recorded slots, never inferred from elapsed wall time.

All eight cases share the geometry cache for the declared 30-second seed.
Data are gzip-compressed, and each completed simulation automatically exports
`network-replay.html`. Before each child starts, free space must cover both the
configured reserve and the estimated next case. The reserve is also monitored
while the child runs. Crossing it stops that process group and exits with code
75 and `blocked_disk_space`; no records or other user's files are deleted.
The estimate is a planning allowance, not a guarantee of exact recording size.

Resume with the same command and `--resume`. Completed cases are skipped only
after verifying their original audit checksum, exact file inventory and every
recording/HTML hash. A completed recording that changed is rejected. A completed
child whose queue was interrupted before auditing can be audited on resume.
An incomplete case is moved, with its log, into `attempts/` before restarting
that simulation from slot zero. There is no mid-simulation state checkpoint.
A still-live orphan child blocks resume so two simulations cannot overwrite
the same case. SIGINT/SIGTERM terminates the child process group, waits for a
bounded grace period, then kills remaining workers if necessary.

Post-case auditing streams the trace and receipts with bounded memory. It
verifies every ordered slot, all six packet byte balances, actual exogenous
traffic, frozen model counters and actual applied control receipts. The
source, AP profile and all model bytes must remain unchanged. Bad but valid
KPI results are retained; the queue has no profitability or success threshold
that silently schedules more seeds or training.

If no frozen choices or actual applied controls occur, the completed physical
data remain in the queue. The result is marked `controller_evaluable=false`
with explicit reasons; it is not credited as a demonstrated xApp effect. Missing
records, changed/unfrozen model history, inconsistent source or failed packet
conservation still stop the queue. Event and invalidation counters remain
visible for partial coverage even when some real controls did execute.

`audits/<case>.json` retains whole-run energy/goodput, control coverage and
task metrics. Long-horizon task completion is reconstructed from enriched
generation/terminal events. The last eight public cohort buckets are only a
secondary diagnostic: they cannot provide all 30 seconds of final outcomes.
Primary MEC delay pools completed offloaded tasks in all mature creation
cohorts starting after 100 ms; pending and failed outcomes remain explicit.
For a short smoke test with no mature tasks, the mean is `null` and the target
is `not evaluable`, not zero latency or a claimed success.

`comparison-summary.json` updates after each complete matched pair, including
raw KPI values and descriptive relative changes. This is a longer single-seed
realization, not multiple independent repetitions or a confidence interval.
The previously trained models can encounter new channel/queue contexts during
30-second mobility; the script reports outcomes without retraining or claiming
that frozen forecasts are calibrated in those new contexts.

## Frozen-model checkpoints for an active queue

`scripts/orin/checkpoint_xapp_evaluation.py` is a separate, stdlib-only sidecar.
It can be installed under `scripts/` without changing the running simulator's
source fingerprint or restarting the queue. Never modify the executed
`experiments/`, model files, queue plan or completed case folders while the
declared queue is active.

Create and verify a checkpoint immediately, using the exact source archive
that was retained when the queue started:

```bash
.venv-orin/bin/python scripts/orin/checkpoint_xapp_evaluation.py once \
  --queue outputs/xapp30s_20261105 \
  --source-archive outputs/source-dbcc9763.tgz
.venv-orin/bin/python scripts/orin/checkpoint_xapp_evaluation.py verify \
  --queue outputs/xapp30s_20261105
```

The default checkpoint directory is `<queue>/checkpoints/`. One immutable
model set saves all four trained JSON artifacts, their original relative
manifest, the queue plan, the AP profile, the source archive and checksums.
The archive's reconstructed source fingerprint must equal the queue plan.
Files are checksum-verified before and after copying, fsynced, and published
atomically; an existing valid model set is never replaced. The sidecar has
its own lock and does not take the running queue's lock.

For continued progress checkpoints, run `watch` in a detached session with
the same queue and archive arguments and `--interval-s 60`. It updates
`latest-progress.json` every minute and saves an immutable receipt in
`completed/<case>.json` for each observed completed logical group. Each receipt
references the exact original case audit checksum and frozen model-set ID.
Models are frozen during evaluation, so their parameter files are saved once,
not copied every minute. No large channel cache or raw record is duplicated.
The watcher stops after the queue completes, fails, is interrupted, or pauses
for disk space. Restart it alongside a resumed queue to continue checkpointing.
Stopping the watcher does not signal or stop the simulator.

Restore a model checkpoint only into a separate new or empty outputs directory:

```bash
.venv-orin/bin/python scripts/orin/checkpoint_xapp_evaluation.py restore-models \
  --queue outputs/xapp30s_20261105 --destination outputs/restored_xapp_models
```

This restores the original relative model layout and manifest, even if the
original input weights are missing. It refuses the active model directory,
its descendants, queue recording directories, and nonempty destinations.
Restoring into a new path does not directly resume the original queue: that
queue's immutable plan still names its original input paths. Original raw
case records and audits remain necessary to verify already completed groups.

The checkpoint kind is `frozen_model_and_group_progress`, with
`supports_mid_simulation_resume=false`. It does not save live UE/HARQ queues,
channel arrays, RNG states or pending controller actions. An interrupted
simulation therefore restarts from slot zero; completed, verified groups are
skipped by the main queue's existing `--resume` mechanism. Do not represent a
saved progress slot number as a complete simulator-state checkpoint.
