# Context-dependent parameter–KPI weights

`experiments/context_identification.py` measures fixed control interventions.
It does not train, tune, enable or select the released xApps. Their shadow
proposals are recorded in separate instrumentation replays and never applied.

The primary design keeps the accepted moving six-UE, three-cell Munich
fast/system model, radio capacity and physical settings unchanged. Three
independent seed blocks are measured separately in each context:

| Context | Downlink packet traffic | URLLC arrival profile |
|---|---|---|
| `native_1x` | Native | Steady |
| `steady_2x` | Double native | Steady |
| `burst_2x` | Same mean as `steady_2x` | 4× peak, 20 ms on / 60 ms off |

Two additional alternating-profile pilots are diagnostic, not a fourth complete
coefficient matrix. The MEC task-source profile and radio settings remain fixed.
Burst pressure therefore does not gain radio capacity or a more generous deadline.

| Parameter | Tested values / transitions | Scope |
|---|---|---|
| MEC offload probability | 0, .25, .5, .75, 1 | Both MEC sources |
| DRX inactivity timer | 0, 1, 4, 8, 16, 40 ms | All DRX-enabled UEs |
| DRX cycle | 20, 40, 80, 160 ms | All DRX-enabled UEs |
| Scheduling priority | MEC: .5, .75, 1, 1.5, 2; URLLC: .5, 1, 1.5, 2, 3, 4 | Respective service group |
| Downlink scheduler | Round-robin → latency-aware / QoS deadline | All cells; weights 1 |
| BS sleep policy | Active → micro / light / deep sleep | All cells |
| Joint DRX profile | Cycle/inactivity 40/8 → 80/1 ms | All DRX-enabled UEs |

There are 28 unique primary conditions per seed. The supplemental fixed
UE8 priority component in the steady double-load context covers an actual
recorded X-08 proposal: .5, 1, 1.5 and 2, other weights 1, latency-aware mode.
Its target KPIs remain population aggregates. It is explicitly an identified
member component, not a claim of interchangeable per-UE derivatives.

Every primary run is 1.8 s / 3,600 slots. Controls switch at 100 ms through
normal RIC admission and delay. Counter-difference metrics score 400–1,600 ms.
Task metrics use 48 generated tasks in 24 matched mature birth cohorts, followed
until 1,800 ms. Pending tasks and completion counts remain recorded. Conditional
completed-task latency is not an uncensored service-level latency guarantee.
The URLLC event rate uses the score interval; the separate cumulative failure
ratio includes warm-up. Each estimate carries its actual `metric_window`.

`smo.context_identify` verifies source pins, full recorded controls in every
scored slot, packet-byte conservation, matched exogenous arrivals, task births,
checkpoints and frozen learning. A service-weight value follows the UE across
handover. Only an explicitly permitted delayed-refresh scope rejection can be
tolerated, after the actual held value is verified in every scored slot.

Numeric ranges use seed-specific affine fits, summarized across seed slopes.
Categorical transitions use paired seed differences. Independent seeds, not
slots or grid points, determine the Student interval. Fits retain all signs,
constant responses, residuals, leave-level-out error and undefined outcomes.
Intervals are exploratory per response, without familywise significance claims.
Three seeds and a finite observation window can still leave uncertain directions.

Signed reference importance is `gain × tested parameter span / |reference KPI|`.
A zero reference uses an explicitly labeled observed-magnitude/one-unit fallback.
Positive means the KPI increases; it does not mean improvement. These are
context-dependent first-order reference weights for the conflict module, not
predictions of actual combined xApp outcomes. The graph never pools traffic
contexts or silently substitutes zero for missing evidence.

Remote source, caches, temporary files and output stay in an owned `/dev/shm`
directory. Both devices retain their global environments. A local coordinator
mirrors raw records continuously and verifies sealed SHA-256 checkpoints;
completed data and trained models are retained. Deadline and storage guards
stop admissions before the analysis reserve. No global cache cleanup is needed.

```sh
uv run --offline --with numpy python scripts/audit-context-progress.py --artifact /absolute/study
uv run --offline --with numpy python -m smo.context_identify --artifact /absolute/study --context burst_2x --output /absolute/new-version/burst_2x
uv run --offline --with numpy --with matplotlib python -m smo.context_report --analysis /absolute/new-version --output /absolute/figures
python -m smo.scenario_service --manifest /absolute/scenario-manifest.json --port 53505 --session-key-stdin
```

The scenario service shares one session-only DeepSeek connection and a
revision-checked preset memo store. Frozen intent ledgers stay separate by
network context. Only the retained native view can use the existing native
deployment pool; pressure views cannot accidentally execute a native-load trial.
`--verify-language --replace-pid PID` stages file/graph/ledger checks without
starting deployment managers, verifies a pure interval parse, and only then
retires the explicitly identified SMO listener on the requested port. The
verified session stays in memory for all scene views. A failed preflight leaves
the old service running. Supply the key through an stdin pipe from a masked
credential prompt, never a command argument or a saved manifest field.
