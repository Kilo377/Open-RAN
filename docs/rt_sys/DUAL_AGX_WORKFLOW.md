# SMO and dual-AGX workflow

The repository includes the Sionna simulation, six Chongyu xApps, the SMO UI,
and a two-worker SSH task queue. Each AGX runs one independent simulation at a
time. The manager assigns the next combination to an idle worker and retains
job identity, progress, execution receipts, comparisons and downloaded output.
Both workers can evaluate different combinations concurrently.

| Component | Implementation |
|---|---|
| Simulation entry, full/fast models | `experiments/run_rt_sys.py` |
| Slot simulation and control execution | `core/simulator.py`, `ric/controller_manager.py` |
| Trained actors | `ric/near_rt/pools/chongyu/`, `artifacts/xapps/chongyu/20261123/` |
| Knowledge graph and intent UI | `smo/`, `smo/web/` |
| SSH jobs, monitoring and retrieval | `smo/deployment.py` |
| Concurrent combination queue and recovery | `smo/deployment_batch.py` |
| Current profiles and P–K evidence | `artifacts/smo/20261123/` |

Conflict quantification uses the graph, intent intervals and recorded actor
proposals before deployment. Actual post-deployment KPI evaluation is separate;
it is not a fitting input to the conflict score. Final automatic combination
selection remains unimplemented.

## Local graph and archived gains

From the repository root, with Python available:

```sh
python -m smo --port 8891

# Inspect the archived current-environment gain matrix.
python -m smo \
  --influences artifacts/smo/20261123/calibration/influences.json \
  --candidate-selection semantic_influence_coverage_v2 --port 8891
```

Open `http://127.0.0.1:8891`. The standalone commands show contracts or supplied
statistical evidence. Attach `--replay /absolute/path/to/run` or
`--state /absolute/path/to/run/live.json` for actual network observations.
Natural-language parsing uses the existing DeepSeek interface; connect a
session key in the UI or set `DEEPSEEK_API_KEY` in the service environment.

## Two-worker configuration

1. Install the platform-specific simulator environment on each AGX using the
   existing [RT/SYS guide](../../README_SIONNA_RT_SYS.md). Keep each worker in
   its own project folder and use the same source and frozen model bundle.
2. Copy `configs/smo/dual-agx.example.json` to an operator-owned location under
   `outputs/`. Set both SSH targets, authenticated ControlMaster sockets,
   absolute project/interpreter paths, and a local output directory.
3. Install the selected model bundle on each worker:

   ```sh
   # Keep the legacy default available for workers without a runtime recipe.
   python scripts/orin/install-trained-xapps.py
   python scripts/orin/install-trained-xapps.py \
     --bundle artifacts/xapps/chongyu/20261123 \
     --output outputs/trained_xapps/20261123
   ```

4. Start the UI with the configured worker pool:

   ```sh
   python -m smo --replay /absolute/path/to/baseline-run \
     --influences artifacts/smo/20261123/calibration/influences.json \
     --deployment-config /absolute/path/to/pool-config.json \
     --intent-store outputs/smo/intents.json \
     --preset-store outputs/smo/intent-presets.json \
     --candidate-selection semantic_influence_coverage_v2 --port 8891
   ```

The example config describes connection settings only. Exact current-environment
reproduction also requires a freshly verified runtime recipe and baseline-origin
proof for the installed worker source. Register that recipe with the pool's
optional `runtime_recipe` field and use `python -m smo.scenario_service` for certified
environment/scenario manifests. Historical certificates in the snapshot retain
the original source hashes and absolute paths; they are not rewritten to claim
that a different checkout or worker has passed those checks.

The existing configured service and both running workers are unaffected by
these examples. Raw traces, live intent records, memo tags, credentials,
environments and caches are not part of the Git publication.
