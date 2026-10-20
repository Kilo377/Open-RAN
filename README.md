# ORAN5GGLOBALCOMM

A MATLAB-based 5G/O-RAN simulation workspace focused on baseline RAN performance, Near-RT xApp control, and Non-RT policy loop experiments. The project is script-driven and designed for reproducible KPI comparison, policy validation, and dataset generation.

The physical B210 / Quectel lab is documented separately in
[lab/oai-gnb](lab/oai-gnb/README.md). It pins a newer OAI gNB in an isolated
container and reuses the existing 5GC and SIM subscription.

## What This Project Includes

- **Core RAN simulation pipeline**: mobility, traffic, beamforming, radio, handover, scheduling, PHY, energy, and KPI stages.
- **Near-RT RIC / xApp framework**: xApp discovery/registration, action merge, conflict guarding, and KPI impact evaluation.
- **Non-RT RIC workflow**: policy trigger and policy application through A1-style data files.
- **Experiment entry scripts**: baseline runs, single xApp comparisons, xApp testbench sweeps, and network sensitivity scenarios.

## Repository Layout

- `oran-sim/run/`: main run entry points (best place to start)
- `oran-sim/core/`: simulation kernel and models
- `oran-sim/ric/nearRT/`: Near-RT RIC components
- `oran-sim/ric/nonRT/`: Non-RT RIC components
- `oran-sim/xapps/`: xApp implementations and `registry.json`
- `oran-sim/bus_A1/`, `oran-sim/core/bus_E2/`: A1/E2-related bus structures

## Quick Start

### 1) Prerequisites

- MATLAB (R2024+ recommended; ensure required toolbox capabilities are available)
- Run commands from the repository root

### 2) Set up MATLAB path

```bash
matlab -batch "run('oran-sim/run/setup_path.m'); disp('path ready')"
```

### 3) Run a baseline experiment

```bash
matlab -batch "run('oran-sim/run/setup_path.m'); run_xapp_baseline_kpi"
```

## Common Experiment Commands

```bash
# Baseline KPI run
matlab -batch "run('oran-sim/run/setup_path.m'); run_xapp_baseline_kpi"

# Single-xApp vs baseline comparison set
matlab -batch "run('oran-sim/run/setup_path.m'); run_xapp_single_compare_all"

# Batch xApp testbench (writes summary CSV)
matlab -batch "run('oran-sim/run/setup_path.m'); run_xapp_testbench"

# Non-RT policy loop experiment
matlab -batch "run('oran-sim/run/setup_path.m'); run_nonrt_general"
```

> Tip: if MATLAB cannot resolve a function name, run the script file directly (for example: `run('oran-sim/run/run_xapp/run_xapp_testbench.m')`).

## Outputs

- xApp testbench summary CSVs are written under `oran-sim/_results_xapp_testbench/`
- Some scripts write baseline/training CSVs under `oran-sim/xapps/*/`
- Runtime KPI traces are printed to console, and figures are generated for visualization

## Configuration

- Default configuration file: `oran-sim/run/default_config.m`
- Frequently adjusted fields:
  - `cfg.sim.slotPerEpisode`: simulation length
  - `cfg.scenario.numCell` / `cfg.scenario.numUE`: scenario scale
  - `cfg.nearRT.xappRoot`: xApp root directory
  - `cfg.nonRT.*`: Non-RT trigger, timeout, and policy/report paths

## Notes

- `cfg.nonRT.smoCommand` in `default_config.m` may contain a machine-specific absolute path; update it when moving to another machine.
- This repository is script-first rather than unit-test-first; treat `oran-sim/run/*.m` as scenario-level smoke tests.

---

If you are new to this project, start in this order:
1. `run_xapp_baseline_kpi`
2. `run_xapp_single_compare_all`
3. `run_xapp_testbench`

This sequence gives the fastest end-to-end understanding of KPI behavior and xApp effects.

## Intent Tutorial (SMO + Non-RT)

For an intent-driven Non-RT flow tutorial, see `docs/INTENT_TUTORIAL.md`.
