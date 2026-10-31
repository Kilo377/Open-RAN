# Validated AGX execution profile

The selected preset is `configs/rt_sys/agx_munich_6ue_depth2_fast.json`.
It keeps original UE IDs 0, 1, 2, 4, 6 and 8, their traffic and trajectories,
three cells, 3.5 GHz, 10 MHz, 24 active PRBs and 0.5 ms slots. UE speed is
1.5 m/s. The native scheduler and control algorithms remain unchanged;
the `chongyu` pool is selected and has no active xApps.

Run ten simulated seconds in an existing Orin environment:

```bash
bash scripts/orin/run-six-ue.sh --slots 20000 --seed 1 --ue-speed-mps 1.5 \
  --live-xapp-test --output outputs/agx_fast_10s \
  --live-state outputs/agx_fast_10s/live.json \
  --channel-cache-root outputs/agx_fast_10s/channel_cache
```

Use a new output directory for each run. The command requires neither package
installation nor system configuration changes. The live viewer can read this
run's `live.json` independently of the simulation process.

## Execution choices

* Geometry updates every 5 ms, bounded to 1 cm displacement, with Doppler still
  reconstructed each slot. RT retains depth 2 and 100,000 samples per source.
* RT uses one Dr.Jit LLVM thread, with geometry prefetched by a separate process.
  Multiple Dr.Jit threads triggered a queue assertion on the tested AGX.
* NR uses six independent CPU processes, each with two Torch threads, and the
  original 20 BP decoder iterations. The main process uses one Torch thread.
  HARQ state and causality remain in the main process.
* Small matrix algebra uses Torch CPU. CuPy GPU algebra was tested but was
  slower for this workload. CPU affinity is disabled after comparison.
* Same-slot physical reference measurements are reused with invalidation;
  ideal-CSI calculations omit unused duplicate receiver work. Packed GPU
  diagnostics and persistent GPU CFR storage remain disabled in this preset.

These are workload-specific choices, not a claim that CPU is generally faster
than GPU. DGX's explicit OptiX/CUDA platform selection remains available.

## Measured validation on 2026-10-30

Seventeen serial cases each simulated 400 slots (0.2 s), seed 1, with cold
propagation caches and a warm shared compiler cache. The earlier 10 s baseline
job was still running during these comparisons; absolute timing depends on
contention. The table compares runs with the same 5 ms geometry update period.

| Measurement | Scalar reference | Selected profile | Speedup |
| --- | ---: | ---: | ---: |
| Simulation wall time | 654.39 s | 264.14 s | 2.48x |
| Complete launcher wall time | 665.02 s | 276.47 s | 2.41x |

Simulation wall time fell by 59.64%. All 17 cases passed their fidelity checks:
40,800 UE-slot rows were compared, packet and HARQ event streams matched,
ray-path arrays matched, and each case performed 2,256 actual NR decodes.
The paired workload delivered 749,900 bytes (29.996 Mbps).

The selected implementation regression suite passed 114 tests, with two
Torch-CUDA skips and one existing high-effective-code-rate warning. A separate
400-slot check through the deployed default launcher completed in 271.27 s;
its packet/HARQ streams, physical manifest and reported packet, link, OLLA and
HARQ statistics matched the selected candidate.

## Interpreting the 10 s estimate

The historical 10 s run took 54,749.62 s (15 h 12 min). It used 0.5 ms geometry
updates, whereas this preset uses 5 ms. Scaling the selected 400-slot timings
to 20,000 slots suggests approximately 3.7–4 hours, or roughly 4x faster than
that historical run. This is an estimate, not a completed 10 s measurement.
Cold compiler caches, changing propagation, decode workload and machine load
can change the result.

The 5 ms geometry interval is an explicit propagation approximation and must
not be described as a mathematically identical implementation optimization.
The 2.48x implementation comparison above holds the interval fixed. A complete
long-run comparison must also inspect packet accounting, decode counters,
coverage changes and per-UE outcomes rather than accepting elapsed time alone.
