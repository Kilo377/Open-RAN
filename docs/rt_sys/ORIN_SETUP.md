# Jetson AGX Orin deployment (agx1)

Project: `/home/lboroldn/Desktop/oran_slot_sim_sionna`, branch `open_ran_siona`.
The isolated `.venv-orin` uses Python 3.13.5, Sionna/RT 2.2.0, PyTorch 2.9.1
CPU and CuPy 14.2.0. Exact Python dependencies: `requirements-orin.lock.txt`.
The machine runs Jetson Linux R36.4.4 / Ubuntu 22.04 ARM64; nvcc is CUDA 12.6.
No system driver was replaced.

The preferred runtime uses Orin CUDA through CuPy for CIR-to-CFR reconstruction
(delay/Doppler phase and path summation), SVD, RZF precoding, interference
computations and LMMSE SINR. The existing project equations are preserved.
Sionna RT geometry/ray tracing uses LLVM CPU because the Jetson driver lacks
OptiX. Actual Sionna NR encoding/decoding and SYS transport sizing remain CPU.

The tested upstream Torch 2.9.1+cu126 Python 3.13 ARM64 wheel detects Orin but
fails actual operations with `no kernel image is available for execution on
the device` (listed CUDA architectures: sm_80/sm_90). The JetPack 6 native
Torch index inspected during setup provides Python 3.10 wheels, whereas this
project uses Sionna 2.2's supported Python >=3.11 environment. The incompatible
test venv was removed; failure logs remain in `outputs/setup_orin_gpu/`.

```bash
cd /home/lboroldn/Desktop/oran_slot_sim_sionna
bash scripts/orin/run.sh --cells 2 --ues 4 --slots 40
bash scripts/orin/run.sh --scene munich --cells 3 --ues 12 --slots 40
```

`run.sh` first executes real CUDA solve/SVD checks. If they fail, it prints the
reason and selects CPU before simulation starts. It does not silently restart
a failed or partially completed simulation. Use `scripts/orin/run-gpu.sh` to
require the GPU path, or `scripts/orin/run-cpu.sh` to explicitly use only CPU.

For manual use:

```bash
source scripts/orin/activate-gpu.sh
python scripts/orin/preflight.py
python experiments/run_rt_sys.py --device cpu --rt-device cpu --algebra-backend cupy
```

`--device` selects Sionna PHY/SYS. `--rt-device` independently selects RT;
omitting it preserves the original device selection. `--algebra-backend cupy`
enables CUDA algebra and compact path reconstruction. Omitting it preserves
the Torch implementation. GPU outputs get a `_cupy` suffix if no explicit
output is supplied. Default feedback remains `nr_direct`, with actual NR
CRC/bit checking, 20 decoder iterations and no online BLER/AMC probes.

Activation selects system LLVM 15 and preloads system libstdc++ because Conda's
older runtime lacks GLIBCXX_3.4.30. The project-scoped `sitecustomize.py` selects
LLVM before Sionna imports Mitsuba. Conda/system libraries were not modified.
GPU and CPU paths share the same venv; `torch.cuda.is_available()` is false
because Torch is CPU-only. CuPy's actual CUDA execution is recorded separately
in `execution_devices`, `algebra_gpu_name`, `algebra_compute_capability` and
CuPy allocation fields. `peak_cuda_bytes` remains the Torch-only metric.

Recreate on this machine with `bash scripts/orin/bootstrap.sh`. This requires
the existing Python 3.13, system LLVM 15 and JetPack CUDA for the optional GPU
path. Do not use the CUDA 13 Spark lock or Spark bootstrap on this Jetson.

Verification outputs: `outputs/setup_orin_gpu/`. These are deployment and
numerical smoke checks, not long experiments or real-time performance claims.
Timings from concurrently executed validation jobs are not a benchmark.

Completed verification: 467 tests passed, 3 skipped, 2 subtests passed.
The existing Sionna high effective-code-rate warning remains in the test log.

street: 40 slots, 38885 delivered bytes, 122 actual decodes. All compared ACK/MCS/delivery/HARQ/scheduling/queue results match CPU; maximum SINR difference 2.0673637e-05 dB.

munich: 40 slots, 85832 delivered bytes, 359 actual decodes. All compared ACK/MCS/delivery/HARQ/scheduling/queue results match CPU; maximum SINR difference 0.0023101355 dB.

The largest SINR difference was checked separately: corresponding CFR relative L2 error is 5.503e-8. The paired-run SINR tolerance is 0.01 dB; discrete decisions match exactly for these smoke cases.

The default CUDA selection and forced GPU-unavailable CPU fallback were both verified. Machine-readable results: `outputs/setup_orin_gpu/setup_summary.json`.

3D viewing is available in a separate `.venv-orin-gui` environment. The official
`sionna-rt-gui==0.1.1` requires Sionna RT 1.2.2; it does not share or downgrade
the simulator's RT 2.2 environment. `bash scripts/orin/gui.sh` opens the native
GUI on the AGX desktop (DISPLAY :1 for the observed logged-in session), with
OpenGL rasterization, LLVM propagation, and automatic heavy radio-map/path
recomputation disabled initially. Configuration: `configs/orin/gui.yaml`.
This GUI is an independent scene editor, not a live view of the closed-loop
simulator. GUI dependencies: `outputs/preview/requirements-gui.lock.txt`.

The HTML views under `outputs/preview/` are generated using the simulator's
actual Sionna RT 2.2 scene, project BS/UE initial positions and real paths.
They support browser rotation/zoom/pan and are static initial-state previews;
widget JavaScript is loaded from the standard widget CDN. The standalone GUI
and browser preview do not stream slot-by-slot simulator state.

The project's live execution viewer is now available with
`bash scripts/orin/live.sh` from the remote desktop. Its default run is Munich,
3 cells / 12 UEs, 400 slots, fixed baseline control, `nr_direct` feedback and
CuPy algebra when the existing CUDA preflight succeeds. Extra runner arguments
may override the scenario, UE count or slot count. Closing the window stops
that demo run; completed/failed/stopped runs are labeled and are not replayed.

This viewer uses Polyscope/OpenGL from `.venv-orin-gui`, but does not import
that environment's Sionna or run a second radio simulation. The RT 2.2 runner
exports its actual transformed scene meshes and atomically publishes actual
slot results, physical-time UE positions, serving associations and execution
devices through `--live-state outputs/.../live.json`. Rendering and physics
run in separate processes. Green lines are serving associations, not traced
multipath rays. There is no artificial UE motion or claim of wall-clock
real-time physics. The display holds the latest completed slot during the
next slot's computation and shows data age, simulated time and telemetry wall
time. Run artifacts and logs are under `outputs/live_project/<timestamp>/`.

Live validation: 11 targeted tests passed. With and without telemetry, paired
4-slot / 2-cell / 4-UE CuPy/NR runs produced byte-identical full RT traces,
packet/HARQ events, UE/cell metrics and identical evaluation summaries.

Explicit platform selection is supported by `experiments/run_rt_sys.py`:
`--platform dgx` selects CUDA/OptiX RT and PyTorch CUDA; `--platform agx`
selects the existing Orin hybrid path, with a recorded CPU algebra fallback
only if its initial CuPy probe fails. The Orin run wrappers now use this AGX
profile. Do not activate the Orin environment on a DGX; use its compatible
CUDA environment. See `README_SIONNA_RT_SYS.md` for commands and prerequisites.

Platform-switch verification on this AGX: 485 tests passed, 3 skipped,
2 subtests passed; one existing Sionna high-code-rate warning. The new AGX
profile, legacy hybrid invocation and explicit AGX CPU profile had identical
4-slot evaluation metrics (3916 bytes delivered, 15.664 Mbps). DGX option
resolution and strict failure behavior are covered; actual DGX/OptiX hardware
execution still requires verification by the DGX collaborator.

The live viewer now provides a unified read-only network workspace with
Overview / UEs / Cells / Config / Control tabs and bounded throughput/queue
trends. Integrated, Network only and Scene only views consume the same actual
slot stream; selecting a UE can focus its actual position in 3D. Effective
runtime controls, pending pipeline requests and execution receipts remain
separate. Baseline empty envelopes are labeled so receipts are not mistaken
for parameter modifications. No manual actuation endpoint was added.

Workspace verification on this AGX: 487 tests passed, 3 skipped, 2 subtests
passed, with the existing high-code-rate warning. Expanded monitoring produced
byte-identical RT traces, packet/HARQ events, UE/cell CSVs and evaluation
summaries in paired 4-slot runs. Unit checks cover effective vs pending vs
rejected controls, observer immutability, bounded complete-slot history and
configuration allowlisting. Optional GUI-only dependency: requirements-viewer.txt.
