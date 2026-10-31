# AGX Orin on JetPack 7.2

JetPack 7.2 / L4T R39 uses Ubuntu 24.04, Python 3.12 and CUDA 13.2. Keep its
dependencies separate from the existing JetPack 6 / Python 3.13 environment.
`requirements-orin-jp7.txt` selects PyTorch 2.12.0+cu132 from the official
PyTorch index, Sionna 2.2.0, Mitsuba 3.9.1, Dr.Jit 1.5.0 and CuPy CUDA 13.
`requirements-orin-jp7.lock.txt` records the complete resolved environment,
including the Polyscope viewer, and is the file used by bootstrap.

Inside a fresh checkout:

```bash
bash scripts/orin/bootstrap.sh
# Optional desktop viewer:
.venv-orin/bin/python -m pip install --no-cache-dir -r requirements-viewer.txt
```

Bootstrap selects the R39 lock file from `/etc/nv_tegra_release` and
installs only into `.venv-orin`. `ORAN_PYTHON` can select a compatible
interpreter; this lock is validated with Python 3.12. It does not install or
upgrade the system driver or CUDA toolkit.
Activation discovers the installed LLVM shared library, respects an existing
valid `DRJIT_LIBLLVM_PATH`, and uses CUDA's `lib64` link across the older
`aarch64-linux` and newer `sbsa-linux` toolkit layouts.

NVIDIA's [JetPack 7.2 PyTorch support discussion](https://forums.developer.nvidia.com/t/how-do-i-correctly-install-pytorch-on-jetpack-7-2/372773)
includes actual CUDA execution with the upstream cu132 wheel. Still verify
real kernels on each device; CUDA detection alone does not establish that a
wheel supports the GPU. A working Torch CUDA installation does not establish
OptiX support. The existing AGX platform keeps ray tracing on LLVM CPU.

PyTorch 2.12 pins cuSPARSELt 0.8.1, whose ARM wheel contains the incorrect
internal tag `manylinux2014_sbsa` despite shipping AArch64 binaries. Newer pip
reports a platform error. This preset pins the Ubuntu 24.04 baseline pip 24.0
as the temporary workaround described by a PyTorch collaborator in
[upstream issue #202632](https://github.com/pytorch/pytorch/issues/202632).
It does not rewrite vendor metadata or ignore dependency-check failures.
Actual CUDA and NR execution must still pass; dependency checks alone are
insufficient. Revisit the installer pin once the upstream Torch dependency
selects the metadata-corrected cuSPARSELt release.

On the tested R39.2.1 AGX, this Torch wheel still emits an SM 8.7 compatibility
warning. The warning is retained, not suppressed. Elementwise CUDA operations,
complex solve/SVD, CuPy kernels and actual Sionna NR CUDA decoding passed.
The relevant regression suite passed 159 tests, including its CUDA cases,
with that architecture warning and the existing high-code-rate warning.
This establishes the exercised project paths, not universal operator support.
The initial regression used the device's existing 50 W mode (CPU limit about
1.5 GHz). The user then explicitly selected maximum performance: MAXN mode 0
plus `jetson_clocks`, yielding CPU 2.2016 GHz, GPU 1.3005 GHz and EMC 3.199 GHz.
The interrupted 50 W benchmark is retained separately; it is not a completed
timing result. Bootstrap itself does not change power policy.

For an explicitly requested maximum-performance deployment, run:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
sudo jetson_clocks --show
```

`nvpmodel` retains its selected mode. `jetson_clocks` alone is temporary across
reboots. `scripts/orin/oran-max-performance.service` can be installed as an
explicit startup setting to restore maximum clocks after the system nvpmodel
service. It follows the retained power profile rather than changing it, and
does not replace automatic fan control or thermal protection.

```bash
sudo install -m 0644 scripts/orin/oran-max-performance.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now oran-max-performance.service
```

## Execution modes

The existing measured CPU profile remains the default:

```bash
bash scripts/orin/run-six-ue.sh --slots 400
```

An explicit alternative uses CUDA NR and Torch CUDA algebra while retaining
LLVM CPU ray tracing, the same six UEs, ray settings and geometry interval:

```bash
bash scripts/orin/run-six-ue-cuda-nr.sh --slots 400
```

This alternative uses `--device cuda --rt-device cpu --nr-execution batched`.
It intentionally does not select `--platform dgx`, which requires CUDA/OptiX
ray tracing, or `--platform agx`, whose established profile requires CPU NR.
CPU process-parallel decoding is replaced by same-slot batched CUDA decoding.
There is no automatic fallback if CUDA NR fails.

Use distinct output/cache directories, the same seed and moving UE workload,
and serial timing runs when comparing these modes. Cross-device floating-point
and random sampling details can change decode outcomes; a speed comparison
must retain packet/transport-block accounting and report any KPI differences.
Neither a successful import nor this optional launcher is a performance claim.

## Measured 400-slot comparison

On 2026-10-31, two serial runs used MAXN plus fixed maximum clocks, seed 1,
six moving UEs, 5 ms geometry updates, and separate cold propagation caches.
Both simulate 0.2 s. The physical manifests, business configurations and all
328 arrays in 41 ray-path files were identical. Each run passed the transport
block, packet-byte conservation, delivery and RF-power accounting audit.

| Execution | Complete launcher | Simulation | Network goodput |
| --- | ---: | ---: | ---: |
| Batched CUDA NR + Torch CUDA algebra | 241.44 s | 230.90 s | 29.96256 Mbps |
| Six CPU NR workers, two threads each | 362.88 s | 350.67 s | 29.99600 Mbps |

For this one short workload, CUDA reduced launcher time by 33.46% (1.50x
speedup). The CUDA entry is therefore recommended on the tested JetPack 7
device; the established CPU preset remains available for older devices.
Goodput differs by about 0.11%; device-specific Torch random streams and
floating-point execution do not promise identical CPU/GPU ACK histories.
This is a timing comparison with matched physical inputs, not a claim of
bitwise equivalence or long-run QoS validation. Initialization and codec
construction are included. No controller or scheduling algorithm was tuned.
