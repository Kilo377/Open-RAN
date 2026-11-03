# AGX1 GPU fast deployment and long-run recording — 2026-11-03

The original working tree and CPU environment are preserved. A separate worktree
holds the accepted fast code and a Python 3.10 GPU environment. System CUDA,
drivers and OS packages are unchanged.

Validated platform: Ubuntu 22.04.5, L4T R36.4.4, kernel 5.15.148-tegra, CUDA 12.6,
AGX Orin in MAXN, sleep targets masked. PyTorch 2.9.1 uses the JetPack 6/cu126
Jetson AI Lab wheel, compiled for `sm_87`. The generic ARM CUDA wheel detected
the GPU but failed its first actual kernel and was removed.

Jetson wheel: `torch-2.9.1-cp310-cp310-linux_aarch64.whl`, SHA256
`02fde421eabbf62633092de30405ea4d917323c55bea22bfd10dfeb1f1023506`.
Index: <https://pypi.jetson-ai-lab.io/jp6/cu126/torch/>.

Sionna/Sionna RT 2.2.0 declare Python >=3.11. This is a validated compatibility
deployment, not an upstream claim of Python 3.10 support. Their pure Python
wheels are installed with `--no-deps --ignore-requires-python`, after separately
installing compatible dependencies. Only two `typing.Self` imports are changed
to `typing_extensions.Self`; computational functions are unchanged. The scoped,
idempotent helper is `scripts/orin/backport_sionna_typing.py`. GPU environment
dependencies include NumPy 2.2.6, SciPy 1.15.3, Mitsuba 3.9.1, Dr.Jit 1.5.0,
Matplotlib 3.10.9 and h5py 3.15.1.

cuDSS 0.7.1.6 is installed within the virtual environment to satisfy the Torch
binary loader. Activation searches optional local runtime libraries after the
JetPack libraries, preserving native CUDA precedence. GPU dense/complex linear
algebra was **not accepted** on these existing JetPack libraries; replacement
SBSA math libraries were incompatible. The accepted fast profile continues to
use CPU RT, CPU small-matrix algebra and CPU SYS feedback, with **actual NR/LDPC
fallback on CUDA**. It does not enable full CUDA algebra.

CUDA matrix multiplication, cuDNN convolution (9.3) and actual NR/LDPC tests
passed. The final targeted project suite passed 52 tests. A real Munich scene
400-slot run completed in 31.51 s internally, with 77 actual CUDA NR decodes,
2,149 SYS samples and a self-contained replay whose audits passed.

## Recording long runs

```sh
bash scripts/orin/run-six-ue-cuda-nr.sh --model fast --duration-s 60 \
  --record-compression gzip --output outputs/fast_gpu_60s
```

The 60 s duration is 120,000 slots at 0.5 ms. Geometry, scheduler, measurements,
HARQ, traffic and control clocks are unchanged. Compression is optional and
defaults to `none`; it writes CSV/JSONL as `.gz`. The automatic HTML exporter
reads both formats. All records are retained. CSV export keeps one flattened
row at a time, avoiding a second full copy of long-run metrics in RAM. Status
updates every 1,000 slots and at the final slot expose `last_completed_slot`,
`simulated_seconds` and `simulation_wall_s_so_far`.

A paired 400-slot same-seed test reproduced packet, HARQ, RT trace, OLLA and all
CSV bytes after decompression. Control receipt values also matched; dictionary
key order varied between fresh Python hash seeds. Both HTML audits passed.
Total output was 16,503,237 B plain versus 3,573,995 B compressed (21.66%,
including the unchanged HTML). Compression, streaming CSV and replay tests
passed 37 tests. The deployed 60 s run is background work; launch is not a
claim that all 120,000 slots have completed.
