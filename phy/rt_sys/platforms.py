"""Explicit DGX/AGX execution profiles, independent of machine naming/CPU ISA."""
import sys


def check_agx_cuda():
    # Test real kernels: Jetson can report CUDA while a Torch wheel lacks sm_87.
    from scripts.orin.preflight import check
    return check()


def check_dgx_cuda_optix():
    import ctypes
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('DGX mode requires a CUDA-enabled PyTorch environment; no CPU fallback')
    try:
        ctypes.CDLL('libnvoptix.so.1')
    except OSError as exc:
        raise RuntimeError('DGX mode requires the NVIDIA OptiX driver library libnvoptix.so.1; check driver/container mounts') from exc
    # Also reject wheels that recognize the GPU but cannot execute its kernels.
    matrix = torch.eye(2, device='cuda', dtype=torch.complex64)
    torch.linalg.solve(matrix, matrix)
    torch.linalg.svd(matrix)
    torch.cuda.synchronize()
    return dict(gpu_name=torch.cuda.get_device_name(0), optix_driver_loaded=True,
        actual_cuda_solve_and_svd=True)


def resolve_platform(args):
    """Resolve once, before scene construction; no mid-run backend changes.

    Dry runs describe the requested profile without importing CUDA or OptiX.
    An explicit low-level flag may refine a profile only when compatible.
    Legacy calls without --platform keep their original device semantics.
    """
    selected = getattr(args, 'platform', None)
    device = getattr(args, 'device', None)
    rt_device = getattr(args, 'rt_device', None)
    algebra = getattr(args, 'algebra_backend', None)
    report = dict(requested=selected or 'custom', preflight='not_requested', fallback_reason=None)
    if selected is None:
        args.device = device or 'cuda'
        args.rt_device = rt_device or args.device
        return report
    expected = 'cuda' if selected == 'dgx' else 'cpu'
    if device is not None and device != expected:
        raise ValueError(f'--platform {selected} requires --device {expected}; remove the conflicting --device')
    if rt_device is not None and rt_device != expected:
        raise ValueError(f'--platform {selected} requires --rt-device {expected}; remove the conflicting --rt-device')
    args.device = args.rt_device = expected
    if selected == 'dgx':
        if algebra not in (None, 'torch'):
            raise ValueError('--platform dgx uses PyTorch CUDA algebra; remove --algebra-backend cupy')
        args.algebra_backend = 'torch'
        report['rt_backend'] = 'cuda_optix'
        if not args.dry_run:
            report['checks'] = check_dgx_cuda_optix()
            report['preflight'] = 'passed'
    else:
        args.algebra_backend = algebra or 'cupy'
        report['rt_backend'] = 'llvm_cpu'
        if not args.dry_run and args.algebra_backend == 'cupy':
            try:
                report['checks'] = check_agx_cuda()
                report['preflight'] = 'passed'
            except Exception as exc:
                if algebra == 'cupy':
                    raise RuntimeError('Explicit CuPy request failed its CUDA preflight; use --platform agx without --algebra-backend to allow CPU fallback') from exc
                args.algebra_backend = 'torch'
                report['preflight'] = 'cpu_fallback'
                report['fallback_reason'] = f'{type(exc).__name__}: {exc}'
                print(f'AGX GPU preflight failed; selecting CPU algebra before simulation: {report["fallback_reason"]}', file=sys.stderr)
    if args.dry_run:
        report['preflight'] = 'not_run_dry_run'
    report['algebra_backend'] = args.algebra_backend
    return report
