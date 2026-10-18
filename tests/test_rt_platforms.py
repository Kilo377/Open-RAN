"""Profiles select backends before execution and never hide DGX failures."""
import json
from types import SimpleNamespace

import pytest

from experiments.run_rt_sys import parse_args, run_case
from phy.rt_sys import platforms


def args(**overrides):
    values = dict(platform=None, device=None, rt_device=None, algebra_backend=None, dry_run=False)
    return SimpleNamespace(**(values | overrides))


@pytest.mark.parametrize('selected,device,algebra', [('dgx','cuda','torch'), ('agx','cpu','cupy')])
def test_profile_dry_run_selects_all_devices_without_hardware_checks(monkeypatch, capsys, selected, device, algebra):
    def forbidden():
        pytest.fail('Dry run must not load CUDA/OptiX or require their libraries')
    monkeypatch.setattr(platforms, 'check_dgx_cuda_optix', forbidden)
    monkeypatch.setattr(platforms, 'check_agx_cuda', forbidden)
    monkeypatch.setattr('sys.argv', ['run_rt_sys', '--platform', selected, '--dry-run'])
    run_case(parse_args())
    result = json.loads(capsys.readouterr().out)
    config = result['phy_config']['rt_sys']
    assert config['device'] == config['rt_device'] == device
    assert config['algebra_backend'] == algebra
    assert config['channel_array_backend'] == ('cupy' if selected == 'agx' else 'numpy')
    assert result['execution_devices']['linear_algebra'] == 'cuda'
    assert result['platform_profile']['preflight'] == 'not_run_dry_run'


def test_dgx_executes_its_check_without_agx_probe(monkeypatch):
    monkeypatch.setattr(platforms, 'check_dgx_cuda_optix', lambda: dict(optix_driver_loaded=True))
    monkeypatch.setattr(platforms, 'check_agx_cuda', lambda: pytest.fail('DGX must not use AGX fallback'))
    a = args(platform='dgx')
    report = platforms.resolve_platform(a)
    assert (a.device, a.rt_device, a.algebra_backend) == ('cuda','cuda','torch')
    assert report['checks']['optix_driver_loaded']
    assert report['fallback_reason'] is None


def unavailable():
    raise RuntimeError('kernel unavailable')


def test_dgx_failure_does_not_silently_select_cpu(monkeypatch):
    monkeypatch.setattr(platforms, 'check_dgx_cuda_optix', unavailable)
    with pytest.raises(RuntimeError, match='kernel unavailable'):
        platforms.resolve_platform(args(platform='dgx'))


def test_agx_prefers_real_cupy_kernels(monkeypatch):
    monkeypatch.setattr(platforms, 'check_agx_cuda', lambda: dict(actual_cuda_solve_and_svd=True))
    a = args(platform='agx')
    report = platforms.resolve_platform(a)
    assert (a.device, a.rt_device, a.algebra_backend) == ('cpu','cpu','cupy')
    assert report['checks']['actual_cuda_solve_and_svd']


def test_agx_auto_fallback_is_recorded_before_simulation(monkeypatch, capsys):
    monkeypatch.setattr(platforms, 'check_agx_cuda', unavailable)
    a = args(platform='agx')
    report = platforms.resolve_platform(a)
    assert a.device == a.rt_device == 'cpu' and a.algebra_backend == 'torch'
    assert report['preflight'] == 'cpu_fallback'
    assert 'kernel unavailable' in report['fallback_reason']
    assert 'selecting CPU algebra before simulation' in capsys.readouterr().err


def test_explicit_agx_cpu_does_not_need_cupy(monkeypatch):
    monkeypatch.setattr(platforms, 'check_agx_cuda', lambda: pytest.fail('CPU override must not require CuPy'))
    a = args(platform='agx', algebra_backend='torch')
    platforms.resolve_platform(a)
    assert a.algebra_backend == 'torch'


def test_explicit_cupy_request_is_strict(monkeypatch):
    monkeypatch.setattr(platforms, 'check_agx_cuda', unavailable)
    with pytest.raises(RuntimeError, match='Explicit CuPy request failed'):
        platforms.resolve_platform(args(platform='agx', algebra_backend='cupy'))


@pytest.mark.parametrize('values', [dict(platform='agx', device='cuda'),
    dict(platform='agx', rt_device='cuda'), dict(platform='dgx', device='cpu'),
    dict(platform='dgx', rt_device='cpu'), dict(platform='dgx', algebra_backend='cupy')])
def test_conflicting_flags_are_rejected(values):
    with pytest.raises(ValueError, match='--platform'):
        platforms.resolve_platform(args(**values, dry_run=True))


@pytest.mark.parametrize('device,rt_device,expected', [(None,None,('cuda','cuda')),
    ('cpu',None,('cpu','cpu')), ('cuda','cpu',('cuda','cpu'))])
def test_legacy_calls_preserve_device_selection(device, rt_device, expected):
    a = args(device=device, rt_device=rt_device)
    report = platforms.resolve_platform(a)
    assert (a.device,a.rt_device) == expected
    assert a.algebra_backend is None and report['requested'] == 'custom'
