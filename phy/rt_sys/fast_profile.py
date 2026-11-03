"""Resolve fast-model computation without changing MAC or control clocks."""


def configure_fast_profile(config, requested=None):
    """Mutate a run config with explicit, reproducible execution settings."""
    system_keys=('system_phy_enabled','system_gain_update_ms','reference_gain_update_ms','fast_reports','compact_snapshots','fast_channel_mode')
    if config.get('feedback_mode') != 'fast':
        if requested is not None:
            raise ValueError('--fast-profile requires --model fast')
        for key in ('fast_profile', 'spatial_samples_per_prb', 'channel_time_sampling',
                    'fast_feedback_execution', 'algebra_device',*system_keys):
            config.pop(key, None)
        return None
    profile = requested or config.get('fast_profile') or 'system'
    if profile not in ('efficient', 'reference', 'system'):
        raise ValueError('Unknown fast profile')
    efficient = profile != 'reference'
    if profile=='system':
        config.setdefault('system_phy_enabled',True)
        config.setdefault('system_gain_update_ms',2.)
        config.setdefault('reference_gain_update_ms',2.)
        config.setdefault('fast_reports',True)
        config.setdefault('compact_snapshots',True)
        config.setdefault('fast_channel_mode','sampled')
    else:
        for key in system_keys:config.pop(key,None)
    config.update(fast_profile=profile,
                  spatial_samples_per_prb=4 if efficient else None,
                  channel_time_sampling='one_per_slot_per_stream_pattern' if efficient else 'per_symbol',
                  fast_feedback_execution='cpu_batched' if efficient else 'scalar',
                  algebra_device='cpu' if efficient else (
                      'cupy' if config.get('algebra_backend') == 'cupy' else config.get('device', 'cuda')))
    if config.get('system_phy_enabled'):config['channel_time_sampling']='one_per_gain_update_per_stream_pattern'
    return profile


def configure_fast_channel(config, manifest=None, *, path_update_ms=None,
                           displacement_m=None, rt_device=None):
    """Resolve system-channel defaults after CLI overrides, isolating full.

    Explicit CLI geometry values take priority over the fast defaults.
    """
    if config.get('fast_profile')!='system':return
    if config.get('fast_channel_mode')=='sampled':
        config.update(channel_array_backend='numpy',retain_channel_on_gpu=False)
    if manifest is None:return
    if path_update_ms is None:
        manifest['path_update_ms']=config.get('system_path_update_ms',10.)
    if displacement_m is None:
        manifest['max_geometry_displacement_m']=config.get('system_max_geometry_displacement_m',.015)
    config.setdefault('rt_prefetch_workers',2)
    if rt_device=='cpu':config.setdefault('rt_prefetch',True)
    elif rt_device=='cuda':config['rt_prefetch']=False
    from phy.rt_sys.channel import geometry_blocks
    geometry_blocks(manifest)
