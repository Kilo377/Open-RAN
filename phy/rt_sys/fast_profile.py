"""Resolve fast-model computation without changing MAC or control clocks."""


def configure_fast_profile(config, requested=None):
    """Mutate a run config with explicit, reproducible execution settings."""
    system_keys=('system_phy_enabled','system_gain_update_ms','reference_gain_update_ms','fast_reports','compact_snapshots')
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
