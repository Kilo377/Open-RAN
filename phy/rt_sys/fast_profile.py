"""Resolve fast-model computation without changing MAC or control clocks."""


def configure_fast_profile(config, requested=None):
    """Mutate a run config with explicit, reproducible execution settings."""
    if config.get('feedback_mode') != 'fast':
        if requested is not None:
            raise ValueError('--fast-profile requires --model fast')
        for key in ('fast_profile', 'spatial_samples_per_prb', 'channel_time_sampling',
                    'fast_feedback_execution', 'algebra_device'):
            config.pop(key, None)
        return None
    profile = requested or config.get('fast_profile') or 'efficient'
    if profile not in ('efficient', 'reference'):
        raise ValueError('Unknown fast profile')
    efficient = profile == 'efficient'
    config.update(fast_profile=profile,
                  spatial_samples_per_prb=4 if efficient else None,
                  channel_time_sampling='one_per_slot_per_stream_pattern' if efficient else 'per_symbol',
                  fast_feedback_execution='cpu_batched' if efficient else 'scalar',
                  algebra_device='cpu' if efficient else (
                      'cupy' if config.get('algebra_backend') == 'cupy' else config.get('device', 'cuda')))
    return profile
