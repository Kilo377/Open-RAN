"""Named arrival scenarios for the six-UE URLLC experiment.

The general simulator keeps its native arrivals. Only the dedicated URLLC
experiment entry point defaults to burst pressure. A scenario never depends
on which controller is enabled and is not an RF input.
"""
import math
from pathlib import Path

from experiments.urllc_traffic_profile import apply_profile

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ('native', 'burst', 'alternating')
PROFILE = ROOT/'configs/rt_sys/urllc_equal_mean_burst_2x.json'
ALTERNATING_PROFILE = ROOT/'configs/rt_sys/urllc_alternating_2x.json'
PROFILES = {'native': None, 'burst': PROFILE, 'alternating': ALTERNATING_PROFILE}
ALTERNATING_CYCLE_S = 1.6


def add_scenario_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--traffic-profile', type=Path,
        help='Explicit arrival-only overlay; deadlines and radio parameters are unchanged')
    group.add_argument('--urllc-scenario', choices=SCENARIOS,
        help='Six-UE URLLC preset at traffic-load=2: native, burst pressure, or 0.8 s native / 0.8 s burst repeating')


def profile_for(scenario):
    if scenario not in PROFILES:
        raise ValueError('Unknown URLLC scenario: '+str(scenario))
    return PROFILES[scenario]


def apply_named_scenario(manifest, configs, scenario):
    path = profile_for(scenario)
    if path is not None:
        # The files declare absolute rates for the actual integer native CBR
        # means at load=2, not a rate multiplier. Refuse a different baseline
        # instead of silently changing the mean load under an "equal mean" name.
        dt = manifest['slot_duration_ms']
        users = {u['ue_id']: u for u in configs['ue']['ues']}
        if not math.isclose(dt, .5, abs_tol=1e-12):
            raise ValueError('Named URLLC pressure scenarios require 0.5 ms slots')
        for uid, expected in (('ue_2', 175), ('ue_6', 137)):
            user = users.get(uid, {})
            channels = user.get('logical_channels', []) or []
            arrival = dict(user, **channels[0]) if len(channels) == 1 else user
            if (len(channels) > 1 or arrival.get('arrival_model') != 'cbr'
                    or int(float(arrival.get('arrival_rate_bps', 0))*dt/8000) != expected):
                raise ValueError('Named URLLC pressure requires native UE2/UE6 CBR at traffic-load=2')
        apply_profile(manifest, configs, path)
    manifest['urllc_scenario'] = dict(name=scenario,
        phase_duration_ms=800 if scenario == 'alternating' else None,
        phase_order=['native', 'burst'] if scenario == 'alternating' else [scenario],
        scope='Arrival timing only; independent of controller enablement; no state reset')


def configure_arrivals(manifest, configs, args):
    scenario = getattr(args, 'urllc_scenario', None)
    profile = getattr(args, 'traffic_profile', None)
    if scenario is not None and profile is not None:
        raise ValueError('Choose a named URLLC scenario or an explicit traffic profile')
    if scenario is not None:
        apply_named_scenario(manifest, configs, scenario)
    elif profile is not None:
        apply_profile(manifest, configs, profile)
