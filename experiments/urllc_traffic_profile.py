"""Explicit arrival-only overlays; unchanged radio, UE population and deadlines."""
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from traffic.arrival_model import validate_arrival_phases

FIELDS = {'arrival_model', 'arrival_rate_bps', 'packet_size_bytes',
          'on_duration_slots', 'off_duration_slots', 'burst_size_bytes', 'start_offset_slots',
          'arrival_phases'}


def apply_profile(manifest, configs, path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    profile = json.loads(raw)
    if (not isinstance(profile, dict) or set(profile) - {'profile_id', 'provenance', 'ues', 'slot_duration_ms'}
            or not isinstance(profile.get('profile_id'), str) or not profile['profile_id']
            or not isinstance(profile.get('ues'), dict) or not profile['ues']):
        raise ValueError('Arrival profile requires profile_id and a nonempty ues object')
    if 'slot_duration_ms' in profile:
        expected = profile['slot_duration_ms']
        actual = manifest.get('slot_duration_ms', configs.get('sim', {}).get('slot_duration_ms'))
        if (type(expected) not in (int, float) or not math.isfinite(expected) or expected <= 0
                or type(actual) not in (int, float) or not math.isfinite(actual)
                or not math.isclose(expected, actual, rel_tol=0, abs_tol=1e-12)):
            raise ValueError('Arrival profile slot duration does not match this simulation')
    users = {u['ue_id']: u for u in configs['ue']['ues']}
    updates = {}
    for uid, override in profile['ues'].items():
        if uid not in users or not isinstance(override, dict) or not override or set(override) - FIELDS:
            raise ValueError('Arrival profile contains unknown UE or non-arrival fields')
        for key, value in override.items():
            if key == 'arrival_model':
                if value not in ('cbr', 'poisson', 'on_off', 'phased'):
                    raise ValueError('Arrival profile requires cbr, poisson, on_off or phased')
            elif key == 'arrival_phases':
                validate_arrival_phases(value)
            elif key == 'arrival_rate_bps':
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError('Invalid arrival rate')
            elif type(value) is not int or value < (1 if key in ('packet_size_bytes', 'on_duration_slots') else 0):
                raise ValueError('Invalid integer arrival parameter')
        if len(users[uid].get('logical_channels', [])) > 1:
            raise ValueError('Arrival profile currently requires one logical channel per selected UE')
        model = override.get('arrival_model', users[uid].get('arrival_model'))
        phases = override.get('arrival_phases', users[uid].get('arrival_phases'))
        if model == 'phased':
            validate_arrival_phases(phases)
        elif phases is not None:
            raise ValueError('arrival_phases requires arrival_model=phased')
        updates[uid] = deepcopy(override)
    # Validate everything before mutating the generated case.
    for uid, override in updates.items():
        users[uid].update(deepcopy(override))
        # An aggregate-UE overlay must also update explicit logical channels,
        # otherwise core traffic would continue consuming the old channel input.
        channels = users[uid].get('logical_channels', [])
        if channels:
            channels[0].update(deepcopy(override))
    manifest['traffic_profile'] = dict(profile_id=profile['profile_id'], path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(), provenance=profile.get('provenance'),
        overrides=deepcopy(updates), application='after_global_traffic_load; absolute_arrival_parameters_only')
    if 'slot_duration_ms' in profile:
        manifest['traffic_profile']['slot_duration_ms'] = profile['slot_duration_ms']
    if any(u.get('arrival_model') == 'phased' for u in updates.values()):
        manifest['traffic_profile']['phase_clock'] = 'repeat; global slot 1 = phase-local slot 0'
    return manifest['traffic_profile']
