"""Build versioned pressure-discovery plans without executing a simulator.

Input schema ``oran.environment-refinement-design.v1`` has study_id, stage,
recipes[{id, embb_scale, mec_result_bytes}], assignments[{worker_id, seed,
recipe_ids}], arms (OFF/A/B/C/D/E/F), allow_scout_on, include_native_reference,
deadline_utc, runner_sha256 and parent{traffic_profile, traffic_profile_sha256,
config_snapshot, config_snapshot_sha256}. Validation additionally declares all
discovery_seeds and exactly one frozen recipe, two fresh seeds and all six arms.
Recipes may select urllc_pattern=burst4 (default), mixed4, mixed5 or rare6. Mixed
patterns change timing only and preserve registered-window byte/packet means.
Optional embb_scale_by_ue gives all three eMBB multipliers relative to C3;
it cannot be combined with a global embb_scale other than1.
Optional initial_ue4_offload_fraction=.8 or1(default) changes only the initial
task-routing policy; the frozen controllers still choose their own actions.
Pilot AFTER covers2000(default) or4000 slots; validation covers4000(default) or20000.

Only build_bundle()/write_bundle() or the CLI writes protocol files. Importing
this module, validating designs and running tests never launches remote work.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import environment_validation_worker_v2 as worker
from experiments import environment_validation_worker_v3 as worker3
from experiments import environment_validation_worker_v4 as worker4
from experiments.environment_validation_plan import APPS, AP_PROFILE, NATIVE_SERVICE, SERVICE, frozen_inputs
from experiments import environment_urllc_patterns as urllc_patterns

SCHEMA = 'oran.environment-refinement-design.v1'
KNOWN_USED_SEEDS = {55, 137, 138, *range(179, 185), 20261108, *range(20261116, 20261122),
                    20261201, 20261202, 20261203, 20261204, 20261205}
EMBB = ('ue_0', 'ue_4', 'ue_8')
URLLC = ('ue_2', 'ue_6')
USERS = {'ue_0', 'ue_1', 'ue_2', 'ue_4', 'ue_6', 'ue_8'}
FIELDS = ('arrival_model', 'arrival_rate_bps', 'packet_size_bytes', 'burst_size_bytes',
          'on_duration_slots', 'off_duration_slots', 'start_offset_slots')
DEFAULT_TRAFFIC = dict(arrival_model='none', arrival_rate_bps=0., packet_size_bytes=200,
                       burst_size_bytes=0, on_duration_slots=50, off_duration_slots=20,
                       start_offset_slots=0)


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()


def digest(value):
    return worker.digest(value)


def seal(value):
    value = deepcopy(value)
    value.pop('plan_sha256', None)
    value['plan_sha256'] = digest(value)
    return value


def _identity(value, name, limit):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,'+str(limit-1)+'}', value):
        raise ValueError('Invalid '+name)
    return value


def selected_runner(design):
    version = design.get('runner_version', 2)
    if type(version) is not int or version not in (2, 3, 4):
        raise ValueError('runner_version must be 2, 3 or 4')
    module = {2: worker, 3: worker3, 4: worker4}[version]
    parallel = design.get('max_parallel', 1 if version == 2 else 2)
    if type(parallel) is not int or parallel not in ((1,) if version == 2 else (1, 2, 3)):
        raise ValueError('v2 is serial; v3/v4 support two or three parallel cases')
    return version, module, parallel


def embb_scales(recipe):
    """An explicit per-UE map is an alternative to uniform scaling."""
    global_scale = recipe.get('embb_scale', 1.)
    if type(global_scale) not in (int, float) or not math.isfinite(global_scale):
        raise ValueError('Ordinary scale must be finite numeric data')
    mapping = recipe.get('embb_scale_by_ue')
    if mapping is None:
        if 'embb_scale_by_ue' in recipe or 'embb_scale' not in recipe:
            raise ValueError('Provide a global ordinary scale or all three per-UE scales')
        if global_scale not in (1., 1.05, 1.1):
            raise ValueError('Ordinary scale must be 1, 1.05 or 1.10')
        return dict.fromkeys(EMBB, global_scale)
    if global_scale != 1.:
        raise ValueError('Per-UE ordinary scales cannot combine with global embb_scale != 1')
    if not isinstance(mapping, dict) or set(mapping) != set(EMBB):
        raise ValueError('Per-UE ordinary scales require exactly ue_0, ue_4 and ue_8')
    if any(type(v) not in (int, float) or not math.isfinite(v) or not .5 <= v <= 2.0 for v in mapping.values()):
        raise ValueError('Every per-UE ordinary multiplier must be finite and within .5..2.0')
    return {uid:mapping[uid] for uid in EMBB}


def initial_ue4_offload_fraction(recipe):
    """A recorded initial routing policy, never an xApp action override."""
    value = recipe.get('initial_ue4_offload_fraction', 1.)
    if type(value) not in (int, float) or not math.isfinite(value) or value not in (.8, 1.):
        raise ValueError('Initial UE4 offload fraction must be .8 or 1')
    return value


def initial_mec_demand(services):
    return {uid:1000/p['interval_ms']*p['compute_gflop']*p['offload_fraction']
            for uid,p in services['mec']['tasks_by_ue'].items()}


def validate_design(design):
    required = {'schema', 'study_id', 'stage', 'recipes', 'assignments', 'arms',
                'allow_scout_on', 'include_native_reference', 'parent', 'deadline_utc', 'runner_sha256'}
    allowed = required | {'discovery_seeds', 'observe_slots', 'purpose', 'runner_version', 'max_parallel', 'ue0_route_override'}
    if not isinstance(design, dict) or not required <= design.keys() or set(design)-allowed or design['schema'] != SCHEMA:
        raise ValueError('Explicit refinement design schema/fields required')
    _identity(design['study_id'], 'study identity', 60)
    stage = design['stage']
    if stage not in ('pilot', 'validation'):
        raise ValueError('Unknown refinement stage')
    for key in ('allow_scout_on', 'include_native_reference'):
        if type(design[key]) is not bool:
            raise ValueError('Explicit boolean '+key+' required')
    deadline = datetime.fromisoformat(design['deadline_utc'])
    if deadline.tzinfo is None or deadline.utcoffset().total_seconds() != 0 or deadline <= datetime.now(timezone.utc):
        raise ValueError('An explicit future UTC failure-stop deadline is required')
    version, runner, _ = selected_runner(design)
    if version == 4:
        if 'ue0_route_override' in design and digest(design['ue0_route_override']) != digest(worker4.ROUTE_OVERRIDE['ue_0']):
            raise ValueError('v4 requires exactly UE0 route0 start_fraction .65')
        if design['include_native_reference']:
            raise ValueError('v4 geometry must not be labelled a current native reference')
    elif 'ue0_route_override' in design:
        raise ValueError('An explicit route override requires the v4 geometry runner')
    worker._pin(design['runner_sha256'])
    if design['runner_sha256'] != worker.sha(runner.__file__):
        raise ValueError('Pin the exact reviewed selected worker module')
    recipes = design['recipes']
    if not isinstance(recipes, list) or not recipes:
        raise ValueError('Provide explicit recipes')
    ids = []
    for recipe in recipes:
        required_recipe = {'id', 'mec_result_bytes'}
        if (not isinstance(recipe, dict) or not required_recipe <= recipe.keys()
                or set(recipe)-required_recipe-{'embb_scale','embb_scale_by_ue','urllc_pattern','initial_ue4_offload_fraction'}):
            raise ValueError('Only ordinary load, MEC result payload, named URLLC timing and explicit initial UE4 routing may vary')
        ident = _identity(recipe['id'], 'recipe identity', 40)
        if ident == 'native_reference' or ident in ids:
            raise ValueError('Unique nonreserved recipe IDs required')
        ids.append(ident)
        embb_scales(recipe)
        initial_ue4_offload_fraction(recipe)
        if type(recipe['mec_result_bytes']) is not int or recipe['mec_result_bytes'] not in (200, 400, 800):
            raise ValueError('MEC result bytes must be 200, 400 or 800')
        if recipe.get('urllc_pattern', 'burst4') not in urllc_patterns.PATTERNS:
            raise ValueError('Unknown URLLC timing pattern')
    arms = design['arms']
    if (not isinstance(arms, list) or not arms or arms[0] != 'OFF' or len(arms) != len(set(arms))
            or any(a not in {'OFF', *APPS} for a in arms)):
        raise ValueError('OFF must precede explicit registered singleton arms')
    if stage == 'pilot' and len(arms) > 1 and not design['allow_scout_on']:
        raise ValueError('Scout ON requires explicit allow_scout_on for environment discovery')
    if stage == 'validation' and (len(recipes) != 1 or arms != ['OFF', *APPS]):
        raise ValueError('Final validation requires one common recipe and OFF plus all six singleton arms')
    observe = design.get('observe_slots', 2000 if stage == 'pilot' else 4000)
    if type(observe) is not int or observe not in ((2000, 4000) if stage == 'pilot' else (4000, 20000)):
        raise ValueError('Pilot AFTER must be2000 or4000 slots; validation AFTER must be4000 or20000')
    assignments = design['assignments']
    if not isinstance(assignments, list) or not assignments:
        raise ValueError('Explicit host/seed assignments required')
    seen = set()
    for assignment in assignments:
        if not isinstance(assignment, dict) or set(assignment) != {'worker_id', 'seed', 'recipe_ids'}:
            raise ValueError('Assignment schema differs')
        wid, seed, selected = assignment['worker_id'], assignment['seed'], assignment['recipe_ids']
        if wid not in ('agx1', 'agx2') or type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('Known host and valid explicit seed required')
        if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)) or not set(selected) <= set(ids):
            raise ValueError('Assignments must name explicit recipes once')
        for recipe_id in selected:
            key = (wid, seed, recipe_id)
            if key in seen:
                raise ValueError('Repeated host/seed/recipe origin')
            seen.add(key)
    if {a['worker_id'] for a in assignments} != {'agx1', 'agx2'}:
        raise ValueError('Provide a nonempty plan for each of the two hosts')
    if {rid for a in assignments for rid in a['recipe_ids']} != set(ids):
        raise ValueError('Every declared recipe must have an assigned OFF origin')
    used = design.get('discovery_seeds', [])
    if not isinstance(used, list) or any(type(s) is not int or not 0 <= s < 2**32 for s in used):
        raise ValueError('Discovery seeds must be explicit valid integers')
    if stage == 'validation':
        if 'discovery_seeds' not in design or len(assignments) != 2 or len({a['seed'] for a in assignments}) != 2:
            raise ValueError('Two distinct fresh validation seeds and declared discovery history required')
        if any(a['seed'] in KNOWN_USED_SEEDS | set(used) for a in assignments):
            raise ValueError('Validation seed was already used for training or environment discovery')
    parent = design['parent']
    if not isinstance(parent, dict) or set(parent) != {'traffic_profile', 'traffic_profile_sha256', 'config_snapshot', 'config_snapshot_sha256'}:
        raise ValueError('Pin the actual C3 profile and OFF config snapshot')
    return design


def effective_traffic(user):
    channels = user.get('logical_channels', [])
    if len(channels) > 1:
        raise ValueError('The frozen six-UE recipes require one logical channel each')
    merged = dict(user, **channels[0]) if channels else user
    return {key:merged.get(key, value) for key, value in DEFAULT_TRAFFIC.items()}


def load_parent(design, root=ROOT):
    parent = design['parent']
    loaded = {}
    for name in ('traffic_profile', 'config_snapshot'):
        path = Path(parent[name]).resolve()
        worker._pin(parent[name+'_sha256'])
        if worker.sha(path) != parent[name+'_sha256']:
            raise ValueError('Parent input checksum differs: '+name)
        loaded[name] = json.loads(path.read_text())
    profile, snapshot = loaded['traffic_profile'], loaded['config_snapshot']
    from experiments.environment_validation_plan import CANDIDATES, profile as original_profile
    expected = original_profile(CANDIDATES[2])
    if profile != expected:
        raise ValueError('Start from the exact unchanged C3 arrival profile')
    if snapshot.get('source_sha256') != worker.SOURCE_SHA256:
        raise ValueError('Parent replay must use the frozen cbe simulator')
    manifest = snapshot['manifest']
    if (manifest.get('scene') != 'munich' or len(manifest.get('ues', [])) != 6
            or len(manifest.get('cells', [])) != 3 or manifest.get('slot_duration_ms') != .5
            or manifest.get('carrier_frequency_hz') != 3_500_000_000
            or manifest.get('bandwidth_hz') != 10_000_000
            or manifest.get('traffic_profile', {}).get('sha256') != parent['traffic_profile_sha256']):
        raise ValueError('Parent physical/arrival recipe is not the C3 six-UE deployment')
    users = {u['ue_id']:u for u in snapshot['configs']['ue']['ues']}
    if set(users) != USERS:
        raise ValueError('Preserve the released six-UE population')
    traffic = {uid:effective_traffic(u) for uid, u in users.items()}
    for uid in URLLC:
        if any(traffic[uid][key] != value for key, value in profile['ues'][uid].items()):
            raise ValueError('Parent URLLC overlay was not actually used')
    for uid, expected_rate in zip(EMBB, (21_000_000, 18_000_000, 15_000_000)):
        p = traffic[uid]
        if (users[uid].get('slice_id') != 'embb' or p['arrival_model'] != 'on_off'
                or p['arrival_rate_bps'] != expected_rate or p['burst_size_bytes'] != 3300
                or p['on_duration_slots'] != 40 or p['off_duration_slots'] != 16 or p['start_offset_slots'] != 0):
            raise ValueError('Parent ordinary business is not the unchanged global1.5 recipe')
    inputs = frozen_inputs(root)
    if snapshot.get('ue_services_profile', {}).get('sha256') != inputs['service_sha256']:
        raise ValueError('Parent must explicitly use unchanged offload-first services')
    services = json.loads((Path(root)/SERVICE).read_text())
    recorded_services = snapshot['configs']['phy']['rt_sys']['ue_services']
    if any(recorded_services.get(key) != value for key, value in services.items()):
        raise ValueError('Recorded parent services differ from the pinned original task/radio policy')
    return loaded, traffic, inputs


def raw_diff(before, after, prefix=''):
    result = []
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        path = prefix+'/'+str(key)
        if isinstance(old, dict) and isinstance(new, dict):
            result += raw_diff(old, new, path)
        elif old != new:
            result.append(dict(path=path, before=old, after=new))
    return result


def make_recipe(study_id, spec, parent_profile, parent_traffic, service):
    profile = deepcopy(parent_profile)
    pattern = spec.get('urllc_pattern', 'burst4')
    scales = embb_scales(spec)
    initial_fraction = initial_ue4_offload_fraction(spec)
    profile.update(profile_id=study_id+'_'+spec['id'], provenance='Explicit environment discovery. '
        'Only eMBB arrival-rate and frame-start burst scale; global load remains1.5, both URLLC streams '
        'retain the exact C3 profile and integer packetization. No trained action is altered.')
    profile['ues'] = urllc_patterns.arrivals(parent_profile, pattern)
    if 'embb_scale_by_ue' in spec:
        profile['provenance'] = ('Explicit arrival-only environment discovery. Global load remains1.5; '
            'eMBB mean/burst multipliers='+json.dumps(scales, sort_keys=True)+'; URLLC pattern='+pattern+'. '
            +urllc_patterns.metadata(pattern)['description']+' No deadlines, physical resources or trained actions change.')
    elif pattern != 'burst4':
        # Keep legacy recipe bytes/case identities reproducible.
        profile['provenance'] = ('Explicit arrival-only environment discovery. Global load remains1.5; '
            'eMBB mean/burst multiplier='+str(spec['embb_scale'])+'; URLLC pattern='+pattern+'. '
            +urllc_patterns.metadata(pattern)['description']+' No deadlines, physical resources or trained actions change.')
    if initial_fraction != 1:
        profile['provenance'] = profile['provenance'].replace('Explicit arrival-only environment discovery.',
            'Explicit arrival and INITIAL TASK ROUTING environment discovery.').replace(
            'Only eMBB arrival-rate and frame-start burst scale;', 'eMBB arrival-rate and burst scale plus INITIAL TASK ROUTING;')
        profile['provenance'] += (' INITIAL TASK ROUTING: only UE4 initial offload fraction='+str(initial_fraction)
            +'; UE0 routing, task work/timing, radio and frozen autonomous action logic remain unchanged.')
    for uid in EMBB:
        fields = deepcopy(parent_traffic[uid])
        fields['arrival_rate_bps'] *= scales[uid]
        fields['burst_size_bytes'] = round(fields['burst_size_bytes'] * scales[uid])
        profile['ues'][uid] = fields
    services = deepcopy(service)
    for uid in ('ue_0', 'ue_4'):
        services['mec']['tasks_by_ue'][uid]['result_bytes'] = spec['mec_result_bytes']
    if initial_fraction != services['mec']['tasks_by_ue']['ue_4']['offload_fraction']:
        services['mec']['tasks_by_ue']['ue_4']['offload_fraction'] = initial_fraction
    expected = deepcopy(services)
    for uid in ('ue_0', 'ue_4'):
        expected['mec']['tasks_by_ue'][uid]['result_bytes'] = 200
    expected['mec']['tasks_by_ue']['ue_4']['offload_fraction'] = service['mec']['tasks_by_ue']['ue_4']['offload_fraction']
    if expected != service:
        raise ValueError('Only shared MEC result payload and explicit initial UE4 task routing may change')
    for uid in URLLC:
        if pattern == 'burst4' and profile['ues'][uid] != parent_profile['ues'][uid]:
            raise ValueError('URLLC timing, mean and packets must remain C3')
    effective = deepcopy(parent_traffic)
    for uid, fields in profile['ues'].items():
        effective[uid].update(deepcopy(fields))
    return profile, services, effective


def sampled_load(traffic, *, baseline_slot=2040, origin_observe_slots=2000, observe_slots=2000):
    """Actual production arrivals/packetization; immediate drain is not a PHY."""
    from traffic.arrival_model import TrafficProfile, build_arrival_model
    from traffic.packet_ledger import PacketLedger
    total = baseline_slot + observe_slots + 400
    results = {}
    for uid, fields in traffic.items():
        cycle = (sum(p['duration_slots'] for p in fields['arrival_phases']) if fields['arrival_model'] == 'phased'
                 else fields['on_duration_slots'] + fields['off_duration_slots'] if fields['arrival_model'] == 'on_off' else 1)
        windows = dict(cycle=(1, cycle), before=(baseline_slot-origin_observe_slots+1, baseline_slot),
                       after=(baseline_slot+1, baseline_slot+observe_slots),
                       pilot_after=(2041,4040), validation_after=(2041,6040), whole=(1, total))
        model = build_arrival_model(TrafficProfile(ue_id=uid, **fields))
        rows = {}
        for name, (left, right) in windows.items():
            lc = SimpleNamespace(logical_channel_id=0, queue_bytes=0, max_queue_bytes=None, packet_delay_budget_ms=10)
            ledger = PacketLedger(SimpleNamespace(ue_id=uid, logical_channels=[lc]), .5)
            rng = random.Random(0)
            packet_sizes = set()
            for slot in range(left, right+1):
                amount = model.sample_bytes(slot_id=slot, slot_duration_s=.0005, rng=rng)
                ledger.add(lc, amount, slot, packet_size=fields['packet_size_bytes'])
                packet_sizes.update(e['bytes'] for e in ledger.events if e['kind'] == 'arrival')
                ledger.events.clear()
                if amount:
                    ledger.deliver(lc, amount, slot)
            duration = (right-left+1)*.0005
            rows[name] = dict(first_slot=left, last_slot=right, sampled_slots=right-left+1,
                bytes=ledger.stats['offered_bytes'], packets=ledger.stats['admitted_packets'],
                packet_sizes_bytes=sorted(packet_sizes),
                mean_mbps=ledger.stats['offered_bytes']*8/duration/1e6)
        results[uid] = rows
    return results


def _case(design, assignment, recipe, arm, pins):
    native = recipe is None
    scene = 'native_reference' if native else recipe['id']
    key = scene+'_s'+str(assignment['seed'])
    name = key+'_'+arm
    baseline, observe = 2040, design.get('observe_slots', 2000 if design['stage'] == 'pilot' else 4000)
    recipe_pin = pins.get(scene, {})
    result = dict(id=digest(dict(study=design['study_id'], stage=design['stage'], worker=assignment['worker_id'],
            name=name, recipe_sha256=recipe_pin.get('recipe_sha256')))[:32],
        name=name, origin_key=key, scene=scene, seed=assignment['seed'], arm=arm,
        app_ids=[] if arm == 'OFF' else [APPS[arm]], worker_id=assignment['worker_id'],
        recipe_kind='current_native_reference' if native else 'refinement_candidate',
        slots=baseline+observe+400, baseline_slot=baseline, origin_observe_slots=2000, observe_slots=observe,
        traffic_load=1. if native else 1.5,
        traffic_profile=None if native else recipe_pin['traffic_profile'],
        traffic_profile_sha256=None if native else recipe_pin['traffic_profile_sha256'],
        ue_services_config=str(worker.SCOPE/'source'/NATIVE_SERVICE) if native else recipe_pin['ue_services_config'],
        ue_services_config_sha256=pins['native_service_sha256'] if native else recipe_pin['ue_services_config_sha256'],
        ap_services_profile=str(worker.SCOPE/'source'/AP_PROFILE), ap_services_profile_sha256=pins['ap_profile_sha256'])
    if design.get('runner_version', 2) == 4:
        result.update(base_config=recipe_pin['base_config'], base_config_sha256=recipe_pin['base_config_sha256'])
    return result


def build_bundle(design, *, root=ROOT):
    """Return relative-file -> JSON objects; do not write or run anything."""
    validate_design(design)
    runner_version, runner, max_parallel = selected_runner(design)
    root = Path(root).resolve()
    parent, traffic, inputs = load_parent(design, root)
    service = json.loads((root/SERVICE).read_text())
    observe = design.get('observe_slots', 2000 if design['stage'] == 'pilot' else 4000)
    reference_load = sampled_load(traffic, observe_slots=observe)
    files, pins, differences, checks = {}, {}, {}, {}
    base_pin = None
    if runner_version == 4:
        frozen_base = root/worker4.BASE_CONFIG
        if worker.sha(frozen_base) != inputs['frozen_physical_files'][worker4.BASE_CONFIG]:
            raise ValueError('The immutable base configuration pin differs')
        base = json.loads(frozen_base.read_text())
        if 'ue_route_overrides' in base:
            raise ValueError('The immutable base already contains route overrides')
        changed_base = {**base, 'ue_route_overrides': deepcopy(worker4.ROUTE_OVERRIDE)}
        base_file = 'profiles/'+design['study_id']+'_ue0_road0_phase065_base.json'
        files[base_file] = changed_base
        base_pin = dict(base_config=str(worker.SCOPE/'protocol'/base_file),
            base_config_sha256=hashlib.sha256(encoded(changed_base)).hexdigest())
    for recipe in design['recipes']:
        profile, services, effective = make_recipe(design['study_id'], recipe, parent['traffic_profile'], traffic, service)
        stem = design['study_id']+'_'+recipe['id']
        traffic_file, service_file = 'profiles/'+stem+'_traffic.json', 'profiles/'+stem+'_service.json'
        files[traffic_file], files[service_file] = profile, services
        pin = dict(traffic_load=1.5,
            traffic_profile=str(worker.SCOPE/'protocol'/traffic_file),
            traffic_profile_sha256=hashlib.sha256(encoded(profile)).hexdigest(),
            ue_services_config=str(worker.SCOPE/'protocol'/service_file),
            ue_services_config_sha256=hashlib.sha256(encoded(services)).hexdigest())
        if base_pin is not None:
            pin.update(base_pin)
        pin['recipe_sha256'] = digest(dict(spec=recipe, **pin,
            physical_files=inputs['frozen_physical_files'], models=inputs['frozen_models']))
        pins[recipe['id']] = pin
        pattern = recipe.get('urllc_pattern', 'burst4')
        timing = urllc_patterns.metadata(pattern)
        if observe == 20000:
            timing['requested_after_window'] = dict(first_slot=2041,last_slot=2040+observe,
                sampled_slots=observe,duration_s=observe*.0005)
        scales = embb_scales(recipe)
        differences[recipe['id']] = dict(spec=recipe, traffic=raw_diff(traffic, effective),
            services=raw_diff(service, services), reference_global_load=1.5,
            URLLC_unchanged=pattern == 'burst4', URLLC_timing=timing,
            URLLC_registered_window_bytes_packets_sizes_preserved=True,
            radio_compute_and_F_validated_task_shape_unchanged=True,
            physical_topology_unchanged=True,
            mec_result_payload_changed=recipe['mec_result_bytes'] != 200)
        actual = sampled_load(effective, observe_slots=observe)
        if actual['ue_1'] != reference_load['ue_1']:
            raise ValueError('Unrelated voice trace changed')
        for uid in URLLC:
            if pattern == 'burst4' and actual[uid] != reference_load[uid]:
                raise ValueError('The unchanged burst4 trace changed')
            mean = urllc_patterns.MEAN_PACKET_BYTES[uid]
            for name in ('before', 'pilot_after', 'validation_after', 'after'):
                row, old = actual[uid][name], reference_load[uid][name]
                if (row['bytes'] != old['bytes'] or row['packets'] != old['packets']
                        or row['packet_sizes_bytes'] != [mean]
                        or row['bytes'] != row['sampled_slots']*mean or row['packets'] != row['sampled_slots']):
                    raise ValueError('Registered URLLC window bytes/packets/packet sizes changed')
        for uid in EMBB:
            for name, row in actual[uid].items():
                before = reference_load[uid][name]
                row['parent_bytes'] = before['bytes']
                row['parent_packets'] = before['packets']
                row['actual_byte_change_percent'] = 100*(row['bytes']/before['bytes']-1) if before['bytes'] else None
                fields = effective[uid]
                period = fields['on_duration_slots']+fields['off_duration_slots']
                on_count = sum((slot+fields['start_offset_slots']) % period < fields['on_duration_slots']
                               for slot in range(row['first_slot'],row['last_slot']+1))
                burst_count = sum((slot+fields['start_offset_slots']) % period == 0
                                  for slot in range(row['first_slot'],row['last_slot']+1))
                chunk = int(fields['arrival_rate_bps']*.0005/8)
                cap = fields['packet_size_bytes']
                expected_bytes = on_count*chunk+burst_count*fields['burst_size_bytes']
                expected_packets = ((on_count-burst_count)*math.ceil(chunk/cap)
                    +burst_count*math.ceil((chunk+fields['burst_size_bytes'])/cap))
                row.update(nominal_multiplier=scales[uid],expected_bytes=expected_bytes,expected_packets=expected_packets)
                if row['bytes'] != expected_bytes or row['packets'] != expected_packets:
                    raise ValueError('Actual ordinary samples disagree with independently counted frame chunks')
                if ((scales[uid]>=1 and row['bytes']<before['bytes'])
                        or (scales[uid]<=1 and row['bytes']>before['bytes'])):
                    raise ValueError('Actual ordinary samples contradict the declared per-UE scaling direction')
        checks[recipe['id']] = dict(actual=actual, source='Production ArrivalModel and PacketLedger sampled every slot; immediate diagnostic drain, no network/PHY simulation',
            exact_parent_URLLC_and_voice=pattern == 'burst4', exact_parent_voice=True,
            exact_parent_URLLC_trace=pattern == 'burst4', URLLC_timing=timing,
            registered_URLLC_window_bytes_packets_sizes_preserved=True,
            nominal_embb_multiplier=recipe.get('embb_scale') if 'embb_scale_by_ue' not in recipe else None,
            nominal_embb_multiplier_by_ue=scales,
            ordinary_scaling_mode='per_ue' if 'embb_scale_by_ue' in recipe else 'uniform',
            physical_topology_unchanged=True,
            integer_rounding='Actual byte/packet totals are authoritative; nominal 5/10 percent scaling is not falsely claimed exact after integer truncation',
            mec_result_packets_per_task=recipe['mec_result_bytes']//200,
            fixed_mec_task_shape=dict(interval_ms=50,compute_gflop=.05,deadline_ms=100,input_bytes=1000),
            fixed_task_rate_per_source=20, initial_expected_mec_demand_gflops=sum(initial_mec_demand(services).values()))
        if initial_ue4_offload_fraction(recipe) != 1:
            routing=dict(change_kind='INITIAL TASK ROUTING', ue4_initial_offload_fraction=initial_ue4_offload_fraction(recipe),
                ue0_initial_offload_fraction=services['mec']['tasks_by_ue']['ue_0']['offload_fraction'],
                per_source_expected_mec_demand_gflops=initial_mec_demand(services),
                trained_actions_changed=False, F_ue4_budget_gflops=.75,
                ue4_initial_expected_demand_exceeds_F_budget=initial_mec_demand(services)['ue_4']>.75)
            differences[recipe['id']]['initial_task_routing']=routing
            checks[recipe['id']]['initial_task_routing']=routing
        if base_pin is not None:
            geometry = dict(**deepcopy(worker4.GEOMETRY_CHANGE), **base_pin,
                frozen_base_config=worker4.BASE_CONFIG,
                frozen_base_config_sha256=inputs['frozen_physical_files'][worker4.BASE_CONFIG],
                exact_semantic_delta=[dict(path='/ue_route_overrides', before=None, after=worker4.ROUTE_OVERRIDE)],
                initial_association_and_handover_unchanged=True,
                geometry_and_coverage_validation_required_before_ON=True)
            for report in (differences[recipe['id']], checks[recipe['id']]):
                report['physical_topology_unchanged'] = False
                report['geometry_change'] = deepcopy(geometry)
    pins.update(native_service_sha256=inputs['frozen_physical_files'][NATIVE_SERVICE], ap_profile_sha256=inputs['ap_profile_sha256'])
    cases = []
    recipes = {r['id']:r for r in design['recipes']}
    native_seen = set()
    for assignment in design['assignments']:
        key = (assignment['worker_id'], assignment['seed'])
        if design['include_native_reference'] and key not in native_seen:
            cases.append(_case(design, assignment, None, 'OFF', pins))
            native_seen.add(key)
        for ident in assignment['recipe_ids']:
            cases += [_case(design, assignment, recipes[ident], arm, pins) for arm in design['arms']]
    for wid in ('agx1', 'agx2'):
        plan_body = dict(schema=runner.SCHEMA, study_id=design['study_id'], stage=design['stage'], worker_id=wid,
            source_root=str(worker.SCOPE/'source'), source_archive_path=str(worker.SCOPE/'source-v2.tgz'),
            source_sha256=worker.SOURCE_SHA256, source_archive_sha256=worker.ARCHIVE_SHA256,
            runner_sha256=design['runner_sha256'], runner_path=str(worker.SCOPE/'runner'/('worker-v'+str(runner_version)+'.py')),
            max_parallel=max_parallel,
            active_plan_path=str(worker.SCOPE/'plans'/design['study_id']/(design['stage']+'.json')),
            output_root=str(worker.SCOPE/'source/outputs/environment_validation'/design['study_id']/design['stage']),
            frozen_models=inputs['frozen_models'], frozen_physical_files=inputs['frozen_physical_files'],
            deadline_utc=design['deadline_utc'], analysis_reserve_s=0, poll_seconds=2,
            deadline_interpretation='Failure-stop ceiling, not a new user-granted budget',
            design_sha256=digest(design), allow_scout_on=design['allow_scout_on'],
            scout_on_purpose='Environment discovery only; never training a model or quantifier from AFTER KPIs',
            final_acceptance='All six singleton target directions positive under one frozen recipe and two fresh validation seeds',
            cases=[c for c in cases if c['worker_id'] == wid])
        if runner_version == 4:
            plan_body.update(require_on_admission=True, geometry_change=deepcopy(worker4.GEOMETRY_CHANGE))
        plan = seal(plan_body)
        runner.validate_plan(plan, plan['plan_sha256'], plan_path=plan['active_plan_path'])
        files[design['stage']+'/'+wid+'.json'] = plan
    files['design.json'] = deepcopy(design)
    files['recipe-differences.json'] = dict(schema='oran.environment-recipe-differences.v1',
        parent=design['parent'], changes=differences)
    files['actual-load-checks.json'] = dict(schema='oran.environment-actual-load-checks.v1',
        physical_simulations=0, recipes=checks, parent=reference_load)
    files['protocol.json'] = seal(dict(schema='oran.environment-refinement-protocol.v1',
        design_sha256=digest(design), study_id=design['study_id'], stage=design['stage'],
        runner_version=runner_version, runner_sha256=design['runner_sha256'], max_parallel=max_parallel,
        physical_run_count=len(cases), recipes=pins, parent=design['parent'],
        source_sha256=worker.SOURCE_SHA256, source_archive_sha256=worker.ARCHIVE_SHA256,
        frozen_models=inputs['frozen_models'], baseline_slot=2040, origin_observe_slots=2000,
        observe_slots=observe, tail_slots=400, total_slots=2040+observe+400,
        allow_scout_on=design['allow_scout_on'], discovery_seeds=design.get('discovery_seeds', []),
        known_nonfresh_seeds=sorted(KNOWN_USED_SEEDS),
        scout_on_is_environment_discovery=design['stage'] == 'pilot' and len(design['arms']) > 1,
        final_fresh_seed_all_six_required=True, target_costs_allowed_but_reported=True,
        unchanged_topology=dict(ue_ids=sorted(USERS),ue_count=6,cell_count=3,
            geometry_route_radio_overrides_changed=False,
            interpretation=('These recipes alter arrivals, optional result payload and explicit INITIAL TASK ROUTING; actual seed/time-dependent associations remain measured outcomes.'
                if any(initial_ue4_offload_fraction(r)!=1 for r in design['recipes']) else
                'These recipes alter arrivals and optional result payload only; actual seed/time-dependent associations remain measured outcomes.')),
        missing_samples_are_not_success=True, singleton_action_logic_unchanged=True,
        default_changed=False, KG_recalibrated=False, model_training=False, remote_execution_started=False))
    if runner_version == 4:
        protocol = files['protocol.json']
        protocol['geometry_change'] = deepcopy(worker4.GEOMETRY_CHANGE)
        protocol['require_on_admission'] = True
        protocol['ON_admission_marker'] = 'ON_ADMISSION_ALLOWED'
        protocol['unchanged_topology'].update(geometry_route_radio_overrides_changed=True,
            radio_overrides_changed=False, other_five_routes_changed=False,
            interpretation='Six UEs and three cells remain; only UE0 initial phase on the original road changes. Radio, speed, controls and the other five trajectories are unchanged.')
        files['protocol.json'] = seal(protocol)
    files['manifest.json'] = dict(schema='oran.environment-refinement-files.v1',
        generator_path=str(Path(__file__).resolve()), generator_sha256=worker.sha(__file__),
        files_sha256={name:hashlib.sha256(encoded(value)).hexdigest() for name, value in files.items()})
    return files


def write_bundle(files, output):
    """Preflight every name/existing byte before creating any new file."""
    output = Path(output).resolve()
    for name, value in files.items():
        relative = Path(name)
        target = (output/relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(output) or '..' in relative.parts:
            raise ValueError('Protocol filename escapes output')
        if target.exists() and target.read_bytes() != encoded(value):
            raise ValueError('Never overwrite a retained protocol or seal: '+str(target))
    for name, value in files.items():
        path = output/name
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(encoded(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.design.read_text())
    files = build_bundle(design)
    write_bundle(files, args.output)
    print(json.dumps(dict(study_id=design['study_id'], stage=design['stage'],
                         output=str(args.output.resolve()), file_count=len(files), remote_execution_started=False)))


if __name__ == '__main__':
    main()
