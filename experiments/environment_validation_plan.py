"""Prepare frozen, OFF-screened environment experiments; never run simulations.

All pressure candidates share the same explicit offload-first starting policy.
Only ordinary traffic load and equal-mean URLLC burst timing differ. The pilot
selects by OFF pressure, before any singleton outcomes are collected.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCOPE = Path('/dev/shm/oran_environment_20261122')
SOURCE_SHA = 'cbe114446814007b275856e4bae92a2fe609abf4f4bcbbf8c32394a722a36af0'
ARCHIVE_SHA = '4f915fd649e287b68465c4aed047a7b626959b471d532abd8a659e376e3641e1'
SCHEMA = 'oran.environment-validation-worker-plan.v1'
SERVICE = 'configs/rt_sys/ue_services_mec_offload_first.json'
NATIVE_SERVICE = 'configs/rt_sys/ue_services.json'
RELEASE = 'artifacts/xapps/chongyu/20261117'
AP_PROFILE = 'configs/rt_sys/ap_services_reference.json'
APPS = dict(A='xapp:ue_energy_rf', B='xapp:ap_energy_rf',
            C='xapp:throughput_scheduler_mlp', D='xapp:latency_scheduler_rf',
            E='xapp:latency_urllc_rf', F='xapp:mec_offload_budget_qgbdt')
CANDIDATES = (
    dict(id='load125_peak4', traffic_load=1.25, peak_multiplier=4, pilot_worker='agx2'),
    dict(id='load150_peak2', traffic_load=1.5, peak_multiplier=2, pilot_worker='agx1'),
    dict(id='load150_peak4', traffic_load=1.5, peak_multiplier=4, pilot_worker='agx2'),
)
BASE_RATES = {'ue_2': 1_400_000, 'ue_6': 1_100_000}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def seal(value):
    result = deepcopy(value)
    result.pop('plan_sha256', None)
    result['plan_sha256'] = digest(result)
    return result


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError('Refusing to replace an existing protocol file: ' + str(path))
        return
    with path.open('xb') as stream:
        stream.write(payload)


def profile(candidate):
    peak = candidate['peak_multiplier']
    period = 40 * peak
    # BEFORE starts at slot 41. A half-ON boundary makes both adjacent 1 s
    # windows contain exactly the same number of ON samples, including 4x's
    # 12.5-period window. Every candidate's first ON segment starts at slot 21.
    offset = (20 - 41) % period
    users = {}
    for uid, base_rate in BASE_RATES.items():
        mean = int(base_rate * candidate['traffic_load'] * .0005 / 8)
        users[uid] = dict(arrival_model='on_off', arrival_rate_bps=mean * peak * 16_000,
                          packet_size_bytes=mean, on_duration_slots=40,
                          off_duration_slots=period - 40, burst_size_bytes=0,
                          start_offset_slots=offset)
    return dict(profile_id='environment_' + candidate['id'], slot_duration_ms=.5,
                provenance='Arrival-only OFF pilot. Preserve the actual integer CBR byte and packet means. '
                '20 ms ON; first ON at slot 21. Phase is fixed analytically for equal BEFORE/AFTER means, '
                'never selected using xApp outcomes. Radio, motion and packet deadlines are unchanged.',
                ues=users)


def validate_profile(candidate, value, root=ROOT):
    """Use the production arrival sampler and packet ledger, without a PHY run."""
    root = Path(root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from traffic.arrival_model import TrafficProfile, build_arrival_model
    from traffic.packet_ledger import PacketLedger
    allowed = {'arrival_model', 'arrival_rate_bps', 'packet_size_bytes', 'on_duration_slots',
               'off_duration_slots', 'burst_size_bytes', 'start_offset_slots'}
    if (set(value) != {'profile_id', 'slot_duration_ms', 'provenance', 'ues'}
            or value['slot_duration_ms'] != .5 or set(value['ues']) != set(BASE_RATES)
            or value['profile_id'] != 'environment_' + candidate['id']):
        raise ValueError('Arrival profile schema differs')
    result = {}
    for uid, fields in value['ues'].items():
        if set(fields) != allowed or fields != profile(candidate)['ues'][uid]:
            raise ValueError('Arrival fields differ from the declared equal-mean candidate')
        mean = int(BASE_RATES[uid] * candidate['traffic_load'] * .0005 / 8)
        period = fields['on_duration_slots'] + fields['off_duration_slots']
        windows = dict(cycle=(1, period), before=(41, 2040),
                       pilot_after=(2041, 4040), validation_after=(2041, 6040))
        sampler = build_arrival_model(TrafficProfile(ue_id=uid, **fields))
        rows = {}
        for name, (start, end) in windows.items():
            lc = SimpleNamespace(logical_channel_id=0, queue_bytes=0,
                                 max_queue_bytes=None, packet_delay_budget_ms=10)
            ledger = PacketLedger(SimpleNamespace(ue_id=uid, logical_channels=[lc]), .5)
            byte_sum = on_slots = 0
            packet_sizes = set()
            for slot in range(start, end + 1):
                amount = sampler.sample_bytes(slot_id=slot, slot_duration_s=.0005,
                                              rng=random.Random(0))
                byte_sum += amount
                on_slots += amount > 0
                ledger.add(lc, amount, slot, packet_size=fields['packet_size_bytes'])
                packet_sizes.update(e['bytes'] for e in ledger.events if e['kind'] == 'arrival')
                ledger.events.clear()
                if amount:
                    ledger.deliver(lc, amount, slot)
            count = end - start + 1
            expected_packets = count
            if (byte_sum != count * mean or ledger.stats['admitted_packets'] != expected_packets
                    or packet_sizes != {mean} or ledger.stats['offered_bytes'] != byte_sum):
                raise ValueError('Actual production arrival/packet means do not match native integer CBR')
            rows[name] = dict(start_slot=start, end_slot=end, sampled_slots=count,
                              on_slots=on_slots, bytes=byte_sum, packets=expected_packets,
                              packet_sizes_bytes=sorted(packet_sizes), mean_bytes_per_slot=byte_sum/count,
                              actual_mean_bps=byte_sum * 8 / (count * .0005),
                              native_integer_cbr_mean_bytes_per_slot=mean, exact_mean_match=True)
        result[uid] = rows
    return result


def frozen_inputs(root=ROOT):
    root = Path(root).resolve()
    native = json.loads((root/'configs/rt_sys/ue_services.json').read_text())
    service = json.loads((root/SERVICE).read_text())
    expected = deepcopy(native)
    expected['mec']['tasks_by_ue']['ue_4']['offload_fraction'] = 1.
    if service != expected:
        raise ValueError('The common service factor must change only initial UE4 offload .5 -> 1')
    task = service['mec']['tasks_by_ue']['ue_4']
    if {k:task[k] for k in ('interval_ms', 'compute_gflop', 'deadline_ms')} != dict(
            interval_ms=50, compute_gflop=.05, deadline_ms=100):
        raise ValueError('F requires its validated task profile')
    spec = importlib.util.spec_from_file_location('environment_release_verifier',
                                                 root/'scripts/orin/install-trained-xapps.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    release = module.load_release(root/RELEASE)
    models = {RELEASE+'/'+entry['path']: entry['sha256'] for entry in release['models'].values()}
    if len(models) != 6:
        raise ValueError('Exactly six frozen model files required')
    fixed = [SERVICE, NATIVE_SERVICE, AP_PROFILE, RELEASE+'/release.json', 'configs/rt_sys/agx_munich_6ue_depth2_fast.json',
             'configs/rt_sys/scene_catalog.json', 'configs/ue.yaml', 'configs/ue_intent_orchestration.yaml',
             'scripts/orin/run-six-ue-cuda-nr.sh']
    return dict(frozen_models=models, frozen_physical_files={p:sha(root/p) for p in fixed},
                service_sha256=sha(root/SERVICE), ap_profile_sha256=sha(root/AP_PROFILE),
                service_factor=dict(initial_ue4_offload_fraction=1., native_value=.5,
                    interpretation='Explicit common initial routing RCP; not increased task arrival or reduced edge capacity',
                    otherwise_identical_to_native=True), release='20261117')


def make_cases(stage, selected=None):
    if stage not in ('pilot', 'validation'):
        raise ValueError('Unknown stage')
    candidates = CANDIDATES if stage == 'pilot' else [c for c in CANDIDATES if c['id'] == selected]
    if stage == 'validation' and len(candidates) != 1:
        raise ValueError('Select exactly one preregistered pressure candidate')
    rows = []
    for candidate in candidates:
        origins = [(20261205, candidate['pilot_worker'])] if stage == 'pilot' else [
            (20261203, 'agx1'), (20261204, 'agx2')]
        for seed, worker in origins:
            arms = [dict(id='OFF', app_ids=[])]
            if stage == 'validation':
                arms += [dict(id=letter, app_ids=[app]) for letter, app in APPS.items()]
            for arm in arms:
                key = candidate['id'] + '_s' + str(seed)
                name = key + '_' + arm['id']
                identity = dict(study='environment-20261122', stage=stage, name=name)
                rows.append(dict(id=digest(identity)[:32], name=name, origin_key=key,
                    scene=candidate['id'], seed=seed, arm=arm['id'], app_ids=arm['app_ids'], worker_id=worker,
                    recipe_kind='selected_pressure',
                    slots=4440 if stage == 'pilot' else 6440, baseline_slot=2040,
                    origin_observe_slots=2000, observe_slots=2000 if stage == 'pilot' else 4000,
                    traffic_load=candidate['traffic_load'],
                    traffic_profile=str(SCOPE/'protocol/profiles'/(candidate['id']+'.json')),
                    ue_services_config=str(SCOPE/'source'/SERVICE),
                    ap_services_profile=str(SCOPE/'source'/AP_PROFILE)))
            if stage == 'validation':
                key = 'native_reference_s' + str(seed)
                name = key + '_OFF'
                native = deepcopy(rows[-len(arms)])
                native.update(id=digest(dict(study='environment-20261122', stage=stage, name=name))[:32],
                    name=name, origin_key=key, scene='native_reference', seed=seed, arm='OFF', app_ids=[],
                    recipe_kind='current_native_reference', traffic_load=1., traffic_profile=None,
                    ue_services_config=str(SCOPE/'source'/NATIVE_SERVICE))
                rows.insert(len(rows)-len(arms), native)
    return rows


def verify_worker_plan(plan):
    value = deepcopy(plan)
    pin = value.pop('plan_sha256', None)
    if pin != digest(value) or value.get('schema') != SCHEMA:
        raise ValueError('Worker plan schema/digest mismatch')
    if (value.get('source_root') != str(SCOPE/'source') or value.get('source_sha256') != SOURCE_SHA
            or value.get('source_archive_path') != str(SCOPE/'source-v2.tgz')
            or value.get('source_archive_sha256') != ARCHIVE_SHA or len(value.get('frozen_models', {})) != 6):
        raise ValueError('Worker physical source or six-model pin differs')
    stage = value['stage']
    if value.get('output_root') != str(SCOPE/'source/outputs/environment_validation'/stage):
        raise ValueError('Output must stay under the owned source outputs directory')
    expected = make_cases(stage, value.get('selected_candidate'))
    expected = [c for c in expected if c['worker_id'] == value['worker_id']]
    if len(expected) != len(value['cases']):
        raise ValueError('Incorrect number of cases assigned')
    seen_off = set()
    for declared, case in zip(expected, value['cases']):
        if any(case.get(k) != v for k, v in declared.items()):
            raise ValueError('Case differs from the frozen factorial design')
        if case['arm'] == 'OFF':
            seen_off.add(case['origin_key'])
        elif case['origin_key'] not in seen_off:
            raise ValueError('Every origin OFF must precede its singletons on the same worker')
        service = NATIVE_SERVICE if case['scene'] == 'native_reference' else SERVICE
        if case.get('ue_services_config_sha256') != value['frozen_physical_files'][service]:
            raise ValueError('Unequal service factor')
        expected_profile_sha = None if case['scene'] == 'native_reference' else value['traffic_profiles_sha256'][case['scene']]
        if case.get('traffic_profile_sha256') != expected_profile_sha:
            raise ValueError('Traffic profile checksum mismatch')
        if case.get('ap_services_profile_sha256') != value['frozen_physical_files'][AP_PROFILE]:
            raise ValueError('Unequal AP service factor')
    return True


def prepare(output, deadline_utc, *, stage='pilot', selected=None, selection_report=None,
            runner_sha256=None, root=ROOT):
    output = Path(output).resolve()
    if (not isinstance(runner_sha256, str) or len(runner_sha256) != 64
            or any(c not in '0123456789abcdef' for c in runner_sha256)):
        raise ValueError('An exact final runner SHA-256 is required before writing worker plans')
    deadline = datetime.fromisoformat(deadline_utc)
    if deadline.tzinfo is None or deadline <= datetime.now(timezone.utc):
        raise ValueError('Use an explicit future timezone-aware failure-stop deadline')
    inputs = frozen_inputs(root)
    rules_path = output/'decision-rules.json'
    decision_rules = dict(path='decision-rules.json', sha256=sha(rules_path)) if rules_path.is_file() else None
    profiles = {c['id']:profile(c) for c in CANDIDATES}
    proof = {c['id']:validate_profile(c, profiles[c['id']], root) for c in CANDIDATES}
    for name, data in profiles.items():
        write_new(output/'profiles'/(name+'.json'), data)
    profiles_sha = {name:sha(output/'profiles'/(name+'.json')) for name in profiles}
    selection = None
    if stage == 'validation':
        if selection_report is None:
            raise ValueError('Validation requires an explicit OFF-only selection report')
        selected_path = Path(selection_report).resolve()
        evidence = json.loads(selected_path.read_text())
        if evidence.get('candidate') != selected or evidence.get('uses_xapp_outcomes') is not False:
            raise ValueError('Selection must identify this candidate and explicitly exclude singleton outcomes')
        selection = dict(path=str(selected_path), sha256=sha(selected_path), uses_xapp_outcomes=False)
    cases = make_cases(stage, selected)
    for case in cases:
        is_native = case['scene'] == 'native_reference'
        case.update(traffic_profile_sha256=None if is_native else profiles_sha[case['scene']],
                    ue_services_config_sha256=inputs['frozen_physical_files'][NATIVE_SERVICE if is_native else SERVICE],
                    ap_services_profile_sha256=inputs['ap_profile_sha256'])
    files = []
    for worker in ('agx1', 'agx2'):
        plan = seal(dict(schema=SCHEMA, study_id='environment-validation-20261122', stage=stage,
            worker_id=worker, source_root=str(SCOPE/'source'), source_archive_path=str(SCOPE/'source-v2.tgz'),
            output_root=str(SCOPE/'source/outputs/environment_validation'/stage),
            source_archive_sha256=ARCHIVE_SHA, source_sha256=SOURCE_SHA,
            frozen_models=inputs['frozen_models'], frozen_physical_files=inputs['frozen_physical_files'],
            traffic_profiles_sha256=profiles_sha, selected_candidate=selected,
            selection_report=selection, decision_rules=decision_rules,
            deadline_utc=deadline_utc, analysis_reserve_s=0, poll_seconds=2,
            deadline_interpretation='Failure-stop ceiling only; not a new user-granted time budget',
            runner_sha256=runner_sha256,
            cases=[c for c in cases if c['worker_id'] == worker]))
        verify_worker_plan(plan)
        path = output/stage/(worker+'.json')
        write_new(path, plan)
        files.append(path)
    protocol = dict(schema='oran.environment-validation-protocol.v1', stage=stage,
        candidates=list(CANDIDATES), apps=APPS, inputs=inputs, source_sha256=SOURCE_SHA,
        source_archive_sha256=ARCHIVE_SHA, selected_candidate=selected, selection_report=selection,
        decision_rules=decision_rules,
        baseline_slot=2040, origin_observe_slots=2000, observe_slots=2000 if stage == 'pilot' else 4000,
        tail_slots=400, total_slots=4440 if stage == 'pilot' else 6440, slot_duration_ms=.5,
        physical_run_count=len(cases), pilot_seed=20261205, validation_seeds=[20261203, 20261204],
        selection_policy=dict(uses_xapp_outcomes=False, automatic_default_replacement=False,
            basis='OFF-only offered/delivered load, nonzero URLLC deadline/overflow failures, mature MEC task latency and backlog',
            cautions=['Reject near-total failure or permanent all-UE saturation.',
                      'Do not require every KPI absolute value to worsen; added offered load can raise goodput.',
                      'Do not select by singleton gains or by whether xApps propose different actions.']),
        comparison='Each singleton AFTER versus its matched same-seed selected-pressure OFF AFTER; BEFORE/AFTER reported separately',
        native_reference=dict(included=stage == 'validation', runs=2 if stage == 'validation' else 0,
            traffic_load=1., traffic_profile=None, ue_services_config=NATIVE_SERVICE,
            purpose='Same-seed, same-host, equal-duration comparison of new environment with current native environment',
            changed_factors=['ordinary offered-load multiplier', 'URLLC timing at preserved within-recipe mean',
                             'initial UE4 offload probability .5 -> 1']),
        phase_policy='Fixed first-ON slot 21 chosen analytically before outcomes to conserve payload and packet means in all scored windows',
        retained_raw_data=True, model_training=False, network_algorithm_changes=False,
        current_ui_changes=False, remote_execution_started=False)
    design_path = output/(stage+'-protocol.json')
    write_new(design_path, seal(protocol))
    files.append(design_path)
    proof_path = output/(stage+'-arrival-conservation-proof.json')
    write_new(proof_path, dict(schema='oran.environment-arrival-proof.v1',
        method='Production traffic.arrival_model and traffic.packet_ledger, sampled every actual slot',
        network_simulation=False, results=proof))
    files.append(proof_path)
    files += [output/'profiles'/(name+'.json') for name in profiles]
    if decision_rules:
        files.append(rules_path)
    manifest = dict(schema='oran.environment-protocol-file-manifest.v1', stage=stage,
                    generator_path=str(Path(__file__).resolve()), generator_sha256=sha(__file__),
                    files={str(p.relative_to(output)):sha(p) for p in files})
    manifest_path = output/(stage+'-manifest.json')
    write_new(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--stage', choices=['pilot', 'validation'], default='pilot')
    parser.add_argument('--selected-candidate', choices=[c['id'] for c in CANDIDATES])
    parser.add_argument('--selection-report', type=Path)
    parser.add_argument('--runner-sha256', required=True)
    args = parser.parse_args()
    manifest = prepare(args.output, args.deadline_utc, stage=args.stage,
                       selected=args.selected_candidate, selection_report=args.selection_report,
                       runner_sha256=args.runner_sha256)
    print(json.dumps(dict(stage=args.stage, output=str(args.output.resolve()),
                          file_count=len(manifest['files']), remote_execution_started=False)))


if __name__ == '__main__':
    main()
