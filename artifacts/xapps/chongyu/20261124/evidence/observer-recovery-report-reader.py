#!/usr/bin/env python3
"""Explicit v2 recovery report with a narrowly scoped legal-tar-directory amendment.

The selected weights, six requests, two seeds and ten-second windows are unchanged.
The original failed observer status/seal remain failures. A separately approved
supplemental seal may admit only the same uninterrupted 9ca87 TH simulation.
"""
from __future__ import annotations
import argparse
import html
import importlib.util
import json
from pathlib import Path
import re

TASK = Path(__file__).resolve().parents[1]
BASE_PATH = Path(__file__).with_name('report_xapp_comparisons.py')
BASE_SHA = '93ca5b396f72f7a7c0dc1617774262925ba842a5271882a62eba69d1a81132d6'
OLD_WATCHER_PATH = Path(__file__).with_name('watch_final_report_v5_v1.py')
OLD_WATCHER_SHA = '89871ee4c51325532fc9eca232305eccd31cc19a1cd8f8c784074419d5140d70'
OLD_REGISTRATION_PATH = TASK/'analysis/final-report-coordinator-v5-v1/registration.json'
OLD_REGISTRATION_SHA = 'e4340b2d461a50747334e7635ae44ac50fb0f2079c2c0f48639a46362d347a74'
RECOVERED_ID = '9ca87f904d75424aad23c0821cf8b1c7'
RECOVERED_REQUEST_SHA = '94df2c5bdde674ded0e79c64c1f3d437bb87ea49a125ce9bdb9917261108f94b'
ORIGINAL_FAILED_SEAL_SHA = 'ceb8f5bed288e6340997e85fa9754c921f1aebe4ade02d626142b04bfa777013'
ORIGINAL_FAILED_STATUS_SHA = '866fa7c9653d1db5945e3a977c5a886ee8e455d0ba1bdeb4844df90ecd20a644'
FAILURE_ENVELOPE_SHA = '074c63724d77f5c2b864453807849f44958e0346f562e9248a87439f2ae29040'
ROOT_GO_SHA = '2ce1718d1750c88f383c1502601bcf80a401d9f3bcbb797c1cc8278b921b914c'
SIMULATION_CMDLINE_SHA = '38273a9b7ce8070f943a9d8b7e8a60813d3defe7d34558eb6b65338ef425c6cc'
ORIGINAL_RECOVERY_READER = TASK/'workers/v5/observer-recovery-reader-v1.py'
ORIGINAL_RECOVERY_READER_SHA = '46baef20fc933ff789521e74919146e27aac2aa3ea141906cc563a5fcb70c726'
SUPERSEDED_ADAPTER = TASK/'analysis/report_xapp_comparisons_observer_recovery_v1.py'
SUPERSEDED_ADAPTER_SHA = '56c48e71e059438784fb210b408440a4a56264a4d7746882f2e72ac599105423'
SUPERSEDED_COORDINATOR = TASK/'analysis/watch_final_report_observer_recovery_v1.py'
SUPERSEDED_COORDINATOR_SHA = 'e60b2a499093dc60c4dcc547f2383d571685f7aeb1b4eeec35a21d6b66eb5846'
SUPERSEDED_REGISTRATION = TASK/'analysis/final-report-observer-recovery-coordinator-v1/registration.json'
SUPERSEDED_REGISTRATION_SHA = '8497695c5534c978ab75b15adda58187059823eb06642e300384e2e40de9eb79'
# Immutable metadata-only recovery approval; this never authorizes a rerun.
RECOVERY_REGISTRATION = {
    'path': str(TASK/'workers/v5/observer-recovery-registration-v1.json'),
    'sha256': '80c455da6c27ec6e0bf0f0ca71b567ceb9bfb1633a2fec6a9eb2cdf079216e3f',
    'reader_path': str(TASK/'workers/v5/observer-recovery-reader-v2.py'),
    'reader_sha256': '2c926fb87ad3b791e78ec23b0008e4af1d07eef154eef3a359c18e467ce3e783',
    'reader_amendment_path': str(TASK/'workers/v5/observer-recovery-reader-amendment-v2.json'),
    'reader_amendment_sha256': 'df1b1987c69a815e91233ffe9f57592e8a7f020c164371c4f4cc7785aa9ceac3'}


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _module(path, pin, name):
    if _sha(path) != pin:
        raise ValueError('An archival/registered reader changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _module(BASE_PATH, BASE_SHA, '_observer_recovery_canonical_base')
OLD = _module(OLD_WATCHER_PATH, OLD_WATCHER_SHA, '_observer_recovery_archival_coordinator')
ORIGINAL_VERIFY_CASE = BASE.verify_case
read, digest, file_sha = BASE.read, BASE.digest, BASE.file_sha


def recovered_case():
    return (TASK/'runs/agx1'/RECOVERED_ID).resolve()


def approved_reader():
    reference = RECOVERY_REGISTRATION
    if not isinstance(reference, dict):
        raise ValueError('No immutable single-case observer-recovery registration is active')
    registration_path = Path(reference['path']).resolve()
    if _sha(registration_path) != reference['sha256']:
        raise ValueError('The explicitly approved recovery registration changed')
    if _sha(reference['reader_amendment_path']) != reference['reader_amendment_sha256']:
        raise ValueError('The explicit legal-directory-header reader amendment changed')
    amendment = read(reference['reader_amendment_path'])
    expected = {'schema': 'oran.observer-recovery-reader-amendment.v2', 'approved': True,
        'case_id': RECOVERED_ID, 'original_registration_sha256': reference['sha256'],
        'original_reader_sha256': ORIGINAL_RECOVERY_READER_SHA, 'reader_sha256': reference['reader_sha256'],
        'source_archive_sha256': OLD.SOURCE_SHA, 'permitted_change': 'accept-confined-tar-directory-headers-only',
        'original_case_evidence_changed': False, 'new_simulations': 0}
    if any(amendment.get(key) != value for key, value in expected.items()):
        raise ValueError('The v2 reader amendment must change only legal confined tar-directory handling')
    for path, pin in ((ORIGINAL_RECOVERY_READER, ORIGINAL_RECOVERY_READER_SHA),
                      (SUPERSEDED_ADAPTER, SUPERSEDED_ADAPTER_SHA),
                      (SUPERSEDED_COORDINATOR, SUPERSEDED_COORDINATOR_SHA),
                      (SUPERSEDED_REGISTRATION, SUPERSEDED_REGISTRATION_SHA)):
        if _sha(path) != pin:
            raise ValueError('A superseded recovery artifact was overwritten')
    reader = _module(Path(reference['reader_path']), reference['reader_sha256'],
                     '_report_exact_observer_recovery_'+reference['reader_sha256'])
    return reader, read(registration_path), reference


def exact_scope(path, *, role, seed, actor):
    if (Path(path).resolve() != recovered_case() or role != 'holdout'
            or seed != 341721 or actor != 'throughput'):
        raise ValueError('Observer recovery is allowed only for the exact jointly locked 9ca87 TH run')


def _plan_scope(plan_path, protocol_path):
    old_value, joint, _ = OLD.verify_registration(OLD_REGISTRATION_PATH, OLD_REGISTRATION_SHA)
    path = Path(plan_path).resolve()
    plan = read(path)
    if (plan.get('schema') != 'oran.xapp-comparison-plan.v1' or plan.get('role') != 'holdout'
            or plan.get('synthetic_fixture') is True or plan.get('selection') != old_value['selection']
            or plan.get('joint_plan_sha256') != OLD.JOINT_SHA
            or _sha(protocol_path) != old_value['protocol']['sha256']
            or plan.get('canonical_report_adapter') != {'path': str(Path(__file__).resolve()), 'sha256': _sha(__file__)}
            or plan.get('original_worker_failure_preserved') is not True
            or plan.get('weights_origin') != {'power': 'new_target_scaled_rf_144_train_rows', 'throughput': 'retained_legacy'}
            or plan.get('release_weight_composition') != {'new': ['x01'], 'unchanged': ['x02', 'x08', 'x12_mec', 'x12_urllc', 'x13']}):
        raise ValueError('Recovery reporting cannot change the six requests, model lock, role or protocol')
    expected = {}
    for entry in plan.get('seeds', []):
        seed = entry.get('seed')
        if seed not in (341721, 341722) or (seed, 'off') in expected:
            raise ValueError('Each exact registered seed and shared OFF must appear once')
        expected[seed, 'off'] = BASE.resolve(entry['off_case'], path.parent)
        ons = entry.get('on_cases', [])
        if len(ons) != 2 or {item.get('actor') for item in ons} != {'power', 'throughput'}:
            raise ValueError('Recovery does not permit adding, dropping or substituting actors')
        for item in ons:
            if item.get('label') != 'selected':
                raise ValueError('Only the already jointly locked selected actors are admitted')
            expected[seed, item['actor']] = BASE.resolve(item['case'], path.parent)
    if set(expected) != set(OLD.IDS):
        raise ValueError('The comparison plan must contain the original six final cases')
    for key, actual in expected.items():
        worker, ident = OLD.IDS[key]
        if actual != (TASK/'runs'/worker/ident).resolve():
            raise ValueError('A final run was substituted after observer failure')
    ref = plan.get('observer_recovery_registration')
    if (not RECOVERY_REGISTRATION or ref != {key: RECOVERY_REGISTRATION[key] for key in ('path', 'sha256')}):
        raise ValueError('Plan must explicitly bind the registered recovery approval')
    if plan.get('observer_recovery_reader_amendment') != {
            'path': RECOVERY_REGISTRATION['reader_amendment_path'], 'sha256': RECOVERY_REGISTRATION['reader_amendment_sha256']}:
        raise ValueError('The v2 plan must explicitly bind the separate reader-only amendment')
    return plan, old_value, joint


def metadata_readiness(old_value, joint):
    """Inspect every completion state before any audit/summary/trace KPI."""
    pending, recovery_certificate_path = None, None
    for item in joint['cases']:
        request = item['request']
        case = (TASK/'runs'/request['worker_id']/request['id']).resolve()
        if not (case/'request.json').is_file() or not (case/'status.json').is_file():
            pending = pending or 'awaiting_registered_case:'+request['id']
            continue
        if read(case/'request.json') != request:
            raise ValueError('An actual request differs from the original joint lock')
        status = read(case/'status.json')
        if request['id'] == RECOVERED_ID:
            if status.get('phase') != 'failed':
                raise ValueError('The original failed observer status must remain failed')
            cert_path = case/'observer-recovery-v1/recovery-certificate.json'
            if not cert_path.is_file():
                pending = pending or 'awaiting_explicit_same_process_recovery_certificate'
            else:
                recovery_certificate_path = cert_path
        else:
            if status.get('phase') == 'failed':
                raise ValueError('Recovery cannot admit a second failed final observer')
            if status.get('phase') != 'complete' or status.get('sealed') is not True or status.get('slot') != BASE.TOTAL:
                pending = pending or 'awaiting_other_five_original_complete:'+request['id']
            if not (case/'sealed-manifest.json').is_file() or not (case/'mirror-verification.json').is_file():
                pending = pending or 'awaiting_other_five_original_seal_mirror:'+request['id']
    if pending:
        return False, pending
    certificate = read(recovery_certificate_path)
    if (certificate.get('schema') != 'oran.validation-observer-recovery.v1'
            or certificate.get('passed') is not True
            or certificate.get('same_native_simulation_completed') is not True
            or certificate.get('no_restart') is not True):
        raise ValueError('The observer recovery has not established same-process completion')
    for seed in (341721, 341722):
        for actor in ('power', 'throughput'):
            path = Path(old_value['prefix_certificates'][str(seed)][actor])
            if not path.is_file() or not path.with_name(path.name+'.sha256.json').is_file():
                return False, 'awaiting_original_prefix:'+str(seed)+'-'+actor
        path = Path(old_value['power_semantic_certificates'][str(seed)])
        if not path.is_file() or not path.with_name(path.name+'.sha256.json').is_file():
            return False, 'awaiting_original_power_semantics:'+str(seed)
    return True, 'all_six_actually_completed_with_one_explicit_observer_recovery'


def recovered_lane_inputs(case, files):
    """Keep the frozen 8ccb lane-input contract; do not claim its complete gate passed."""
    ref = BASE.V5_VALIDATION_REGISTRATION
    driver = _module(Path(ref['driver_path']), ref['driver_sha256'], '_recovered_original_lane_inputs')
    protocol_path = Path(ref['path'])
    if _sha(protocol_path) != ref['protocol_sha256']:
        raise ValueError('Original registered lane protocol changed')
    protocol = read(protocol_path)
    required = {'operator-driver.json', 'operator-driver.py', 'operator-protocol.json', 'operator-case-plan.json',
        'operator-root-go.json', 'operator-source-members.json', 'operator-selection.json', 'command.json'}
    required |= {'operator-profiles/'+name for name, _ in driver.PROFILE_PINS.values()}
    if not required.issubset(files):
        raise ValueError('Supplemental seal omits actual original lane/profile/GO provenance')
    request, receipt, plan = (read(case/name) for name in ('request.json', 'operator-driver.json', 'operator-case-plan.json'))
    root = driver.validate_plan(plan)
    if (plan['request'] != request or not plan.get('root_go')
            or _sha(case/'operator-root-go.json') != plan['root_go']['sha256']
            or _sha(case/'operator-source-members.json') != plan['source_members']['sha256']):
        raise ValueError('Actual original operator plan/GO/source inventory differs')
    driver.check_root_go(plan, read(case/'operator-root-go.json'))
    if _sha(case/'operator-selection.json') != OLD.SELECTION_SHA:
        raise ValueError('Actual original joint selection copy changed')
    driver.check_joint_selection(read(case/'operator-selection.json'), request, protocol, ref['protocol_sha256'])
    expected = {'schema': 'oran.validation-lane-driver-receipt.v1', 'validation_version': 'v5',
        'physical_worker_id': 'agx1', 'lane_id': 'lane-1', 'case_id': RECOVERED_ID,
        'request_sha256': RECOVERED_REQUEST_SHA, 'source_archive_sha256': OLD.SOURCE_SHA,
        'project_root': str(root), 'max_validation_lanes_per_host': 2, 'original_worker_validate_passed': True,
        'old_v4_exclusive_lock_inactive': True, 'lane_lock_inherited_by_worker': True,
        'legacy_shared_guard_inherited_by_worker': True,
        'validation_protocol_sha256': ref['protocol_sha256'], 'driver_sha256': ref['driver_sha256']}
    if (any(receipt.get(key) != value for key, value in expected.items())
            or _sha(case/'operator-protocol.json') != ref['protocol_sha256']
            or read(case/'operator-protocol.json') != protocol or _sha(case/'operator-driver.py') != ref['driver_sha256']):
        raise ValueError('Actual original lane/source/protocol/driver identity differs')
    paths = {}
    for key, (name, pin) in driver.PROFILE_PINS.items():
        path = str(root.parent/'profiles'/name)
        entry = {'path': path, 'sha256': pin, 'sealed_copy': 'operator-profiles/'+name}
        if (receipt.get('profiles', {}).get(key) != entry
                or request['profiles'].get(key) != {'path': path, 'sha256': pin}
                or _sha(case/entry['sealed_copy']) != pin):
            raise ValueError('Actual original profile byte copy or exact path/pin differs')
        paths[key] = entry
    command = read(case/'command.json'); argv = command['argv']
    if str(root/'scripts/orin/activate-gpu.sh') not in argv:
        raise ValueError('Recovered command activated a different source')
    for flag in ('--output', '--live-state', '--channel-cache-root'):
        if not Path(argv[argv.index(flag)+1]).is_relative_to(root):
            raise ValueError('An actual recovered writable path was shared or escaped')
    if any(not Path(value).is_relative_to(root) for value in command['cache_environment'].values()):
        raise ValueError('An actual recovered runtime cache was shared or escaped')
    return {'passed': True, 'lane_id': 'lane-1', 'project_root': str(root), 'profiles': paths,
        'driver_sha256': ref['driver_sha256'], 'validation_protocol_sha256': ref['protocol_sha256']}


def recovery_evidence():
    reader, registration, reference = approved_reader()
    case, folder = recovered_case(), recovered_case()/'observer-recovery-v1'
    verified = reader.verify_recovered_case(case, registration,
        reader_amendment=reference['reader_amendment_path'])
    if not isinstance(verified, dict) or not isinstance(verified.get('evidence'), dict):
        raise ValueError('Recovery owner did not return verified explicit evidence')
    for field, name in (('status_path', 'complete-status.json'), ('seal_path', 'supplemental-seal.json'), ('audit_path', 'audit.json')):
        if Path(verified.get(field, '')).resolve() != folder/name:
            raise ValueError('Recovery reader redirected a derived metadata input')
    cert_path = folder/'recovery-certificate.json'
    certificate = read(cert_path)
    expected = {'schema': 'oran.validation-observer-recovery.v1', 'passed': True, 'id': RECOVERED_ID,
        'request_sha256': RECOVERED_REQUEST_SHA, 'original_failed_seal_sha256': ORIGINAL_FAILED_SEAL_SHA,
        'original_failed_status_sha256': ORIGINAL_FAILED_STATUS_SHA,
        'failure_envelope_sha256': FAILURE_ENVELOPE_SHA, 'root_go_sha256': ROOT_GO_SHA,
        'source_archive_sha256': OLD.SOURCE_SHA,
        'validation_protocol_sha256': '6ac948119da228b83facbed26d3624559994557d29bd03b972fe727c89893fc2',
        'driver_sha256': '8ccb1e4797a016a5468f4965a8b4a525eac280a1414fda8565c907d783935bdd',
        'selection_sha256': OLD.SELECTION_SHA, 'model_sha256': OLD.MODELS['throughput'],
        'observer_failure': 'FileNotFoundError:run/live.tmp', 'original_worker_phase': 'failed',
        'same_native_simulation_completed': True, 'no_restart': True}
    if any(certificate.get(key) != value for key, value in expected.items()):
        raise ValueError('Recovery certificate does not match the exact approved observer failure and model/source lock')
    body = dict(certificate)
    identifier = body.pop('certificate_id', None)
    if identifier != digest(body) or read(cert_path.with_name(cert_path.name+'.sha256.json')).get('certificate_sha256') != _sha(cert_path):
        raise ValueError('Recovery certificate canonical ID/creation receipt changed')
    continuity = certificate.get('pid_continuity') or {}
    if (continuity.get('pid') != 25336 or continuity.get('start_ticks') != 15559801
            or continuity.get('cmdline_sha256') != SIMULATION_CMDLINE_SHA
            or continuity.get('exit_code') is not None
            or continuity.get('exit_code_status') != 'unavailable_non_child'
            or type(continuity.get('before_producer_slot')) is not int
            or not 8865 <= continuity['before_producer_slot'] < BASE.TOTAL
            or BASE.utc(continuity['process_end_observed_utc']) < BASE.utc(continuity['before_observed_utc'])):
        raise ValueError('Recovery cannot claim a restart, another process, or an unavailable successful exit code')
    original_status = read(case/'status.json')
    if (original_status.get('phase') != 'failed' or _sha(case/'status.json') != ORIGINAL_FAILED_STATUS_SHA
            or _sha(case/'sealed-manifest.json') != ORIGINAL_FAILED_SEAL_SHA
            or read(case/'sealed-manifest.json').get('failed') is not True):
        raise ValueError('The actual failed observer record was replaced')
    derived = read(folder/'complete-status.json')
    if (derived.get('schema') != 'oran.observer-recovered-status.v1' or derived.get('phase') != 'complete'
            or derived.get('original_worker_phase') != 'failed'
            or derived.get('completion_basis') != 'same_native_process_after_observer_failure'):
        raise ValueError('Derived recovery status must preserve the original failed-observer meaning')
    seal_path = folder/'supplemental-seal.json'
    seal = read(seal_path)
    if (seal.get('schema') != 'oran.validation-observer-recovery-sealed.v1'
            or certificate.get('supplemental_seal_sha256') != _sha(seal_path)
            or seal.get('id') != RECOVERED_ID or seal.get('request_sha256') != RECOVERED_REQUEST_SHA):
        raise ValueError('A separately identified supplemental seal is required')
    files = seal.get('files')
    required = {'request.json', 'status.json', 'sealed-manifest.json',
        'run/config_snapshot.json', 'run/final_summary.json', 'run/run_status.json',
        'run/throughput_mlp_model.json', 'observer-recovery-v1/audit.json',
        'observer-recovery-v1/complete-status.json', 'observer-recovery-v1/root-go.json',
        'observer-recovery-v1/pid-before.json', 'observer-recovery-v1/checks-before.json',
        'observer-recovery-v1/checks-after.json'}
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError('Supplemental seal lacks an actually consumed stable file')
    trace = [case/name for name in ('run/rt_sys_trace.jsonl.gz', 'run/rt_sys_trace.jsonl') if (case/name).is_file()]
    if len(trace) != 1 or str(trace[0].relative_to(case)) not in files:
        raise ValueError('The actual recovered full trace must belong to its supplemental seal')
    for relative, entry in files.items():
        rel = Path(relative); path = case/rel
        if (rel.is_absolute() or '..' in rel.parts or path.is_symlink() or not path.resolve().is_relative_to(case)
                or not path.is_file() or path.stat().st_size != entry.get('bytes') or _sha(path) != entry.get('sha256')):
            raise ValueError('A supplemental stable file is missing, changed or unconfined')
    mirror = read(folder/'mirror-verification.json')
    if mirror.get('supplemental_manifest_sha256') != _sha(seal_path) or mirror.get('files_verified') != len(files):
        raise ValueError('Recovery mirror does not verify every supplemental stable member')
    supplied = verified['evidence']
    if supplied.get('original_failed_status_sha256') != _sha(case/'status.json'):
        raise ValueError('The explicit recovery evidence does not bind the preserved failed status')
    supplied_lane = supplied.get('validation_lane')
    if (not isinstance(supplied_lane, dict) or supplied_lane.get('passed') is not True
            or supplied_lane.get('driver_sha256') != expected['driver_sha256']
            or supplied_lane.get('validation_protocol_sha256') != expected['validation_protocol_sha256']
            or supplied_lane.get('lane_id') != 'lane-1'):
        raise ValueError('The recovered case does not prove the original registered isolated lane/profile inputs')
    lane = recovered_lane_inputs(case, files)
    if supplied_lane.get('project_root') != lane['project_root']:
        raise ValueError('Owner recovery evidence differs from the independently checked original nested ROOT')
    members = read(case/'operator-source-members.json')
    if (certificate.get('audit_function') != 'audit_run'
            or certificate.get('audit_worker_sha256') != members['experiments/throughput_improvement_worker.py']['sha256']):
        raise ValueError('Recovered audit did not use the actual original sealed throughput audit_run')
    evidence_files = {}
    for index, (label, item) in enumerate((supplied.get('evidence_files') or {}).items()):
        if not isinstance(item, dict) or _sha(item['path']) != item.get('sha256'):
            raise ValueError('A recovered failure/approval evidence file changed')
        safe = re.sub('[^A-Za-z0-9_-]', '-', str(label))
        evidence_files[f'owner-{index:02d}-{safe}'] = {'path': str(Path(item['path']).resolve()), 'sha256': item['sha256']}
    required_evidence = {'original-failed-status': case/'status.json', 'original-failed-seal': case/'sealed-manifest.json',
        'root-go': folder/'root-go.json', 'supplemental-seal': seal_path, 'recovery-certificate': cert_path,
        'recovery-certificate-receipt': cert_path.with_name(cert_path.name+'.sha256.json'),
        'recovery-mirror': folder/'mirror-verification.json', 'recovery-complete-status': folder/'complete-status.json',
        'recovery-audit': folder/'audit.json', 'recovery-reader': Path(reference['reader_path']),
        'recovery-registration': Path(reference['path']), 'canonical-base-reader': BASE_PATH,
        'recovery-reader-amendment': Path(reference['reader_amendment_path']),
        'original-recovery-reader': ORIGINAL_RECOVERY_READER, 'superseded-recovery-adapter': SUPERSEDED_ADAPTER,
        'superseded-recovery-coordinator': SUPERSEDED_COORDINATOR, 'superseded-recovery-registration': SUPERSEDED_REGISTRATION,
        'canonical-recovery-adapter': Path(__file__), 'archival-coordinator': OLD_WATCHER_PATH,
        'archival-coordinator-registration': OLD_REGISTRATION_PATH}
    for name in ('operator-driver.json', 'operator-driver.py', 'operator-protocol.json', 'operator-case-plan.json',
                 'operator-root-go.json', 'operator-source-members.json', 'operator-selection.json'):
        required_evidence['original-'+name.replace('.', '-')] = case/name
    for key, item in lane['profiles'].items():
        required_evidence['original-profile-'+key.replace('_', '-')] = case/item['sealed_copy']
    for label, path in required_evidence.items():
        evidence_files[label] = {'path': str(path.resolve()), 'sha256': _sha(path)}
    # The owner must expose the actual immutable failure envelope, not just its digest.
    if not any(item['sha256'] == FAILURE_ENVELOPE_SHA for item in evidence_files.values()):
        raise ValueError('Actual approved failure envelope is missing from the recovery evidence map')
    return {'case': str(case), 'case_id': RECOVERED_ID, 'original_worker_phase': 'failed',
        'same_native_simulation_completed': True, 'no_restart': True,
        'request_sha256': RECOVERED_REQUEST_SHA, 'source_archive_sha256': OLD.SOURCE_SHA,
        'selection_sha256': OLD.SELECTION_SHA, 'model_sha256': OLD.MODELS['throughput'],
        'original_failed_status_sha256': _sha(case/'status.json'), 'original_failed_seal_sha256': ORIGINAL_FAILED_SEAL_SHA,
        'failure_envelope_sha256': FAILURE_ENVELOPE_SHA, 'root_go_sha256': ROOT_GO_SHA,
        'supplemental_seal_sha256': _sha(seal_path), 'supplemental_files_verified': len(files),
        'recovery_certificate_sha256': _sha(cert_path), 'pid_continuity': continuity,
        'audit_function': 'audit_run', 'audit_worker_sha256': certificate['audit_worker_sha256'],
        'reader_amendment': {'path': reference['reader_amendment_path'], 'sha256': reference['reader_amendment_sha256'],
            'original_reader_sha256': ORIGINAL_RECOVERY_READER_SHA, 'actual_reader_sha256': reference['reader_sha256'],
            'original_registration_sha256': reference['sha256'], 'original_case_files_changed': False},
        'exit_code': None, 'exit_code_status': 'unavailable_non_child',
        'validation_lane': {**lane, 'original_worker_seal_audited': False, 'supplementary_seal_audited': True,
            'original_worker_lane_inputs_audited': True, 'reader_path': str(Path(reference['reader_path']).resolve()),
            'reader_sha256': reference['reader_sha256']}, 'evidence_files': evidence_files}


def verify_recovered_case(path, *, role, seed, actor, power_selection=None,
                          expected_source_sha256=None, validation_context=None):
    exact_scope(path, role=role, seed=seed, actor=actor)
    proof = recovery_evidence()
    case, run = recovered_case(), recovered_case()/'run'
    request, status = read(case/'request.json'), read(case/'status.json')
    body = dict(request); pin = body.pop('request_sha256', None)
    if (pin != RECOVERED_REQUEST_SHA or digest(body) != pin or request.get('id') != RECOVERED_ID
            or request.get('schema') != 'oran.throughput-improvement-job.v1'
            or request.get('stage') != role or request.get('seed') != seed or request.get('mode') != 'frozen'
            or request.get('slots') != BASE.TOTAL or request.get('balanced_start_windows') != 0
            or request.get('source_archive_sha256') != OLD.SOURCE_SHA
            or request.get('selection_sha256') != OLD.SELECTION_SHA
            or request.get('initial_model_sha256') != OLD.MODELS['throughput']
            or expected_source_sha256 != OLD.SOURCE_SHA or validation_context is None):
        raise ValueError('The completed same-process recovery differs from its exact jointly locked request')
    if (validation_context.get('source_archive_sha256') != OLD.SOURCE_SHA
            or validation_context.get('sha256') != '6ac948119da228b83facbed26d3624559994557d29bd03b972fe727c89893fc2'
            or validation_context.get('driver_sha256') != '8ccb1e4797a016a5468f4965a8b4a525eac280a1414fda8565c907d783935bdd'):
        raise ValueError('Recovery does not authorize another validation profile or admission budget')
    profile_pins = {name: value.get('sha256') for name, value in request.get('profiles', {}).items()}
    if profile_pins != BASE.PROFILE_PINS:
        raise ValueError('The four original environment profiles changed')
    config = read(run/'config_snapshot.json')
    audit_path = case/'observer-recovery-v1/audit.json'
    audit = read(audit_path)
    run_status = read(run/'run_status.json')
    if run_status.get('completed') is not True or run_status.get('last_completed_slot') != BASE.TOTAL:
        raise ValueError('The same native simulation did not complete all22440slots')
    manifest = config['manifest']
    if manifest.get('rng_seed') != seed or manifest.get('total_slots') != BASE.TOTAL or manifest.get('slot_duration_ms') != BASE.SLOT_MS:
        raise ValueError('Actual recovered seed, total duration or slot duration differs')
    if audit.get('source_sha256') != config.get('source_sha256') or not config.get('source_sha256'):
        raise ValueError('Recovered recorded source identity differs')
    effective = {'traffic_profile': manifest.get('traffic_profile', {}).get('sha256'),
        'ue_services_config': config.get('ue_services_profile', {}).get('sha256'),
        'ap_services_profile': config.get('ap_services_profile', {}).get('sha256')}
    if any(effective[key] != BASE.PROFILE_PINS[key] for key in effective):
        raise ValueError('Effective recovered service/traffic profile differs')
    if audit.get('all_other_xapps_off') is not True or audit.get('profiles_verified') is not True:
        raise ValueError('Original sealed worker audit did not verify recovered actor isolation/profiles')
    window = audit.get('metrics') or {}
    if (window.get('window_start_slot') != BASE.START or window.get('slot') != BASE.END
            or window.get('window_slots') != BASE.END-BASE.START
            or not BASE.close(audit.get('observation_window_seconds'), 10.)):
        raise ValueError('Recovered canonical metrics are not the original exact10sAFTER')
    apps = config.get('configs', {}).get('ric', {}).get('xapps', [])
    if [app.get('implementation_id') for app in apps] != ['throughput_scheduler_mlp']:
        raise ValueError('Actual recovered actor differs from the operator plan')
    model_pin = file_sha(run/'throughput_mlp_model.json')
    if (model_pin != OLD.MODELS['throughput'] or audit.get('model_sha256') != model_pin
            or apps[0].get('parameters', {}).get('mode') != 'frozen'):
        raise ValueError('Recovered frozen candidate identity changed')
    values = {key: window.get('states', {}).get('kpi:'+key, {}).get('value') for key in BASE.METRICS}
    if any(value is not None and not BASE.finite(value) for value in values.values()):
        raise ValueError('Nonfinite recovered canonical metric')
    raw = BASE.raw_reading(run, actor)
    if not BASE.close(values['ue_power_w'], raw['ue_power_w']) or not BASE.close(values['network_goodput_mbps'], raw['network_goodput_mbps']):
        raise ValueError('Recovered canonical KPI differs from independent six-UE raw endpoint deltas')
    return {'case': str(case), 'run': str(run), 'request': request, 'status': status,
        'seed': seed, 'actor': actor, 'profiles_sha256': profile_pins, 'values': values,
        'raw_endpoint_verification': raw, 'model_sha256': model_pin,
        'seal_sha256': proof['supplemental_seal_sha256'], 'seal_type': 'explicit_observer_recovery_supplemental',
        'source_archive_sha256': request['source_archive_sha256'], 'recorded_source_sha256': config['source_sha256'],
        'audit_sha256': file_sha(audit_path), 'audit_path': str(audit_path), 'files_verified': proof['supplemental_files_verified'],
        'validation_lane': proof['validation_lane'], 'observer_recovery': proof}


def verify_case(path, **kwargs):
    if Path(path).resolve() == recovered_case():
        return verify_recovered_case(path, **kwargs)
    return ORIGINAL_VERIFY_CASE(path, **kwargs)


BASE.verify_case = verify_case


def validate_report_recovery(report):
    proof = recovery_evidence()
    if report.get('observer_recovery') != {'admitted_case_count': 1, 'cases': [proof]}:
        raise ValueError('Canonical report must disclose exactly the approved observer recovery')
    cases = report.get('case_evidence', [])
    admitted = [case for case in cases if case.get('observer_recovery') is not None]
    if (len(cases) != 6 or len(admitted) != 1 or admitted[0].get('case') != str(recovered_case())
            or admitted[0].get('observer_recovery') != proof
            or admitted[0].get('validation_lane', {}).get('original_worker_seal_audited') is not False
            or admitted[0].get('validation_lane', {}).get('supplementary_seal_audited') is not True
            or any(case.get('validation_lane', {}).get('original_worker_seal_audited') is not True
                   for case in cases if case is not admitted[0])):
        raise ValueError('Only one explicit recovery may replace the original complete-worker seal gate')
    return proof


def full_preflight(plan, old_value, joint, protocol_path):
    """Read stable-file hashes/lane/identity certificates, not any AFTER KPI."""
    context = BASE.validation_profile(read(old_value['selection']['path']), read(protocol_path))
    recovered = recovery_evidence()
    roots = [recovered['validation_lane']['project_root']]
    for item in joint['cases']:
        request = item['request']
        if request['id'] == RECOVERED_ID:
            continue
        case = (TASK/'runs'/request['worker_id']/request['id']).resolve()
        seal, mirror = read(case/'sealed-manifest.json'), read(case/'mirror-verification.json')
        files = seal.get('files')
        expected_schema = ('oran.throughput-study-sealed-job.v1' if request['schema'] == 'oran.throughput-improvement-job.v1'
                           else 'oran.power-study-sealed-job.v1')
        required = {'request.json', 'audit.json', 'run/config_snapshot.json', 'run/final_summary.json', 'run/run_status.json'}
        trace = [name for name in ('run/rt_sys_trace.jsonl.gz', 'run/rt_sys_trace.jsonl') if (case/name).is_file()]
        if (seal.get('failed') or seal.get('schema') != expected_schema or seal.get('id') != request['id']
                or seal.get('request_sha256') != request['request_sha256'] or not isinstance(files, dict)
                or not required.issubset(files) or len(trace) != 1 or trace[0] not in files
                or mirror.get('manifest_sha256') != _sha(case/'sealed-manifest.json')
                or mirror.get('files_verified') != len(files)):
            raise ValueError('Another final case lacks its original successful complete seal/mirror')
        if item['label'] != 'off' and 'run/'+BASE.ACTORS[item['label']][3] not in files:
            raise ValueError('An original final frozen model is outside its seal')
        for relative, entry in files.items():
            rel = Path(relative); path = case/rel
            if (rel.is_absolute() or '..' in rel.parts or path.is_symlink() or not path.resolve().is_relative_to(case)
                    or not path.is_file() or path.stat().st_size != entry.get('bytes') or _sha(path) != entry.get('sha256')):
                raise ValueError('An original final sealed file is changed or unconfined')
        roots.append(BASE.verify_validation_lane(case, context)['project_root'])
    if len(roots) != 6 or len(set(roots)) != 6:
        raise ValueError('Exactly six distinct original nested project roots are required')
    plan_base = Path(plan['_resolved_plan_path']).parent
    for entry in plan['seeds']:
        seed = entry['seed']
        for item in entry['on_cases']:
            actor = item['actor']; reference = item['prefix_certificate']
            path = BASE.resolve(reference['path'], plan_base)
            if path != Path(old_value['prefix_certificates'][str(seed)][actor]).resolve() or _sha(path) != reference['sha256']:
                raise ValueError('A prefix certificate path/byte pin differs from the original final scope')
            proof = OLD.certificate(path, 'oran.study-prefix-compatibility.v1')
            if (proof is None or proof.get('stage') != 'holdout' or proof.get('seed') != seed
                    or proof.get('selection_sha256') != OLD.SELECTION_SHA
                    or proof.get('off_run') != str((TASK/'runs'/OLD.IDS[seed, 'off'][0]/OLD.IDS[seed, 'off'][1]/'run').resolve())
                    or proof.get('candidate_run') != str((TASK/'runs'/OLD.IDS[seed, actor][0]/OLD.IDS[seed, actor][1]/'run').resolve())
                    or proof.get('off_source_archive_sha256') != OLD.SOURCE_SHA
                    or proof.get('candidate_source_archive_sha256') != OLD.SOURCE_SHA):
                raise ValueError('The prefix does not bind these exact unchanged final requests/source/selection')
            special = seed == 341721 and actor == 'throughput'
            expected = {'case_id': RECOVERED_ID,
                'recovery_certificate_sha256': recovered['recovery_certificate_sha256'],
                'supplemental_seal_sha256': recovered['supplemental_seal_sha256'],
                'original_failed_seal_sha256': ORIGINAL_FAILED_SEAL_SHA}
            if special and proof.get('observer_recovery') != expected:
                raise ValueError('Recovered native prefix must explicitly bind the preserved failure and supplemental seal')
            if not special and proof.get('observer_recovery') is not None:
                raise ValueError('Observer recovery cannot propagate to another final case')
    semantic_entries = plan.get('power_semantic_certificates', [])
    if len(semantic_entries) != 2 or {x.get('seed') for x in semantic_entries} != {341721, 341722}:
        raise ValueError('Both original final power semantic certificates remain required')
    for entry in semantic_entries:
        seed = entry['seed']; path = BASE.resolve(entry['path'], plan_base)
        proof = OLD.certificate(path, 'oran.holdout-power-action-semantics.v1')
        expected = {'seed': seed, 'run': str((TASK/'runs'/OLD.IDS[seed, 'power'][0]/OLD.IDS[seed, 'power'][1]/'run').resolve()),
            'source_archive_sha256': OLD.SOURCE_SHA, 'selection_sha256': OLD.SELECTION_SHA,
            'model_sha256': OLD.MODELS['power'], 'mode': 'frozen', 'power_action_set_version': 'legacy9',
            'actual_config_action_set_version': 'legacy9', 'baseline_guard_parameters_unchanged': True,
            'validation_version': 'v5', 'validation_protocol_sha256': context['sha256']}
        catalogue = '84033c2a8b31a33bd8760d62977ec46d81a698df20717831396cc1a8690b0dd5'
        expected.update({key: catalogue for key in ('power_action_catalogue_sha256', 'resolver_catalogue_sha256',
            'startup_catalogue_sha256', 'summary_catalogue_sha256')})
        power_case = TASK/'runs'/OLD.IDS[seed, 'power'][0]/OLD.IDS[seed, 'power'][1]
        expected['sealed_manifest_sha256'] = _sha(power_case/'sealed-manifest.json')
        if (path != Path(old_value['power_semantic_certificates'][str(seed)]).resolve()
                or _sha(path) != entry['sha256'] or proof is None
                or any(proof.get(key) != value for key, value in expected.items())):
            raise ValueError('An original power semantic certificate differs from its unchanged selected model/action/source/seal')
    return recovered


def compare_plan(plan_path, protocol_path):
    plan, old_value, joint = _plan_scope(plan_path, protocol_path)
    ready, reason = metadata_readiness(old_value, joint)
    if not ready:
        raise ValueError('No final KPI may be read before all six actual completions and certificates: '+reason)
    plan['_resolved_plan_path'] = str(Path(plan_path).resolve())
    proof = full_preflight(plan, old_value, joint, protocol_path)
    result = BASE.compare_plan(plan_path, protocol_path)
    result['observer_recovery'] = {'admitted_case_count': 1, 'cases': [proof]}
    result['canonical_report_source'] = {'base_path': str(BASE_PATH), 'base_sha256': BASE_SHA,
        'adapter_path': str(Path(__file__).resolve()), 'adapter_sha256': _sha(__file__),
        'recovery_registration': {key: RECOVERY_REGISTRATION[key] for key in ('path', 'sha256')}}
    result['canonical_report_source']['reader_amendment'] = {
        'path': RECOVERY_REGISTRATION['reader_amendment_path'], 'sha256': RECOVERY_REGISTRATION['reader_amendment_sha256']}
    result['methods']['observer_recovery'] = 'One unchanged native simulation completed after its file-stat observer failed; a separately approved supplemental seal is verified. The original failed worker status/seal are preserved.'
    result['limitations'].append('One throughput run has an explicit observer recovery; its native simulation was not restarted, and its OS exit code is unavailable because the observer was not its parent. No successful original-worker exit is claimed. A separately pinned v2 reader-only amendment accepts legal confined tar directory headers; original failed metadata, registration, supplemental seal and certificate remain unchanged.')
    validate_report_recovery(result)
    return result


def html_report(report, image):
    value = BASE.html_report(report, image)
    summary = report.get('observer_recovery', {})
    if summary.get('admitted_case_count') != 1 or len(summary.get('cases', [])) != 1:
        raise ValueError('The recovery renderer requires exactly one disclosed same-process recovery')
    proof = summary['cases'][0]
    disclosure = ('<p class="warning">One observer-recovered run: seed 341721 throughput. The original worker failed while statting '
        'a renamed live.tmp file. The same native simulation continued without restart and completed all 22,440 slots. '
        'The original failed status/seal are retained; a separate approved supplemental seal was verified. '
        'OS exit code: unknown (non-child process), not a claimed successful worker exit.</p>')
    pins = {key: proof[key] for key in ('case', 'original_failed_status_sha256', 'original_failed_seal_sha256',
        'failure_envelope_sha256', 'root_go_sha256', 'supplemental_seal_sha256', 'recovery_certificate_sha256', 'pid_continuity')}
    details = '<details><summary>Explicit observer recovery provenance</summary><pre>'+html.escape(json.dumps(pins, indent=2))+'</pre></details>'
    if 'reader_amendment' in proof:
        details += '<details><summary>Reader-only tar-directory compatibility amendment</summary><pre>'+html.escape(json.dumps(proof['reader_amendment'], indent=2))+'</pre></details>'
    end = value.index('</h1>') + len('</h1>')
    return value[:end] + disclosure + value[end:].replace('</html>', details+'</html>')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(TASK/'analysis'):
        parser.error('Use a create-only report directory inside T/analysis')
    report = compare_plan(args.plan, args.protocol)
    output.mkdir(parents=True, exist_ok=False)
    BASE.figures(report, output)
    BASE.write_new(output/'report.json', report)
    with (output/'report.html').open('x') as stream:
        stream.write(html_report(report, (output/'comparison.png').read_bytes()))
    return 0


def __getattr__(name):
    return getattr(BASE, name)


if __name__ == '__main__':
    raise SystemExit(main())
