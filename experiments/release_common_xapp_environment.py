"""Create-only export of an already validated common six-xApp environment.

No simulation, policy selection, training, default changes or KG calibration is
performed. Without --output the CLI only validates existing local evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'cbe114446814007b275856e4bae92a2fe609abf4f4bcbbf8c32394a722a36af0'
ARCHIVE = '4f915fd649e287b68465c4aed047a7b626959b471d532abd8a659e376e3641e1'
SCHEMA = 'oran.common-xapp-environment-release.v1'
APPS = {
    'A': ('ue_energy_rf', 'ue_power_w', -1, 'W'),
    'B': ('ap_energy_rf', 'ap_power_w', -1, 'W'),
    'C': ('throughput_scheduler_mlp', 'network_goodput_mbps', 1, 'Mbit/s'),
    'D': ('latency_scheduler_rf', 'mec_offloaded_latency_ms', -1, 'ms'),
    'E': ('latency_urllc_rf', 'urllc_failure_fraction', -1, 'ratio'),
    'F': ('mec_offload_budget_qgbdt', 'mec_expected_gflops', -1, 'GFLOP/s'),
}
MODEL_NAMES = ('x01', 'x02', 'x08', 'x12_mec', 'x12_urllc', 'x13')
MODEL_PATHS = tuple('artifacts/xapps/chongyu/20261117/models/'+n+'.json' for n in MODEL_NAMES)
BASE_CONFIG = 'configs/rt_sys/agx_munich_6ue_depth2_fast.json'
FROZEN_BASE_SHA256 = 'c9fd8a51a060f545cfdf56083b3fb0669edb1df3c85ba75f7d399130b5e83892'
AP_CONFIG = 'configs/rt_sys/ap_services_reference.json'
V4_RUNNER_SHA256 = '0aa03b21946b0baa600ab25e24502ef4439fe08e503304ce70aee8209b26edb2'
ROUTE_OVERRIDE = {'ue_0':{'route_index':0,'start_fraction':.65}}
GEOMETRY_CHANGE = dict(kind='UE0_INITIAL_ROAD_PHASE',ue_route_overrides=ROUTE_OVERRIDE,
    source_files_changed=False,speed_radio_controls_and_other_five_routes_unchanged=True)
RECIPE_METADATA_PINS = {'native_service_sha256':'configs/rt_sys/ue_services.json',
                        'ap_profile_sha256':AP_CONFIG}
GUARD_FLAGS = ('verified_before_activation', 'observed_prefix_equal', 'configuration_equal',
               'models_equal', 'network_dynamics_code_equal')
EXOG_FLAGS = ('before_exogenous_arrivals_equal', 'after_exogenous_arrivals_equal',
              'before_task_births_equal', 'after_task_births_equal')
RECIPE_FIELDS = ('scene', 'slots', 'baseline_slot', 'origin_observe_slots', 'observe_slots',
                 'traffic_load', 'traffic_profile', 'traffic_profile_sha256',
                 'ue_services_config', 'ue_services_config_sha256', 'ap_services_profile',
                 'ap_services_profile_sha256', 'base_config', 'base_config_sha256')


class ReleaseError(ValueError):
    """Evidence is insufficient or inconsistent; no release may be written."""


def require(condition, reason):
    if not condition:
        raise ReleaseError(reason)


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def checked(path, pin):
    path = Path(path)
    require(isinstance(pin, str) and re.fullmatch('[0-9a-f]{64}', pin), 'complete_sha256_required')
    require(path.is_file(), 'pinned_file_missing:'+str(path))
    raw = path.read_bytes()
    require(sha_bytes(raw) == pin, 'pinned_file_changed:'+str(path))
    return raw


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def integral_counter(value):
    """Accept exact JSON integer counts, including safe integer-valued floats."""
    valid_int = type(value) is int and value >= 0
    valid_float = (type(value) is float and math.isfinite(value) and 0 <= value <= 2**53-1
                   and value.is_integer())
    require(valid_int or valid_float, 'URLLC_integer_ratio_evidence_required')
    return int(value)


def sealed_json(value):
    body = deepcopy(value)
    pin = body.pop('plan_sha256', None)
    require(pin is not None and digest(body) == pin, 'protocol_or_plan_digest_mismatch')


def recipe(case):
    return {k: case.get(k) for k in RECIPE_FIELDS}


def protocol_recipes(protocol):
    """Separate the producer's exact two metadata pin keys from recipe rows."""
    values=protocol.get('recipes')
    require(isinstance(values,dict), 'protocol_recipe_map_required')
    metadata={k:v for k,v in values.items() if k in RECIPE_METADATA_PINS}
    require(not metadata or metadata.keys()==RECIPE_METADATA_PINS.keys(), 'complete_protocol_recipe_metadata_pins_required')
    require(all(isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) for v in metadata.values()),
            'complete_protocol_recipe_metadata_pins_required')
    rows={k:v for k,v in values.items() if k not in RECIPE_METADATA_PINS}
    require(all(isinstance(v,dict) for v in rows.values()), 'unknown_protocol_recipe_metadata')
    return rows,metadata


def validate_geometry_protocol(design,protocol,plans):
    """Admit the one reviewed external base; never arbitrary config overrides."""
    version=design.get('runner_version',2)
    if version != 4:
        require(not protocol.get('geometry_change'), 'geometry_requires_reviewed_v4_protocol')
        for plan in plans:
            root=Path(plan.get('source_root','/scope/source'))
            for case in plan['cases']:
                if case.get('base_config') is not None:
                    require(Path(case['base_config'])==root/BASE_CONFIG
                            and case.get('base_config_sha256')==plan.get('frozen_physical_files',{}).get(BASE_CONFIG),
                            'external_base_requires_reviewed_v4_protocol')
        return
    require(protocol.get('runner_version')==4
            and protocol.get('runner_sha256')==V4_RUNNER_SHA256
            and digest(protocol.get('geometry_change'))==digest(GEOMETRY_CHANGE)
            and protocol.get('require_on_admission') is True
            and protocol.get('ON_admission_marker')=='ON_ADMISSION_ALLOWED',
            'reviewed_v4_geometry_and_ON_admission_required')
    require(design.get('include_native_reference') is False, 'geometry_is_not_current_native_reference')
    for plan in plans:
        root=Path(plan['source_root'])
        expected=root.parent/'protocol/profiles'/(design['study_id']+'_ue0_road0_phase065_base.json')
        require(plan.get('schema')=='oran.environment-validation-worker-plan.v4'
                and plan.get('require_on_admission') is True
                and digest(plan.get('geometry_change'))==digest(GEOMETRY_CHANGE)
                and Path(plan.get('runner_path',''))==root.parent/'runner/worker-v4.py',
                'reviewed_v4_worker_geometry_required')
        for case in plan['cases']:
            require(Path(case.get('base_config',''))==expected
                    and isinstance(case.get('base_config_sha256'),str)
                    and re.fullmatch('[0-9a-f]{64}',case['base_config_sha256']),
                    'owned_pinned_v4_base_required')


def validate_geometry_base(raw,frozen_raw):
    """Canonical JSON equality rejects hidden extra fields and bool/number swaps."""
    require(sha_bytes(frozen_raw)==FROZEN_BASE_SHA256, 'exact_CBE_frozen_base_required')
    frozen=json.loads(frozen_raw)
    require('ue_route_overrides' not in frozen, 'frozen_base_already_has_routes')
    expected={**frozen,'ue_route_overrides':deepcopy(ROUTE_OVERRIDE)}
    require(digest(json.loads(raw))==digest(expected), 'only_UE0_road0_phase065_base_change_allowed')
    return {'schema':'oran.common-environment-base-difference.v1',
            'frozen_base_config':BASE_CONFIG,'frozen_base_sha256':sha_bytes(frozen_raw),
            'external_base_sha256':sha_bytes(raw),'geometry_change':deepcopy(GEOMETRY_CHANGE),
            'exact_semantic_delta':[{'path':'/ue_route_overrides','before':None,'after':deepcopy(ROUTE_OVERRIDE)}]}


def clear(value):
    return value == 0 or isinstance(value, (list, dict)) and not value


def positive_pair(pair, on, off):
    arm = pair['arm']; app, key, direction, _ = APPS[arm]
    require(pair.get('app_id') == app and pair.get('target_kpi') == key, 'wrong_app_target')
    require(pair.get('before') == on['windows']['before'] and pair.get('after') == on['windows']['after']
            and pair.get('off_after') == off['windows']['after'], 'pair_window_payload_mismatch')
    require(pair.get('before_comparison', {}).get('comparable_under_origin_protocol') is True,
            'same_before_proof_required')
    require(all(pair.get('external_input_comparison', {}).get(k) is True for k in EXOG_FLAGS),
            'paired_external_input_proof_required')
    for window in ('before', 'after'):
        for key_hash in ('exogenous_arrivals_sha256', 'task_births_sha256'):
            a = on['flows'][window].get(key_hash); b = off['flows'][window].get(key_hash)
            require(isinstance(a, str) and re.fullmatch('[0-9a-f]{64}', a) and a == b,
                    'paired_external_input_hash_mismatch')
    a = pair['after']['states']['kpi:'+key]; b = pair['off_after']['states']['kpi:'+key]
    require(finite(a.get('value')) and finite(b.get('value')), 'finite_observed_target_required')
    effect = pair.get('target', {}); improvement = direction*(a['value']-b['value'])
    tolerance = effect.get('numerical_tolerance')
    require(finite(tolerance) and tolerance >= 0, 'numeric_target_resolution_required')
    if arm == 'E':
        counters=[]
        for state in (a, b):
            counts={k:integral_counter(state.get(k)) for k in ('failure_events','offered_packets')}
            require(counts['offered_packets'] > 0, 'URLLC_integer_ratio_evidence_required')
            require(math.isclose(state['value'], counts['failure_events']/counts['offered_packets'],
                                 rel_tol=1e-12, abs_tol=1e-15), 'URLLC_ratio_disagrees_with_counts')
            counters.append(counts)
        ac,bc=counters
        require(ac['failure_events']*bc['offered_packets'] < bc['failure_events']*ac['offered_packets'],
                'zero_or_negative_target_effect:E')
    else:
        if arm == 'C':
            resolutions = [s.get('resolution') for s in (a,b) if finite(s.get('resolution')) and s['resolution'] > 0]
            if resolutions:
                tolerance = max(tolerance,max(resolutions)*.5)
        require(improvement > max(tolerance, 1e-9, 1e-10*max(abs(a['value']), abs(b['value']))),
                'zero_or_negative_target_effect:'+arm)
    require(effect.get('status') == 'improved' and finite(effect.get('target_improvement'))
            and math.isclose(effect['target_improvement'], improvement, rel_tol=1e-9, abs_tol=1e-12),
            'target_effect_payload_mismatch')
    control = pair.get('control_provenance', {}); evidence = control.get('evidence', {})
    require(control == on.get('control_provenance') and control.get('qualified') is True,
            'qualified_real_control_required')
    require(evidence.get('status') == 'ready' and all(evidence.get(k) is True for k in
            ('activation_verified', 'frozen_verified', 'consumer_verified', 'actual_changed')),
            'executed_changed_control_required')
    require(all(clear(evidence.get(k)) for k in
            ('consumer_fields_unresolved', 'consumer_mismatches', 'trace_consumer_mismatches')),
            'unresolved_control_evidence')
    require(all(type(evidence.get(k)) is int and evidence[k] > 0 for k in
            ('own_applied_receipt_count','changed_receipt_count','decision_count'))
            and evidence.get('before_baseline_binding_slots') == [] and evidence.get('unexpected_active_apps') == [],
            'own_executed_control_counts_required')
    for k, expected in (('job_id', pair['job_id']), ('checkpoint_sha256', on['checkpoint']['sha256']),
                        ('source_sha256', on['source_sha256']), ('runner_sha256', on['runner_sha256']),
                        ('source_archive_sha256', on['source_archive_sha256']), ('worker_id', on['worker_id'])):
        require(evidence.get(k) == expected, 'control_identity_mismatch:'+k)
    if arm == 'D':
        require(all(type(s.get('sample_count')) is int and s['sample_count'] > 0 for s in (a, b)),
                'observed_MEC_completions_required')
    if arm == 'F':
        quota = pair.get('post_first_apply_ue4_quota') or {}
        require(quota == on.get('post_first_apply_ue4_quota') and quota.get('status') == 'observed'
                and quota.get('target_ue_id') == 'ue_4'
                and quota.get('start_basis') == 'first_actual_own_applied_consumer_action_even_if_noop',
                'F_post_first_ACK_quota_required')
        first = quota.get('first_applied_slot'); end = on['case']['baseline_slot']+on['case']['observe_slots']
        require(type(first) is int and on['case']['baseline_slot'] < first <= end
                and first == evidence.get('first_applied_receipt_slot')
                and quota.get('after_end_slot') == end
                and quota.get('post_apply_exposure_slots') == end-first+1, 'F_exact_post_ACK_window_required')
        require(all(quota.get(k) is True for k in ('mean_compliant', 'maximum_compliant'))
                and all(finite(quota.get(k)) for k in ('mean_gflops', 'maximum_gflops', 'budget_gflops'))
                and 0 <= quota['mean_gflops'] <= quota['maximum_gflops'] <= quota['budget_gflops'],
                'F_post_first_ACK_quota_failed')


def validate_report(report, plans, design, protocol):
    """Pure structural gate. Actual file/trace seals are verified separately."""
    require(report.get('schema') == 'oran.environment-validation-analysis.v1', 'full_analysis_schema_required')
    require(report.get('analysis_role') == 'validation', 'final_validation_role_required')
    require('complete_cases' in report and not report.get('pending_cases')
            and not report.get('intentionally_not_run_cases') and not report.get('environment_exclusions'),
            'complete_unexcluded_report_required')
    require(report.get('comparison') == 'single_app_after_minus_same_origin_off_after'
            and report.get('training_performed') is False and report.get('new_simulations') == 0,
            'matched_frozen_evaluation_required')
    require(report.get('recording_admission', {}).get('receiver_seals_required') is True
            and report['recording_admission'].get('all_published_raw_sha256_required') is True,
            'receiver_and_all_raw_seals_required')
    require(all(report.get(k) is True for k in ('same_environment_two_seed_validation_passed',
            'all_six_targets_numerically_positive', 'all_six_controls_verified_and_numerically_positive')),
            'all_six_positive_real_control_validation_required')
    sealed_json(protocol)
    require(design.get('stage') == protocol.get('stage') == 'validation'
            and protocol.get('design_sha256') == digest(design), 'validation_design_pin_required')
    require(all(type(protocol.get(k)) is int and protocol[k] > 0 for k in
            ('baseline_slot','origin_observe_slots','observe_slots','total_slots'))
            and type(protocol.get('tail_slots')) is int and protocol['tail_slots'] >= 0
            and protocol['baseline_slot'] == ((protocol['origin_observe_slots']+40)//40)*40
            and protocol['total_slots'] == protocol['baseline_slot']+protocol['observe_slots']+protocol['tail_slots'],
            'exact_observed_window_protocol_required')
    require(set(design.get('arms', [])) == {'OFF', *APPS} and len(design['arms']) == 7,
            'exactly_six_singleton_arms_required')
    recipes,metadata_pins=protocol_recipes(protocol)
    require(len(design.get('recipes', [])) == 1 and len(recipes) == 1,
            'one_common_recipe_required')
    scene = design['recipes'][0]['id']
    require(set(recipes) == {scene}, 'recipe_identity_mismatch')
    assignments = design.get('assignments', []); seeds = [a.get('seed') for a in assignments]
    require(len(seeds) >= 2 and all(type(s) is int for s in seeds) and len(set(seeds)) == len(seeds),
            'two_distinct_fresh_seeds_required')
    history = set(protocol.get('known_nonfresh_seeds', [])) | set(design.get('discovery_seeds', []))
    require(bool(design.get('discovery_seeds')) and bool(protocol.get('known_nonfresh_seeds'))
            and protocol.get('discovery_seeds') == design['discovery_seeds'] and not set(seeds) & history,
            'declared_fresh_seed_history_required')
    require(all(a.get('recipe_ids') == [scene] for a in assignments), 'cross_recipe_assignment')
    planned = {}; model_pins = protocol.get('frozen_models', {}); runners = set()
    require(set(model_pins) == set(MODEL_PATHS), 'exact_frozen_six_models_required')
    workers = [p.get('worker_id') for p in plans]
    require(len(set(workers)) == len(workers) and len(plans) >= 2, 'distinct_worker_plans_required')
    for plan in plans:
        sealed_json(plan); runners.add(plan.get('runner_sha256'))
        require(plan.get('stage') == 'validation' and plan.get('study_id') == design.get('study_id')
                and plan.get('design_sha256') == digest(design), 'same_validation_plan_design_required')
        require(plan.get('source_sha256') == protocol.get('source_sha256') == SOURCE
                and plan.get('source_archive_sha256') == protocol.get('source_archive_sha256') == ARCHIVE
                and plan.get('frozen_models') == model_pins, 'source_archive_or_models_mismatch')
        for name,pin in metadata_pins.items():
            require(plan.get('frozen_physical_files',{}).get(RECIPE_METADATA_PINS[name])==pin,
                    'protocol_recipe_metadata_pin_mismatch:'+name)
        for case in plan.get('cases', []):
            require(case.get('id') not in planned and case.get('worker_id') == plan['worker_id'], 'duplicate_or_misowned_case')
            require((case['worker_id'], case['seed']) in {(a['worker_id'], a['seed']) for a in assignments},
                    'case_not_in_validation_assignment')
            require(case.get('observe_slots') == protocol.get('observe_slots')
                    and case.get('baseline_slot') == protocol.get('baseline_slot')
                    and case.get('origin_observe_slots') == protocol.get('origin_observe_slots')
                    and case.get('slots') == protocol.get('total_slots'), 'case_observation_window_mismatch')
            planned[case['id']] = case
    require(runners == {protocol.get('runner_sha256')} == {design.get('runner_sha256')}, 'runner_pin_mismatch')
    validate_geometry_protocol(design,protocol,plans)
    require(protocol.get('physical_run_count') == len(planned), 'planned_case_count_mismatch')
    complete = report['complete_cases']; rows = {r['case']['id']: r for r in complete}
    require(len(rows) == len(complete) == len(planned) and set(rows) == set(planned), 'partial_or_duplicate_case_report')
    origins = {}; common = []; references = set()
    for ident, row in rows.items():
        case = row['case']; require(case == planned[ident], 'reported_case_differs_from_plan')
        require(row.get('status') == 'complete' and row.get('stage') == 'validation'
                and row.get('worker_id') == case['worker_id']
                and row.get('source_sha256') == SOURCE and row.get('source_archive_sha256') == ARCHIVE
                and row.get('frozen_models') == model_pins and row.get('runner_sha256') in runners,
                'complete_case_provenance_mismatch')
        if case['scene'] != scene:
            require(case.get('recipe_kind') == 'native_reference' and case.get('arm') == 'OFF'
                    and case.get('app_ids') == [] and design.get('include_native_reference') is True,
                    'cross_recipe_case')
            reference = (case['worker_id'],case['seed'])
            require(reference not in references, 'duplicate_native_reference')
            references.add(reference); continue
        common.append(row); group = (case['worker_id'], case['seed'], case['origin_key'])
        if case['arm'] == 'OFF':
            require(not case.get('app_ids') and group not in origins, 'one_own_OFF_per_seed_required')
            origins[group] = row
        else:
            require(case['arm'] in APPS and case.get('app_ids') == ['xapp:'+APPS[case['arm']][0]],
                    'singleton_app_mismatch')
    require(references == ({(a['worker_id'],a['seed']) for a in assignments}
            if design.get('include_native_reference') else set()), 'native_reference_count_mismatch')
    require(len(common) == 7*len(seeds) and len(origins) == len(seeds), 'all_six_and_own_OFF_each_seed_required')
    common_recipe = {digest(recipe(r['case'])) for r in common}
    require(len(common_recipe) == 1, 'cross_recipe_physical_inputs')
    recipe_pin = recipes[scene]
    for k in ('traffic_load','traffic_profile','traffic_profile_sha256','ue_services_config','ue_services_config_sha256'):
        require(recipe_pin.get(k) == common[0]['case'].get(k), 'protocol_recipe_pin_mismatch:'+k)
    if design.get('runner_version')==4:
        for k in ('base_config','base_config_sha256'):
            require(recipe_pin.get(k)==common[0]['case'].get(k), 'protocol_recipe_pin_mismatch:'+k)
    pairs = report.get('paired_results', []); seen = set(); results = []
    for pair in pairs:
        arm = pair.get('arm'); seed = pair.get('seed'); key = (seed, arm)
        require(arm in APPS and seed in seeds and key not in seen and pair.get('stage') == 'validation'
                and pair.get('scene') == scene, 'duplicate_or_cross_recipe_pair')
        seen.add(key); on = rows.get(pair.get('job_id')); off = rows.get(pair.get('off_job_id'))
        require(on is not None and off is not None and on['case']['arm'] == arm
                and on['case']['seed'] == seed, 'pair_job_identity_mismatch')
        case = on['case']; group = (case['worker_id'], case['seed'], case['origin_key'])
        require(origins.get(group) is off and pair.get('worker_id') == case['worker_id']
                and pair.get('origin_key') == case['origin_key'], 'pair_requires_own_same_host_OFF')
        require(on.get('origin_id') and on['origin_id'] == off.get('origin_id')
                and isinstance(on.get('actual_system_hostname'),str) and bool(on['actual_system_hostname'])
                and on.get('actual_system_hostname') == off.get('actual_system_hostname')
                and on['windows']['before'] == off['windows']['before'], 'same_native_origin_required')
        require(all(on.get('origin_guard', {}).get(k) is True for k in GUARD_FLAGS)
                and on['origin_guard'].get('origin_id') == on['origin_id'], 'native_origin_guard_required')
        positive_pair(pair, on, off)
        results.append({'arm':arm, 'app_id':APPS[arm][0], 'target_kpi':APPS[arm][1], 'unit':APPS[arm][3],
                        'seed':seed, 'off':pair['off_after']['states']['kpi:'+APPS[arm][1]]['value'],
                        'on':pair['after']['states']['kpi:'+APPS[arm][1]]['value'], 'effect':pair['target'],
                        'job_id':pair['job_id'], 'off_job_id':pair['off_job_id']})
    require(seen == {(s,a) for s in seeds for a in APPS}, 'complete_six_app_seed_matrix_required')
    summary = report.get('singleton_summary', [])
    require(len(summary) == 6 and {s.get('arm') for s in summary} == set(APPS), 'six_app_summary_required')
    for s in summary:
        require(s.get('scene') == scene and s.get('planned_pairs') == s.get('paired') == len(seeds)
                and s.get('all_planned_pairs_numerically_positive') is True
                and s.get('control_provenance_qualified_all_pairs') is True
                and not s.get('missing_job_ids') and not s.get('intentionally_not_run_job_ids')
                and {p.get('seed') for p in s.get('per_seed', [])} == set(seeds), 'summary_matrix_mismatch')
        if s['arm'] == 'F':
            require(s.get('released_ue4_quota_satisfied_all_pairs') is True, 'F_summary_quota_required')
    assessments = report.get('recipe_assessments', [])
    require(len(assessments) == 1 and assessments[0].get('scene') == scene
            and assessments[0].get('same_environment_independent_two_seed_validation_passed') is True
            and set(assessments[0].get('common_complete_seeds_across_all_apps', [])) == set(seeds),
            'final_common_recipe_assessment_required')
    return {'scene':scene, 'seeds':sorted(seeds), 'recipe':recipe(common[0]['case']),
            'source_sha256':SOURCE, 'source_archive_sha256':ARCHIVE,
            'runner_sha256':next(iter(runners)), 'frozen_models':model_pins,
            'paired_targets':sorted(results, key=lambda x:(x['arm'],x['seed']))}


def validate_criteria(criteria,gate,design,protocol):
    """Bind the already declared acceptance rules; never alter KPI effects."""
    require(criteria.get('schema')=='oran.common-xapp-validation-criteria.v1', 'declared_criteria_schema_required')
    require(criteria.get('common_recipe')==design['recipes'][0]
            and criteria.get('required_apps')==list(APPS)
            and sorted(criteria.get('fresh_seeds',[]))==gate['seeds'], 'criteria_recipe_apps_seeds_mismatch')
    for key in ('all_six_targets_positive_in_each_seed','real_frozen_changed_controls_required',
                'same_own_OFF_origin_and_before_after_exogenous_hashes_required'):
        require(criteria.get(key) is True, 'criteria_required_rule_missing:'+key)
    require(criteria.get('training_or_manual_actions') is False and criteria.get('default_or_KG_changed') is False,
            'criteria_frozen_nonintervention_required')
    window=criteria.get('measurement',{})
    require(all(type(window.get(k)) is int and window[k]==protocol[k] for k in
            ('baseline_slot','origin_observe_slots','observe_slots','tail_slots','total_slots'))
            and finite(window.get('slot_ms')) and window['slot_ms']==.5, 'criteria_measurement_window_mismatch')
    minimum=criteria.get('C_minimum_improvement_percent_each_seed')
    require(finite(minimum) and minimum==1., 'predeclared_C_one_percent_rule_required')
    values={r['seed']:r['effect'].get('improvement_percent') for r in gate['paired_targets'] if r['arm']=='C'}
    require(set(values)==set(gate['seeds']) and all(finite(v) and v>=minimum for v in values.values()),
            'predeclared_C_minimum_not_met_each_seed')
    quota=criteria.get('F_quota',{}).get('ue4_gflops_maximum')
    require(finite(quota) and quota==.75, 'predeclared_F_quota_rule_required')
    require(isinstance(criteria.get('D_cohort_diagnostic'),str) and bool(criteria['D_cohort_diagnostic'].strip()),
            'predeclared_D_identity_diagnostic_required')
    if design.get('runner_version')==4:
        geometry=criteria.get('geometry_change',{})
        require(digest(geometry.get('ue_route_overrides'))==digest(ROUTE_OVERRIDE)
                and geometry.get('other_five_UEs_unchanged') is True
                and finite(geometry.get('all_UE_speed_mps')) and geometry['all_UE_speed_mps']==1.5
                and criteria.get('OFF_admission_gate',{}).get('explicit_ON_ADMISSION_ALLOWED_required') is True,
                'predeclared_v4_route_and_admission_required')
    return {'C_minimum_improvement_percent_each_seed':minimum,'C_observed_improvement_percent_by_seed':values,
            'F_ue4_gflops_maximum':quota,'D_identity_diagnostic_required':True,
            'canonical_metric_or_threshold_changed':False}


def bind_D_diagnostics(refs,report,results_path,gate):
    """Verify the independent frozen reader's identity reports, not a new KPI."""
    require(len(refs)==len(gate['seeds']), 'one_pinned_D_identity_diagnostic_per_seed_required')
    by_id={r['case']['id']:r for r in report['complete_cases']}
    pairs={p['seed']:p for p in report['paired_results'] if p['arm']=='D'}
    result=[];files={};seen=set();reader_sha=None
    for path,pin in refs:
        raw=checked(path,pin);cert=json.loads(raw);seed=cert.get('seed')
        require(cert.get('schema')=='oran.environment-MEC-completed-cohort-diagnostic.v2'
                and seed in gate['seeds'] and seed not in seen and cert.get('scene')==gate['scene'],
                'D_identity_schema_seed_or_recipe_mismatch')
        seen.add(seed);pair=pairs[seed]
        require(cert.get('worker_id')==pair['worker_id']
                and cert.get('source_results',{}).get('sha256')==file_sha(results_path)
                and Path(cert['source_results']['path']).resolve()==Path(results_path).resolve(),
                'D_identity_final_results_or_worker_mismatch')
        require(cert.get('exact_primary_birth_exposure_used') is True
                and cert.get('terminal_followup_beyond_primary_end_used') is False
                and cert.get('primary_metric_or_analysis_unchanged') is True
                and cert.get('canonical_target')==pair['target'], 'D_identity_primary_metric_changed')
        for label,job_id in (('OFF',pair['off_job_id']),('D',pair['job_id'])):
            row=by_id[job_id];run=Path(row['run_path']).resolve()
            traces=[name for name in ('data/run/rt_sys_trace.jsonl','data/run/rt_sys_trace.jsonl.gz') if name in row['files_sha256']]
            require(len(traces)==1, 'D_identity_one_trace_required')
            expected={'checkpoint':row['checkpoint'],
                'trace':{'path':str(run/Path(traces[0]).name),'sha256':row['files_sha256'][traces[0]]},
                'config':{'path':str(run/'config_snapshot.json'),'sha256':row['files_sha256']['data/run/config_snapshot.json']}}
            require(cert.get('evidence',{}).get(label)==expected, 'D_identity_pair_recording_mismatch:'+label)
        for key in ('eligible_task_keys_equal','eligible_task_clock_route_equal','completed_task_keys_equal','failed_task_keys_equal'):
            require(type(cert.get(key)) is bool, 'D_identity_membership_diagnostic_missing:'+key)
        for key in ('OFF_only_completed','D_only_completed','OFF_pending','D_pending'):
            require(isinstance(cert.get(key),list), 'D_identity_task_list_missing:'+key)
        require(all(isinstance(cert.get(k),dict) for k in ('OFF_failed','D_failed')), 'D_identity_failed_task_sets_required')
        count=cert.get('common_completed_count')
        require(type(count) is int and count>=0, 'D_identity_common_count_required')
        for label,window in (('OFF','off_after'),('D','after')):
            require(cert.get(label+'_completed_count')==pair[window]['states']['kpi:mec_offloaded_latency_ms']['sample_count'],
                    'D_identity_completed_count_mismatch')
        for key in ('common_completed_OFF_mean_ms','common_completed_D_mean_ms','common_completed_mean_change_ms'):
            require((finite(cert.get(key)) if count else cert.get(key) is None), 'D_identity_common_task_diagnostic_missing')
        code=cert.get('code',{});reader=checked(code['path'],code.get('sha256'))
        require(reader_sha is None or reader_sha==code['sha256'], 'D_identity_reader_versions_differ')
        reader_sha=code['sha256'];files['evidence/D-diagnostics/reader.py']=reader
        files['evidence/D-diagnostics/seed-'+str(seed)+'.json']=raw
        result.append({'seed':seed,'scene':gate['scene'],'worker_id':pair['worker_id'],
            'job_id':pair['job_id'],'off_job_id':pair['off_job_id'],'path':str(Path(path).resolve()),
            'sha256':pin,'source_results_sha256':cert['source_results']['sha256'],'reader_sha256':reader_sha,
            'common_completed_count':count,'completed_task_keys_equal':cert['completed_task_keys_equal'],
            'interpretation':'Identity/composition diagnostic only; unequal cohorts or common-task delay do not replace the canonical target or introduce a new success threshold.'})
    return {'complete':True,'refs':sorted(result,key=lambda r:r['seed'])},files


def archive_files(path):
    require(file_sha(path) == ARCHIVE, 'tested_source_archive_required')
    files = {}
    with tarfile.open(path, 'r:*') as archive:
        for item in archive:
            if not item.isfile():
                continue
            name = item.name.removeprefix('./')
            require(not Path(name).is_absolute() and '..' not in Path(name).parts and name not in files,
                    'unsafe_or_duplicate_archive_member')
            files[name] = archive.extractfile(item).read()
    h = hashlib.sha256()
    for group in ('core','phy','ran','ric','state','models','traffic','experiments','configs'):
        for name in sorted(n for n in files if Path(n).parts[0] == group and Path(n).suffix in ('.py','.json','.yaml','.yml')):
            h.update(name.encode()); h.update(files[name])
    require(h.hexdigest() == SOURCE, 'tested_source_fingerprint_mismatch')
    return files


def verify_recording(row, plan):
    seal = json.loads(checked(row['checkpoint']['path'], row['checkpoint']['sha256']))
    require(seal.get('completed') is True and seal.get('case') == row['case']
            and seal.get('plan_sha256') == plan['plan_sha256'] and seal.get('models_unchanged') is True
            and seal.get('training_performed') is False
            and seal.get('post_deployment_kpis_used_for_model_training') is False, 'frozen_case_seal_required')
    for k in ('source_sha256','source_archive_sha256','runner_sha256','frozen_models','origin_id','actual_system_hostname'):
        require(seal.get(k) == row.get(k), 'recording_seal_identity_mismatch:'+k)
    require(seal.get('files_sha256') == row.get('files_sha256'), 'recording_file_set_mismatch')
    run = Path(row['run_path']).resolve(); job = run.parents[1]
    require(run == job/'data/run', 'recording_layout_mismatch')
    required = {'data/run/config_snapshot.json','data/run/run_status.json','data/timeline.json'}
    if row['case'].get('app_ids'):
        required |= {'data/expected_origin.json','data/run/evaluation_origin_verified.json'}
    require(required <= seal['files_sha256'].keys(), 'required_recording_proofs_must_be_sealed')
    require(sum(n in seal['files_sha256'] for n in ('data/run/rt_sys_trace.jsonl','data/run/rt_sys_trace.jsonl.gz')) == 1,
            'one_sealed_trace_required')
    for name,pin in seal['files_sha256'].items():
        path = (job/name).resolve()
        require(path.is_relative_to(job) and '..' not in Path(name).parts, 'unsafe_recording_file')
        checked(path, pin)
    receiver = row.get('receiver_admission') or {}
    require(receiver.get('required') is True and receiver.get('checkpoint_sha256') == row['checkpoint']['sha256']
            and receiver.get('verification', {}).get('seal_sha256') == row['checkpoint']['sha256'],
            'receiver_publication_proof_required')
    config = read(run/'config_snapshot.json')
    require(config.get('source_sha256') == SOURCE and config['manifest']['rng_seed'] == row['case']['seed']
            and config['manifest']['total_slots'] == row['case']['slots']
            and read(run/'run_status.json').get('completed') is True, 'completed_recorded_configuration_required')
    if plan.get('schema')=='oran.environment-validation-worker-plan.v4':
        extra={'data/request.json','data/command.json'}
        require(extra<=seal['files_sha256'].keys(), 'sealed_v4_configuration_request_required')
        require(seal.get('schema')=='oran.environment-validation-run.v4'
                and seal.get('base_config_sha256')==row['case']['base_config_sha256']
                and digest(seal.get('geometry_change'))==digest(GEOMETRY_CHANGE)
                and seal.get('require_on_admission') is True and seal.get('scheduling_only_change') is False,
                'sealed_v4_geometry_identity_required')
        request=read(job/'data/request.json');argv=read(job/'data/command.json').get('argv',[])
        require(request.get('schema')=='oran.environment-run-request.v4' and request.get('case')==row['case']
                and request.get('plan_sha256')==plan['plan_sha256']
                and request.get('base_config_sha256')==row['case']['base_config_sha256']
                and digest(request.get('geometry_change'))==digest(GEOMETRY_CHANGE)
                and request.get('require_on_admission') is True and request.get('scheduling_only_change') is False
                and argv[-2:]==['--config',row['case']['base_config']] and argv.count('--config')==2
                and digest(config['manifest'].get('ue_route_overrides'))==digest(ROUTE_OVERRIDE),
                'recorded_v4_base_request_command_or_route_mismatch')
    for field,key in (('traffic_profile','traffic_profile'),('ue_services_config','ue_services_profile'),
                      ('ap_services_profile','ap_services_profile')):
        if row['case'].get(field) is not None:
            meta = config['manifest'].get(key,{}) if field == 'traffic_profile' else config.get(key,{})
            require(meta.get('sha256') == row['case'][field+'_sha256'], 'recorded_recipe_hash_mismatch:'+field)
    if row['case'].get('app_ids'):
        require(read(run/'evaluation_origin_verified.json') == row['origin_guard'], 'recorded_origin_guard_mismatch')
        proof = read(job/'data/expected_origin.json'); body = deepcopy(proof); origin = body.pop('origin_id', None)
        require(origin == digest(body) == row['origin_id'] and proof.get('seed') == row['case']['seed']
                and proof.get('schema') == 'oran.evaluation-origin.v1'
                and proof.get('source_sha256') == SOURCE and proof.get('source_archive_sha256') == ARCHIVE
                and proof.get('baseline_slot') == row['case']['baseline_slot']
                and proof.get('observe_slots') == row['case']['origin_observe_slots']
                and proof.get('before_frame',{}).get('slot') == row['case']['baseline_slot']
                and proof.get('after_kpis_used') is False, 'sealed_native_origin_proof_required')
    return config


def verify_F_quota(row,config):
    """Recheck the admitted post-first-own-ACK quota against actual trace policy."""
    q = row['post_first_apply_ue4_quota']; first = q['first_applied_slot']; end = q['after_end_slot']
    profile = config['configs']['phy']['rt_sys']['ue_services']['mec']['tasks_by_ue']['ue_4']
    apps = [a for a in config['configs']['ric']['xapps'] if a.get('implementation_id') == APPS['F'][0]]
    require(len(apps) == 1 and apps[0].get('parameters',{}).get('budget_gflops') == q['budget_gflops'],
            'F_released_quota_config_mismatch')
    require(all(finite(profile.get(k)) and profile[k] > 0 for k in ('interval_ms','compute_gflop')),
            'F_task_work_profile_required')
    candidates = [Path(row['run_path'])/name for name in ('rt_sys_trace.jsonl','rt_sys_trace.jsonl.gz')]
    traces = [p for p in candidates if p.is_file()]
    require(len(traces) == 1, 'one_sealed_trace_required')
    values = []; expected = first; trace = traces[0]
    opener = gzip.open if trace.suffix == '.gz' else open
    with opener(trace,'rt') as stream:
        for line in stream:
            frame = json.loads(line); slot = frame['slot_id']
            if slot < first:
                continue
            if slot > end:
                break
            require(slot == expected, 'F_post_ACK_trace_gap'); expected += 1
            fraction = frame.get('ue_services',{}).get('ue_4',{}).get('ue_service_policy',{}).get('offload_fraction')
            require(finite(fraction) and 0 <= fraction <= 1, 'F_actual_UE4_policy_required')
            values.append(1000/profile['interval_ms']*profile['compute_gflop']*fraction)
    require(len(values) == end-first+1 and all(v <= q['budget_gflops'] for v in values), 'F_actual_post_ACK_quota_failed')
    require(math.isclose(sum(values)/len(values),q['mean_gflops'],rel_tol=1e-10,abs_tol=1e-12)
            and max(values) == q['maximum_gflops'], 'F_reported_quota_does_not_match_actual_trace')


def reproduction_recipe(gate,protocol):
    """An argument recipe for the existing launcher, not a new runtime."""
    ident=digest({'recipe':gate['recipe'],'source':SOURCE,'models':gate['frozen_models']})[:16]
    output='outputs/common_xapp_replay/'+ident+'/{arm}/seed-{seed}'
    argv=['bash','scripts/orin/run-six-ue-cuda-nr.sh','--model','fast','--fast-profile','system',
          '--slots',str(protocol['total_slots']),'--seed','{seed}',
          '--config','{environment_dir}/configs/base.json','--traffic-load',str(gate['recipe']['traffic_load']),
          '--traffic-profile','{environment_dir}/configs/traffic.json','--ue-services','drx-mec',
          '--ue-services-config','{environment_dir}/configs/ue-services.json',
          '--ap-services-profile','{environment_dir}/configs/ap-services.json','--xapp-pool','chongyu',
          '--live-xapps','--xapps-start','','--xapp-toggle-timeline','{environment_dir}/timelines/{arm}.json',
          '--record-compression','gzip','--xapp-decision-preview','--xapp-preview-before-slot',str(protocol['baseline_slot']),
          '--replay-max-frames','100','--output',output,'--live-state',output+'/live.json',
          '--channel-cache-root','channel_cache/common_xapp_replay/'+ident+'/{arm}/seed-{seed}']
    return {'schema':'oran.common-environment-reproduction-arguments.v1',
        'cwd':'Root of the tested CBE source copy, with its existing six-UE launcher and frozen installed model release.',
        'existing_launcher':'scripts/orin/run-six-ue-cuda-nr.sh','argv_template':argv,
        'choices':{'arm':['OFF',*APPS],'seed':'User-selected uint32; the validated seeds are listed separately.',
                   'environment_dir':'Absolute path to this exported artifact.'},
        'validated_seeds':gate['seeds'],'slot_duration_ms':.5,
        'window_slots':{k:protocol[k]for k in ('baseline_slot','origin_observe_slots','observe_slots','tail_slots','total_slots')},
        'before_interval':{'left_exclusive':protocol['baseline_slot']-protocol['origin_observe_slots'],
                           'right_inclusive':protocol['baseline_slot']},
        'after_interval':{'left_exclusive':protocol['baseline_slot'],
                          'right_inclusive':protocol['baseline_slot']+protocol['observe_slots']},
        'control_origin':'All xApps initially OFF; replay the copied exact timeline after the native warm-up. Starting an app at slot0 is a different, unvalidated experiment.',
        'comparison':'Run each arm independently with the same chosen seed, physical host, source, models and recipe; compare ON AFTER with its own OFF AFTER. A new seed/run is not automatically validated.',
        'origin_guard':'Prepared origin proofs in evidence certify the recorded validation only; do not reuse another host/seed proof for a new run.',
        'no_executable_wrapper_created':True,'existing_runtime_output_guards_still_apply':True}


def reproduction_arguments(reproduction,environment_dir,arm,seed):
    """Render arguments only, restricting arm/seed so output paths cannot escape."""
    require(arm in ('OFF',*APPS), 'choose_one_copied_timeline')
    require(type(seed) is int and 0 <= seed < 2**32, 'reproduction_seed_must_be_uint32')
    values={'environment_dir':str(Path(environment_dir).resolve()),'arm':arm,'seed':str(seed)}
    argv=[x.format(**values)for x in reproduction['argv_template']]
    for flag in ('--output','--live-state'):
        path=Path(argv[argv.index(flag)+1])
        require(not path.is_absolute() and path.is_relative_to('outputs') and '..' not in path.parts
                and path != Path('outputs'), 'reproduction_output_must_stay_under_project_outputs')
    return argv


def sealed_timelines(rows,scene):
    """Preserve identical recorded timeline bytes for each independent arm."""
    timelines={}
    for row in rows:
        case=row['case']
        if case['scene'] != scene:
            continue
        arm=case['arm'];job=Path(row['run_path']).resolve().parents[1]
        raw=checked(job/'data/timeline.json',row['files_sha256']['data/timeline.json'])
        expected=[{'at_ms':case['baseline_slot']*.5,'operation':'set',
                   'apps':[]if arm=='OFF'else[APPS[arm][0]],'cell':'*'}]
        require(json.loads(raw)==expected, 'exact_post_warmup_singleton_timeline_required')
        require(arm not in timelines or timelines[arm]==raw, 'timeline_bytes_differ_across_fresh_seeds:'+arm)
        timelines[arm]=raw
    require(set(timelines)=={'OFF',*APPS}, 'all_seven_sealed_timelines_required')
    return timelines


def prepare_release(results_path, protocol_dir, source_archive, runner_path,
                    criteria_path=None,criteria_sha256=None,D_diagnostics=()):
    """Read/verify all retained evidence and return bytes; never publish here."""
    results_path = Path(results_path).resolve(); protocol_dir = Path(protocol_dir).resolve()
    report = read(results_path)
    # Reject discovery/partial/compact before doing expensive raw-file checks.
    require(report.get('analysis_role') == 'validation', 'final_validation_role_required')
    require('complete_cases' in report, 'full_results_not_compact_required')
    manifest = read(results_path.parent/'analysis-manifest.json')
    require(manifest.get('files', {}).get('results.json', {}).get('sha256') == file_sha(results_path),
            'analysis_manifest_result_pin_required')
    design = read(protocol_dir/'design.json'); protocol = read(protocol_dir/'protocol.json')
    protocol_manifest = read(protocol_dir/'manifest.json')
    for name,pin in protocol_manifest.get('files_sha256', {}).items():
        path = (protocol_dir/name).resolve()
        require(path.is_relative_to(protocol_dir), 'unsafe_protocol_file')
        checked(path, pin)
    require({'design.json','protocol.json'} <= protocol_manifest.get('files_sha256', {}).keys(),
            'protocol_manifest_required')
    plans = [json.loads(checked(p['path'],p['sha256'])) for p in report['plan_inputs']]
    require(manifest.get('plans') == report['plan_inputs'], 'analysis_plan_inputs_mismatch')
    gate = validate_report(report,plans,design,protocol)
    require(criteria_path is not None and criteria_sha256 is not None, 'original_pinned_validation_criteria_required')
    criteria_raw=checked(criteria_path,criteria_sha256);criteria=json.loads(criteria_raw)
    criteria_assessment=validate_criteria(criteria,gate,design,protocol)
    for pair in report['paired_results']:
        if pair['arm']=='F':
            require(pair['post_first_apply_ue4_quota']['budget_gflops']==criteria_assessment['F_ue4_gflops_maximum'],
                    'criteria_F_quota_does_not_match_executed_contract')
    require(manifest.get('complete') == manifest.get('planned') == len(report['complete_cases'])
            and manifest.get('pending') == 0 and manifest.get('paired_single_apps') == len(report['paired_results'])
            and manifest.get('intentionally_not_run',0) == 0, 'analysis_case_counts_mismatch')
    for ref in report['plan_inputs']:
        require(file_sha(protocol_dir/'validation'/Path(ref['path']).name) == ref['sha256'], 'protocol_plan_path_mismatch')
    checked(runner_path,gate['runner_sha256']); archive = archive_files(source_archive)
    for name,pin in gate['frozen_models'].items():
        require(name in archive and sha_bytes(archive[name]) == pin, 'archive_model_pin_mismatch')
    for plan in plans:
        for name,pin in plan['frozen_physical_files'].items():
            require(name in archive and sha_bytes(archive[name]) == pin, 'archive_physical_file_pin_mismatch')
    audit_ref = report.get('activation_audit') or {}
    audit_raw = checked(audit_ref['path'],audit_ref.get('sha256')); audit = json.loads(audit_raw)
    require(audit.get('schema') == 'oran.environment-activation-audit.v1', 'activation_audit_required')
    audit_entries = {e.get('job_id',e.get('case_id')):e for e in audit['entries']}
    require(len(audit_entries) == len(audit['entries']), 'duplicate_activation_entry')
    by_worker = {p['worker_id']:p for p in plans}; configurations = {}
    for row in report['complete_cases']:
        configurations[row['case']['id']] = verify_recording(row,by_worker[row['worker_id']])
        if row['case']['arm'] in APPS:
            require(row['control_provenance']['evidence'] == audit_entries.get(row['case']['id']), 'activation_audit_payload_mismatch')
        if row['case']['arm'] == 'F':
            verify_F_quota(row,configurations[row['case']['id']])
    D_identity,D_files=bind_D_diagnostics(D_diagnostics,report,results_path,gate)
    timelines=sealed_timelines(report['complete_cases'],gate['scene'])
    selected = next(r for r in report['complete_cases'] if r['case']['scene'] == gate['scene'])
    case = selected['case']; source_root = Path(by_worker[selected['worker_id']]['source_root'])
    files = {'timelines/'+arm+'.json':raw for arm,raw in timelines.items()}
    files.update(D_files)
    for field,output,default in [('traffic_profile','traffic.json',None),
                                 ('ue_services_config','ue-services.json',None),
                                 ('ap_services_profile','ap-services.json',AP_CONFIG),
                                 ('base_config','base.json',BASE_CONFIG)]:
        remote = Path(case[field]) if case.get(field) else source_root/default if default else None
        require(remote is not None, 'explicit_selected_recipe_required:'+field)
        pin = case.get(field+'_sha256')
        if remote.is_relative_to(source_root):
            relative = str(remote.relative_to(source_root)); raw = archive.get(relative)
            pin = pin or by_worker[selected['worker_id']]['frozen_physical_files'].get(relative)
            require(raw is not None and sha_bytes(raw) == pin, 'archive_recipe_pin_mismatch:'+field)
        else:
            raw = checked(protocol_dir/'profiles'/remote.name,pin)
        if field=='base_config' and design.get('runner_version')==4:
            require(not remote.is_relative_to(source_root), 'explicit_external_v4_base_required')
            difference=validate_geometry_base(raw,archive[BASE_CONFIG])
            files['configs/frozen-base.json']=archive[BASE_CONFIG]
            files['evidence/base-config-difference.json']=encoded(difference)
        files['configs/'+output] = raw
    proofs = []
    for pair in report['paired_results']:
        row = next(r for r in report['complete_cases'] if r['case']['id'] == pair['job_id'])
        proofs.append({k:deepcopy(pair.get(k)) for k in ('arm','seed','worker_id','job_id','off_job_id',
            'before_comparison','external_input_comparison','post_first_apply_ue4_quota')})
        proofs[-1].update(origin_id=row['origin_id'],origin_guard=row['origin_guard'],checkpoint=row['checkpoint'],
            control_provenance=pair['control_provenance'],external_hashes={w:{k:row['flows'][w][k]
                for k in ('exogenous_arrivals_sha256','task_births_sha256')}for w in ('before','after')})
    files.update({'evidence/results.json':results_path.read_bytes(), 'evidence/analysis-manifest.json':encoded(manifest),
                  'evidence/criteria.json':criteria_raw,
                  'evidence/design.json':encoded(design), 'evidence/protocol.json':encoded(protocol),
                  'evidence/pair-proofs.json':encoded(proofs), 'evidence/activation-audit.json':audit_raw})
    reproduction=reproduction_recipe(gate,protocol)
    # Check both extrema of the permitted seed substitution without executing.
    for arm in ('OFF',*APPS):
        for seed in (0,2**32-1):reproduction_arguments(reproduction,Path('/environment-artifact'),arm,seed)
    files['reproducibility.json']=encoded(reproduction)
    for plan in plans:
        files['evidence/plans/'+plan['worker_id']+'.json'] = encoded(plan)
    release = {**gate,'schema':SCHEMA,'status':'validated_common_environment',
        'validation_criteria':{'path':str(Path(criteria_path).resolve()),'sha256':criteria_sha256,'assessment':criteria_assessment},
        'D_identity_diagnostics':D_identity,
        'validation_results_sha256':file_sha(results_path), 'validation_results_original_path':str(results_path),
        'observed_window':{k:protocol[k] for k in ('baseline_slot','origin_observe_slots','observe_slots','tail_slots','total_slots')},
        'six_target_KPIs':{arm:{'app_id':app,'target_kpi':k,'direction':'increase'if d>0 else'decrease','unit':u}
                           for arm,(app,k,d,u) in APPS.items()},
        'portable_config_paths':{'traffic_profile':'configs/traffic.json','ue_services_config':'configs/ue-services.json',
                                 'ap_services_profile':'configs/ap-services.json','base_config':'configs/base.json'},
        'labels':{'tested_source':'CBE; exact pinned source archive and frozen six models',
                  'working_tree':'Its whole-source hash may differ because of harness changes; compatibility is not asserted here.',
                  'KG':'No borrowed native/scenario gains; this artifact does not calibrate or publish KG relationships.',
                  'evidence':'Observed matched single-app gains on the declared fresh seeds/window; not a universal performance guarantee.'},
        'automatic_runtime_default_UI_Git_or_global_changes':False,
        'files_sha256':{n:sha_bytes(b)for n,b in files.items()}}
    if design.get('runner_version')==4:
        release['geometry_change']=deepcopy(GEOMETRY_CHANGE)
        release['base_config_evidence']={
            'external_base_path':'configs/base.json','external_base_sha256':sha_bytes(files['configs/base.json']),
            'frozen_base_path':'configs/frozen-base.json','frozen_base_sha256':sha_bytes(files['configs/frozen-base.json']),
            'difference_path':'evidence/base-config-difference.json',
            'difference_sha256':sha_bytes(files['evidence/base-config-difference.json'])}
    files['environment.json'] = encoded(release)
    return files


def publish_bundle(files, output):
    """Write only a new directory, or accept a byte-identical existing bundle."""
    output = Path(output).absolute()
    require(not output.is_symlink(), 'release_destination_must_not_be_symlink')
    if output.resolve().is_relative_to(ROOT):
        require(output.resolve().is_relative_to(ROOT/'outputs') and output.resolve() != ROOT/'outputs',
                'project_destination_must_be_dedicated_outputs_directory')
    require(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in files), 'unsafe_release_filename')
    if output.exists():
        require(output.is_dir() and not any(p.is_symlink() for p in output.rglob('*')), 'release_destination_not_plain_directory')
        present = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}
        require(present == set(files) and all((output/n).read_bytes() == data for n,data in files.items()),
                'refusing_to_overwrite_existing_release')
        return False
    require(output.parent.is_dir(), 'choose_existing_release_parent_directory')
    output.mkdir()
    for name,data in files.items():
        path = output/name; path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',required=True,type=Path)
    p.add_argument('--protocol',required=True,type=Path)
    p.add_argument('--source-archive',required=True,type=Path)
    p.add_argument('--runner',required=True,type=Path)
    p.add_argument('--criteria',required=True,type=Path,help='Original prospective acceptance criteria JSON')
    p.add_argument('--criteria-sha256',required=True,help='Original pinned criteria bytes, not a newly selected rule')
    p.add_argument('--D-diagnostic',action='append',type=Path,default=[],help='Final-source independent D identity diagnostic; one per seed')
    p.add_argument('--D-diagnostic-sha256',action='append',default=[],help='SHA256 corresponding to each --D-diagnostic')
    p.add_argument('--output',type=Path,help='Explicit new release directory; absent means verify only')
    args = p.parse_args()
    try:
        require(len(args.D_diagnostic)==len(args.D_diagnostic_sha256),'D_diagnostic_paths_and_pins_must_correspond')
        files = prepare_release(args.results,args.protocol,args.source_archive,args.runner,args.criteria,args.criteria_sha256,
                                list(zip(args.D_diagnostic,args.D_diagnostic_sha256)))
        created = publish_bundle(files,args.output) if args.output else False
    except (ReleaseError,OSError,KeyError,TypeError,ValueError,tarfile.TarError) as error:
        p.exit(2,'Release refused: '+str(error)+'\n')
    print(json.dumps({'status':'validated','published':args.output is not None,'created':created,
                      'output':str(args.output)if args.output else None,'new_simulations':0}))


if __name__ == '__main__':
    main()
