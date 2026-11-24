"""Admit an independently measured P–K matrix only for its pinned environment."""
import hashlib
import json
from pathlib import Path


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify_current_calibration(entry, recipe):
    """No deployment outcome or intent score is an admissible fitting input."""
    if entry.get('calibration_reuse'):
        from smo.calibration_reuse import verify
        return verify(entry,recipe,verify_current_calibration)
    matrix = _read(entry['influences'])
    if not entry.get('current_calibration'):
        if matrix.get('estimates'):
            raise ValueError('Current-environment gains require an independent calibration certificate')
        return None
    certificate = _read(entry['current_calibration'])
    if certificate.get('schema') != 'oran.smo.current-environment-calibration.v1':
        raise ValueError('Unknown current calibration certificate')
    if certificate.get('combination_evaluation_data_used') is not False or certificate.get('learned_xapps_invoked') is not False:
        raise ValueError('P–K calibration must be independent of deployed xApp combinations')
    if certificate.get('environment_id') != entry['id'] or certificate.get('environment_id') != recipe['id']:
        raise ValueError('Calibration belongs to another network environment')
    if certificate.get('dynamics_sha256') != recipe['dynamics_sha256'] or certificate.get('source_archive_sha256') != recipe['validated_source_archive_sha256']:
        raise ValueError('Calibrated network dynamics differ from the current runtime')
    if certificate.get('profile_sha256s') != {k:v['sha256'] for k,v in recipe['profiles'].items()}:
        raise ValueError('Calibrated traffic/service/radio profiles differ from the current runtime')
    if certificate.get('frozen_models') != recipe['frozen_models']:
        raise ValueError('Calibration model identity differs from the pinned runtime')
    for key in ('influences', 'design', 'measurements'):
        item = certificate.get('files', {}).get(key, {})
        if not item.get('path') or _sha(item['path']) != item.get('sha256'):
            raise ValueError('Changed calibration evidence: '+key)
    if 'origin_equivalence' in certificate['files']:
        item=certificate['files']['origin_equivalence']
        if _sha(item['path'])!=item['sha256']:
            raise ValueError('Changed calibration origin equivalence evidence')
        bridge=_read(item['path'])
        if (bridge.get('schema')!='oran.current-calibration-origin-equivalence.v1' or
            bridge.get('environment_id')!=recipe['id'] or
            bridge.get('reference_origin_sha256')!=recipe.get('validated_origin',{}).get('sha256')):
            raise ValueError('Calibration prefix belongs to a different UI origin')
        if (bridge.get('all_target_kpis_exactly_equal')is not True or bridge.get('physical_prefix_hashes_equal')is not True or
            bridge.get('checked_slots')!=recipe.get('baseline_slot',2040) or bridge.get('deployed_combination_after_kpis_used')is not False):
            raise ValueError('The native physical prefix does not reproduce the current UI origin')
    if Path(certificate['files']['influences']['path']).resolve() != Path(entry['influences']).resolve():
        raise ValueError('Calibration certificate does not describe the selected gain matrix')
    design = _read(certificate['files']['design']['path'])
    dataset = _read(certificate['files']['measurements']['path'])
    scenario = matrix.get('scenario', {})
    if scenario.get('id') != recipe['id'] or scenario.get('calibration_status') != 'controlled_parameter_calibration':
        raise ValueError('Current matrix has no environment-specific parameter calibration')
    if scenario.get('dynamics_sha256') != recipe['dynamics_sha256'] or scenario.get('source_archive_sha256') != recipe['validated_source_archive_sha256']:
        raise ValueError('Gain matrix source identity differs from its certificate')
    if {k:v['sha256'] for k,v in scenario.get('profiles', {}).items()} != certificate['profile_sha256s']:
        raise ValueError('Gain matrix profile identity differs from its certificate')
    if design.get('combination_evaluation_data_used') is not False or design.get('learned_xapps_enabled') is not False:
        raise ValueError('Invalid calibration experiment design')
    observations = dataset.get('measurements', [])
    if not observations or len(observations) != certificate.get('audited_run_count'):
        raise ValueError('Incomplete audited parameter evidence')
    if dataset.get('design_sha256') != _sha(certificate['files']['design']['path']) or matrix.get('provenance',{}).get('design_sha256') != _digest(design):
        raise ValueError('Gain fit does not bind its experimental design')
    measurement_pin=_digest(observations)
    if matrix.get('provenance',{}).get('measurement_sha256') != measurement_pin:
        raise ValueError('Gain fit does not bind the audited parameter dataset')
    seeds = sorted({o['case']['seed'] for o in observations})
    if seeds != certificate.get('seeds') or seeds != design.get('seeds') or len(seeds) < 3:
        raise ValueError('At least three independent paired seed blocks are required')
    if any(o['proof'].get('physical_source_sha256') != recipe['validated_source_sha256'] or
           o['proof'].get('source_archive_sha256') != recipe['validated_source_archive_sha256'] for o in observations):
        raise ValueError('Audited physical run source differs from the validated runtime')
    rows = matrix.get('estimates', [])
    if not rows or {r['parameter'] for r in rows} != {'ue_offload_fraction','ue_drx_configs','scheduler_mode','ue_scheduler_weights','cell_sleep_mode'}:
        raise ValueError('Current calibration must cover all five RCPs')
    if any(r.get('calibration_environment_id') != recipe['id'] or r.get('combination_evaluation_data_used') is not False for r in rows):
        raise ValueError('Gain row environment/provenance differs')
    if any(r.get('data_sha256')!=measurement_pin for r in rows):
        raise ValueError('A gain row borrows a different measurement dataset')
    return dict(kind='controlled_parameter_calibration', current_calibration=True,
        label='当前环境 P–K 增益', independent_seeds=seeds, audited_runs=len(observations),
        estimated_relations=len({(r['parameter'],r['kpi']) for r in rows}), conditional_estimates=len(rows),
        matrix_sha256=_sha(entry['influences']), certificate_sha256=_sha(entry['current_calibration']),
        combination_evaluation_data_used=False, confidence_level=.95,
        reference_label='历史统计参考', reference_url='/contexts/native_1x/')
