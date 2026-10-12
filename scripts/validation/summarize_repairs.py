"""Build an evidence summary only after the complete GPU matrix passes audit."""
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phy.rt_sys.contracts import source_fingerprint


if __name__ == '__main__':
    evidence = ROOT / 'outputs/rt_sys_targeted_repairs'
    audit = json.loads((evidence / 'matrix_audit.json').read_text())
    matrix = json.loads((ROOT / 'outputs/rt_sys_targeted_matrix_nr/matrix_status.json').read_text())
    assert audit['completed'] and len(audit['cases']) == 12 and not audit['errors']
    assert matrix['completed'] and len(matrix['cases']) == 12
    executed = {case['status']['source_sha256'] for case in audit['cases']}
    assert len(executed) == 1
    cases = audit['cases']
    assert all(case['passed'] and case['nr_reference_accounting_checked'] and case['totals']['slots'] == 1000 for case in cases)
    assert all(case['totals']['mu_resource_groups'] > 0 for case in cases if case['scene'] == 'munich')
    checks = {}
    for filename in ('reference_validation', 'release_reference_validation', 'control_validation', 'reference_replay_validation',
                     'reference_mu_validation', 'geometry_validation', 'native_cache_validation'):
        data = json.loads((evidence / (filename + '.json')).read_text())
        if filename == 'native_cache_validation':
            assert len(data) == 2 and all(row['completed'] and row['cache_replay_exact'] and
                row['max_relative_l2_error'] < 1e-5 for row in data), filename
        elif filename == 'control_validation':
            assert data['completed'] and all(all(row['checks'].values()) for row in data['physical'].values()), filename
            assert all(data['closed_loop']['checks'].values()), filename
        else:
            assert data['passed'], filename
        checks[filename] = {'passed': True, 'sha256': hashlib.sha256((evidence / (filename + '.json')).read_bytes()).hexdigest(),
            'path': 'outputs/rt_sys_targeted_repairs/' + filename + '.json'}
    archive = evidence / ('executed_sources_' + next(iter(executed))[:8] + '.tar.gz')
    assert archive.is_file()
    verification = json.loads((evidence / 'release_verification.json').read_text())
    assert verification['source_sha256'] == source_fingerprint(ROOT)
    assert verification['spark_regression']['passed'] and verification['local_regression']['passed']
    data = {
        'completed': True, 'spark_verified': True, 'feedback_mode': 'nr_tb_reference',
        'default_bler_eesm_used': False, 'fast_bler_eesm_calibrated': False, 'full_phy_calibrated': False,
        'executed_matrix_source_sha256': next(iter(executed)), 'release_source_sha256': source_fingerprint(ROOT),
        'executed_source_archive': str(archive.relative_to(ROOT)),
        'executed_archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'matrix_elapsed_s': matrix['elapsed_s'], 'matrix_workers': matrix['workers'],
        'totals': {key: sum(case['totals'].get(key, 0) for case in cases) for key in
            ('slots', 'ue_rows', 'delivered_bytes', 'mu_resource_groups', 'nr_reference_attempts',
             'nr_crc_acks', 'nr_crc_nacks', 'nr_decode_trials', 'nr_constraint_projections', 'completed_handovers')},
        'max_power_error_w': max(case['max_power_error_w'] for case in cases),
        'max_precoder_norm_error': max(case['max_precoder_norm_error'] for case in cases),
        'peak_case_rss_bytes': max(case['status']['peak_rss_bytes'] for case in cases),
        'peak_case_torch_cuda_bytes': max(case['status']['peak_cuda_bytes'] for case in cases),
        'environment': cases[0]['status']['runtime_environment'],
        'checks': checks,
        'release_verification': verification,
        'cases': [{key: case[key] for key in ('case', 'scene', 'ues', 'seed', 'totals', 'natural_mu_correlation_quantiles')}
                  | {'elapsed_s': case['status']['elapsed_s'], 'cache_key': case['status']['cache_key']} for case in cases],
        'limitations': ['Independent post-LMMSE Gaussian resources, not full OFDM interference waveform',
            'Ideal PHY CSI; delayed/noisy controller measurements retained',
            'Finite-sample reference AMC, not a certified rare-error BLER estimator',
            'RV0 reference attempts with existing MAC HARQ lifecycle; no same-codeword IR/Chase combining',
            'Integration validation, not formal cross-scene generalization evidence'],
        'release_delta_from_executed_matrix': 'Added stricter production gate for unused fitted abstraction and enhanced matrix auditor; NR reference physics unchanged',
    }
    target = ROOT / 'docs/rt_sys/targeted_physics_repairs.json'
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: data[key] for key in ('completed', 'totals', 'matrix_elapsed_s', 'max_power_error_w')}, ensure_ascii=False))
