"""Released-model M2 protocol checks; no synthetic network KPI claims."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments import run_released_m2_case as m2


def comparison_arms(energy_both=93., goodput_both=108.):
    values = {'base': (100., 100.), 'x02': (90., 98.),
              'x08': (101., 110.), 'both': (energy_both, goodput_both)}
    arms = {}
    for name, (energy, goodput) in values.items():
        active = [key for key in ('x02', 'x08') if name in (key, 'both')]
        arms[name] = dict(source_sha256='source', model_sha256=m2.MODEL_HASHES,
            physical_inputs_sha256='physical', exogenous_slot_inputs_sha256='external',
            task_birth_inputs_sha256='births', urllc_mature_inputs_sha256={'ue_2': 'u2', 'ue_6': 'u6'},
            controller_evidence_verified=True,
            controller_evidence={key: dict(valid=True) for key in active},
            declared_rcp={m2.APP_IDS[key]: sorted(m2.RCP[m2.APP_IDS[key]]) for key in active},
            concurrent_control_evidence=dict(verified=True) if name == 'both' else None,
            metrics=dict(network_ap_energy_j=energy, network_goodput_mbps=goodput,
                urllc=dict(mature_packets=100, failed_mature_packets=2, failure_fraction=.02)))
    return arms


def small_plan():
    return dict(duration_s=3., seed=20261110, source_sha256='source', scope='Test protocol')


def test_frozen_latest_release_plan_uses_all_measured_rows():
    args = m2.parse_args(['--output', 'outputs/test_released_m2_dry'])
    plan = m2.build_plan(args)
    assert plan['slots'] == 6000 and plan['seed'] == 20261110
    assert plan['automatic_training'] is False and plan['frozen_models'] is True
    assert list(plan['models']) == ['x02', 'x08']
    for app, rows in m2.MODEL_ROWS.items():
        assert plan['models'][app]['sha256'] == m2.MODEL_HASHES[app]
        assert plan['models'][app]['stamp']['samples'] == rows
        assert plan['models'][app]['stamp']['trained_samples'] == rows
        assert plan['models'][app]['readiness']['ready']
    assert not args.output.exists(), 'Planning must not create recordings or change models'
    assert {case['name'] for case in plan['cases']} == set(m2.ARMS)
    for case in plan['cases']:
        command = case['command']
        def option(key): return command[command.index(key)+1]
        assert option('--model') == 'fast' and option('--fast-profile') == 'system'
        assert option('--traffic-load') == '2'
        assert option('--direct-conflict-mitigation') == 'priority'
        assert option('--ue-services') == 'drx-mec'
        assert option('--ap-services-profile') == str(args.ap_services_profile)
        assert Path(option('--output')) == args.output/'cases'/case['name']
        for mode in ('energy', 'latency', 'budget', 'urllc'):
            assert option('--'+mode+'-xapp-mode') == 'off'
        assert option('--ap-xapp-mode') == ('frozen' if case['name'] in ('x02', 'both') else 'off')
        assert option('--throughput-xapp-mode') == ('frozen' if case['name'] in ('x08', 'both') else 'off')
    assert set(plan['declared_rcp'][m2.APP]).isdisjoint(plan['declared_rcp'][m2.X08])


def test_expected_traffic_hash_uses_the_runtime_cli_float_type():
    from phy.rt_sys.scenario import build_case
    args = m2.parse_args(['--output', 'outputs/test_released_m2_dry'])
    plan = m2.build_plan(args)
    def ue_inputs(load):
        _, configs, _ = build_case('munich', 3, 6, args.seed, plan['slots'],
            device='cuda', config_path=m2.ROOT/'configs/rt_sys/agx_munich_6ue_depth2_fast.json',
            traffic_load=load, xapp_pool='chongyu')
        return configs['ue']
    actual_cli = ue_inputs(float('2'))
    assert plan['expected_ue_inputs_sha256'] == m2.fingerprint(actual_cli)
    # Numerically equal dictionaries can serialize differently. The audit
    # retains exact input hashing, rather than removing traffic fields.
    old_integer = ue_inputs(2)
    assert old_integer == actual_cli
    assert m2.fingerprint(old_integer) != m2.fingerprint(actual_cli)
    assert plan['expected_ue_inputs_sha256'] != m2.fingerprint(ue_inputs(1.0))
    altered = deepcopy(actual_cli)
    altered['ues'][0]['burst_size_bytes'] += 1.
    assert plan['expected_ue_inputs_sha256'] != m2.fingerprint(altered)


def test_native_base_summary_omits_the_xapp_library():
    # Real 6000-slot Base summary shape: radio/service ledgers exist, while
    # xapp_library and all per-xApp summaries are absent for the all-OFF arm.
    summary = dict(total_slots=6000, rt_sys=dict(model='fast', fast_profile='system'),
        packet_metrics={}, ap_services={}, ue_services={})
    assert m2.controller_descriptors(summary, set()) == {}
    for expected in ({m2.APP}, {m2.X08}, {m2.APP, m2.X08}):
        with pytest.raises(ValueError, match='writable RCPs'):
            m2.controller_descriptors(summary, expected)
    summary['xapp_library'] = dict(instances=[dict(implementation_id=m2.APP,
        writable_actions=['cell_sleep_mode'])])
    assert m2.controller_descriptors(summary, {m2.APP}) == {m2.APP:{'cell_sleep_mode'}}


@pytest.mark.parametrize('extra', [
    ['--seed', '55'], ['--duration-s', '3.01'], ['--min-free-gb', '.1'],
    ['--ap-services-profile', 'configs/rt_sys/ue_services.json'],
    ['--model-manifest', '../../escaped.json'],
])
def test_predeclared_inputs_cannot_silently_change(extra):
    with pytest.raises(ValueError):
        m2.parse_args(['--output', 'outputs/test_released_m2_dry']+extra)


def test_latest_model_checksum_is_mandatory(tmp_path):
    args = m2.parse_args(['--output', 'outputs/test_released_m2_dry'])
    args.model_dir = tmp_path; args.model_manifest = tmp_path/'models.json'
    import json
    (tmp_path/'ap.json').write_text('{}')
    args.model_manifest.write_text(json.dumps(dict(models={
        'x02': dict(path='ap.json', sha256=m2.MODEL_HASHES['x02'])})))
    with pytest.raises(ValueError, match='exact latest released'):
        m2.released_models(args)


def test_standalone_harm_is_not_contract_violation():
    result = m2.compare(comparison_arms(), small_plan(), completed=True)
    assert result['individually_effective'] == {'x02': True, 'x08': True}
    assert result['joint_contracts'] == {'x02': True, 'x08': True}
    assert result['standalone_target_harm_observed']
    assert result['classification'] == 'standalone_target_harm_but_primary_contracts_met'
    assert result['operational_m2_conflict_observed'] is False
    assert result['contracts'] == {'ap_energy_ratio_max': .95, 'goodput_ratio_min': 1.05}


@pytest.mark.parametrize('energy,goodput,failed', [(97., 108., 'x02'), (94., 103., 'x08')])
def test_operational_m2_requires_joint_primary_contract_failure(energy, goodput, failed):
    result = m2.compare(comparison_arms(energy, goodput), small_plan(), completed=True)
    assert result['joint_contracts'][failed] is False
    assert result['disjoint_controls_verified']
    assert result['operational_m2_conflict_observed']
    assert result['classification'] == 'operational_m2_contract_violation'


def test_underperforming_standalone_is_not_m2_conflict():
    arms = comparison_arms(98., 100.)
    arms['x02']['metrics']['network_ap_energy_j'] = 98.
    result = m2.compare(arms, small_plan(), completed=True)
    assert result['individually_effective']['x02'] is False
    assert result['classification'] == 'controllers_not_individually_effective'
    assert result['operational_m2_conflict_observed'] is False


def test_no_qos_acceptance_gate_hides_target_conflict():
    arms = comparison_arms(97., 108.)
    arms['both']['metrics']['urllc']['failed_mature_packets'] = 99
    arms['both']['metrics']['urllc']['failure_fraction'] = .99
    result = m2.compare(arms, small_plan(), completed=True)
    assert result['operational_m2_conflict_observed']
    assert result['arms']['both']['metrics']['urllc']['failure_fraction'] == .99


def test_disjoint_declaration_without_actual_joint_actuation_is_insufficient():
    arms = comparison_arms(97., 108.)
    arms['both']['concurrent_control_evidence']['verified'] = False
    result = m2.compare(arms, small_plan(), completed=True)
    assert not result['disjoint_controls_verified']
    assert not result['operational_m2_conflict_observed']
    assert result['classification'] == 'invalid_concurrent_control_evidence'


@pytest.mark.parametrize('key', ['physical_inputs_sha256', 'exogenous_slot_inputs_sha256',
    'task_birth_inputs_sha256', 'source_sha256', 'urllc_mature_inputs_sha256'])
def test_unmatched_inputs_refuse_comparison(key):
    arms = comparison_arms(); arms['both'][key] = 'changed'
    with pytest.raises(ValueError, match='Unmatched M2 inputs'):
        m2.compare(arms, small_plan(), completed=True)


def test_partial_queue_results_cannot_claim_conflict():
    arms = comparison_arms()
    result = m2.compare({'base': arms['base']}, small_plan())
    assert result['classification'] == 'incomplete' and not result['completed']
    result = m2.compare({k: arms[k] for k in ('base', 'x02')}, small_plan())
    assert result['individually_effective'] == {'x02': True}
    assert not result['operational_m2_conflict_observed']
    with pytest.raises(ValueError, match='Four completed arms'):
        m2.compare({'base': arms['base']}, small_plan(), completed=True)


def actual_ack(cell='cell_0', owner=m2.APP):
    return {cell: dict(ap_policy_owner=owner, ap_policy='micro_sleep',
        cell_sleep_mode='micro_sleep', scheduler_mode='latency_aware',
        ue_scheduler_weights={'ue_2': 2.})}


def scheduler_receipt(after=None):
    return dict(status='applied', apply_slot=200,
        provenance=[dict(implementation_id=m2.X08)],
        fields=[dict(field='scheduler_mode', value='latency_aware'),
            dict(field='ue_scheduler_weights', value={'ue_2': 2.})],
        after=actual_ack() if after is None else after)


def test_concurrent_evidence_reads_actual_same_snapshot(monkeypatch):
    receipt = scheduler_receipt()
    monkeypatch.setattr(m2, 'jsonl', lambda folder, name: iter([receipt]))
    proof = m2.concurrent_control_evidence('unused', ['cell_0'])
    assert proof['verified'] and proof['actual_joint_snapshot_count'] == 1
    assert proof['first_actual_joint_snapshot_slots'] == {'cell_0': 200}


def test_actual_empty_cell_merger_prunes_the_per_ue_weight_field(monkeypatch):
    # The real BOTH run has no serving UEs in cell1; its merged ACK owns
    # scheduler_mode and AP sleep together, without a nonexistent UID bundle.
    receipt = scheduler_receipt()
    receipt['fields'] = [dict(field='scheduler_mode', value='latency_aware'),
        dict(field='cell_sleep_mode', value='deep_sleep')]
    receipt['provenance'] = [dict(implementation_id=m2.APP, fields=['cell_sleep_mode']),
        dict(implementation_id=m2.X08, fields=['scheduler_mode'])]
    receipt['after'] = {'cell_1':dict(ap_policy_owner=m2.APP, ap_policy='deep_sleep',
        cell_sleep_mode='deep_sleep', scheduler_mode='latency_aware',
        ue_scheduler_weights={}, ue_service_policies={})}
    monkeypatch.setattr(m2, 'jsonl', lambda folder, name: iter([receipt]))
    proof = m2.concurrent_control_evidence('unused', ['cell_1'])
    assert proof['verified']
    assert proof['empty_cell_mode_only_ack_counts'] == {'cell_1': 1}
    # A missing telemetry map or an occupied UE-policy map cannot authorize
    # the mode-only exception, even if a faulty receipt has empty weights.
    receipt['after']['cell_1'].pop('ue_service_policies')
    assert not m2.concurrent_control_evidence('unused', ['cell_1'])['verified']
    receipt['after']['cell_1']['ue_service_policies'] = {'ue_2':{}}
    assert not m2.concurrent_control_evidence('unused', ['cell_1'])['verified']


def test_occupied_cell_mode_only_ack_cannot_prove_joint_bundle(monkeypatch):
    receipt = scheduler_receipt()
    receipt['fields'] = [dict(field='scheduler_mode', value='latency_aware')]
    monkeypatch.setattr(m2, 'jsonl', lambda folder, name: iter([receipt]))
    assert not m2.concurrent_control_evidence('unused', ['cell_0'])['verified']


@pytest.mark.parametrize('change', ['wrong_ap_owner', 'rejected', 'missing_cell', 'wrong_scheduler_value'])
def test_nonconcurrent_or_unexecuted_controls_fail_proof(monkeypatch, change):
    receipt = scheduler_receipt()
    if change == 'wrong_ap_owner': receipt['after'] = actual_ack(owner='other')
    if change == 'rejected': receipt['status'] = 'rejected'
    if change == 'wrong_scheduler_value': receipt['after']['cell_0']['scheduler_mode'] = 'round_robin'
    monkeypatch.setattr(m2, 'jsonl', lambda folder, name: iter([receipt]))
    cells = ['cell_0', 'cell_1'] if change == 'missing_cell' else ['cell_0']
    assert not m2.concurrent_control_evidence('unused', cells)['verified']
