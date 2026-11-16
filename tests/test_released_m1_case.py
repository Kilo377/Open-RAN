"""Latest-release M1 must preserve pairing and distinguish conflict from harm."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import random
import shutil

import pytest

from experiments.run_released_m1_case import (
    APPS, ARMS, BUDGET, CONTRACTS, ENABLED, ENERGY, ENERGY_FEATURES, POPULATION,
    TARGETS, audit_case, build_plan, compare, energy_controller_evidence,
    fingerprint, parse_args,
)
from experiments.run_mec_budget_conflict_study import arbitration_evidence
from experiments.run_xapp_long_validation import sha256
from experiments.task_event_metrics import TaskEventCollector
from ric.near_rt.pools.chongyu.online_rf import OnlineEnergyRF

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def jsonl(path, values):
    path.write_text(''.join(json.dumps(value)+'\n' for value in values))


@pytest.fixture
def configured(tmp_path):
    # These are the actual published immutable artifacts, copied only to a
    # unit-test sandbox. Tests never create experiment rows or train a model.
    release_dir = Path('artifacts/xapps/chongyu/20261109')
    manifest = json.loads((ROOT/release_dir/'release.json').read_text())
    dump(tmp_path/release_dir/'release.json', manifest)
    model_dir = tmp_path/'outputs/trained_xapps/20261109'
    for app in ('x01', 'x13'):
        name = manifest['models'][app]['path']
        for dest in (tmp_path/release_dir/name, model_dir/name):
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/release_dir/name, dest)
    service = json.loads((ROOT/'configs/rt_sys/ue_services.json').read_text())
    dump(tmp_path/'configs/rt_sys/ue_services.json', service)
    script_dir = tmp_path/'scripts/orin'; script_dir.mkdir(parents=True)
    for name in ('run-six-ue-cuda-nr.sh', 'activate-gpu.sh', 'activate.sh'):
        (script_dir/name).write_text('#!/bin/sh\n')
    args = parse_args(['--output', 'outputs/m1', '--duration-s', '.3'], root=tmp_path)
    return tmp_path, args, build_plan(args, tmp_path)


def test_plan_pins_published_models_and_native_four_arm_configuration(configured):
    root, args, plan = configured
    assert plan['slots'] == 600 and plan['seed'] == 20261110
    assert plan['contracts']['energy_population'] == ['ue_0', 'ue_4']
    assert plan['models']['x01']['stamp']['samples'] == 221
    assert plan['models']['x13']['stamp']['samples'] == 13
    release = json.loads(args.release_manifest.read_text())
    assert {a:m['sha256'] for a,m in plan['models'].items()} == {
        a:release['models'][a]['sha256'] for a in ('x01', 'x13')}
    assert not plan['automatic_training']
    assert [c['name'] for c in plan['cases']] == list(ARMS)
    for case in plan['cases']:
        arm = case['name']; cmd = case['command']
        option = lambda flag: cmd[cmd.index(flag)+1]
        assert option('--ue-services-config') == str(args.service_config)
        assert option('--seed') == '20261110'
        assert option('--traffic-load') == '1'
        assert option('--channel-cache-root') == str(args.output/'channel_cache/seed_20261110')
        assert option('--mitigation-seed') == '1701'
        assert option('--mitigation-window-ms') == '100'
        assert option('--direct-conflict-mitigation') == ('uniform_random' if arm == 'both_random' else 'priority')
        for kind, app in (('energy', 'x01'), ('budget', 'x13')):
            assert option('--'+kind+'-xapp-mode') == ('frozen' if app in ENABLED[arm] else 'off')
            assert ('--'+kind+'-xapp-model' in cmd) == (app in ENABLED[arm])
        for kind in ('ap', 'throughput', 'latency', 'urllc'):
            assert option('--'+kind+'-xapp-mode') == 'off'
    assert not args.output.exists()  # --dry-run/build_plan cannot start a study.


def test_rejects_an_older_or_changed_artifact_instead_of_silently_using_it(configured):
    root, args, plan = configured
    path = Path(plan['models']['x01']['path'])
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='checksum'):
        build_plan(args, root)


def test_offload_first_pressure_profile_cannot_erase_native_energy_headroom(configured):
    root, args, _ = configured
    native = json.loads(args.service_config.read_text())
    native['mec']['tasks_by_ue']['ue_4']['offload_fraction'] = 1.
    dump(args.service_config, native)
    with pytest.raises(ValueError, match='native 1x'):
        build_plan(args, root)


def proof():
    return dict(valid_control_evidence=True, genuine_model_inference=True)


def comparison_fixture(plan):
    result = {}
    for arm in ARMS:
        metrics = dict(target_ue_energy_j=2., ue4_energy_j=1., expected_mec_gflops=.5,
            expected_budget_violating_cohorts=0, mature_birth_counts={u:[[1,2]] for u in TARGETS},
            tasks=dict(generated=4, completed=0, failed=1, pending=3,
                on_time_rate=0., completed_task_mean_latency_ms=None))
        result[arm] = dict(source_sha256=plan['source_sha256'],
            model_sha256={a:m['sha256'] for a,m in plan['models'].items()},
            physical_inputs_sha256='physical', exogenous_slot_inputs_sha256='traffic',
            task_birth_inputs_sha256='births', metrics=metrics,
            controller_evidence={app:proof() for app in ENABLED[arm]},
            arbitration_evidence=None, automatic_html_audits_passed=True)
    result['x01']['metrics']['target_ue_energy_j'] = 1.6
    result['x01']['metrics']['expected_mec_gflops'] = 1.
    result['x01']['metrics']['expected_budget_violating_cohorts'] = 1
    result['both_random']['metrics']['target_ue_energy_j'] = 1.8
    result['both_random']['arbitration_evidence'] = dict(errors=[], evidence_error_free=True,
        valid=True, structural_m1_observed=True, unique_random_epochs=3,
        executed_matching_receipts=5, seeded_rng_replay_passed=True)
    return result


def test_structural_conflict_can_satisfy_both_goals_despite_bad_qos(configured):
    _, _, plan = configured; arms = comparison_fixture(plan)
    result = compare(arms, plan, completed=True)
    assert result['structural_m1_observed']
    assert result['isolated_apps_meet_primary_goals']
    assert result['post_mitigation_goal_violations'] == dict(energy=False, budget=False)
    assert result['goal_results']['both_random']['goals_both_met']
    assert result['arms']['both_random']['metrics']['tasks']['pending'] == 3
    assert not result['contracts']['qos_costs_are_validity_gates']


def test_energy_contract_is_aggregate_ue0_plus_ue4_not_diagnostic_ue4_only(configured):
    _, _, plan = configured; arms = comparison_fixture(plan)
    arms['both_random']['metrics'].update(target_ue_energy_j=1.96, ue4_energy_j=.5)
    result = compare(arms, plan, completed=True)
    assert result['goal_results']['both_random']['ue4_energy_ratio_diagnostic'] == .5
    assert not result['goal_results']['both_random']['energy_goal_met']
    assert result['post_mitigation_goal_violations']['energy']
    assert result['structural_m1_observed']


@pytest.mark.parametrize('key', ['source_sha256', 'model_sha256', 'physical_inputs_sha256',
    'exogenous_slot_inputs_sha256', 'task_birth_inputs_sha256'])
def test_changed_inputs_are_not_a_matched_conflict_experiment(configured, key):
    _, _, plan = configured; arms = comparison_fixture(plan)
    arms['x13'][key] = 'changed'
    with pytest.raises(ValueError, match=key):
        compare(arms, plan)


def test_partial_summary_waits_for_combined_arm_and_complete_requires_all(configured):
    _, _, plan = configured; arms = comparison_fixture(plan)
    result = compare({'base':arms['base']}, plan)
    assert result['classification'] == 'pending_combined_arm'
    assert result['post_mitigation_goal_violations'] is None
    with pytest.raises(ValueError, match='four'):
        compare({'base':arms['base']}, plan, completed=True)


def test_equal_requests_are_no_observed_m1_and_not_a_failed_experiment(configured):
    _, _, plan = configured; arms = comparison_fixture(plan)
    arms['both_random']['arbitration_evidence'].update(valid=False,
        structural_m1_observed=False, unique_random_epochs=0, seeded_rng_replay_passed=False)
    result = compare(arms, plan, completed=True)
    assert result['classification'] == 'no_observed_m1'
    assert not result['structural_m1_observed']
    assert result['matched_inputs_verified']


def receipt(winner, value):
    return dict(status='applied', provenance=[dict(implementation_id=winner)],
        fields=[dict(field='ue_offload_fraction', value={'ue_4':value})],
        after={'cell_2':dict(ue_service_policies={'ue_4':dict(offload_fraction=value)})})


def energy_fixture(configured):
    _, _, plan = configured; folder = Path(plan['output']); folder.mkdir(parents=True)
    model = plan['models']['x01']; stamp = model['stamp']
    summary = dict(ue_energy_xapp=dict(mode='frozen', samples=stamp['samples'],
        trained_samples=stamp['trained_samples'], fit_count=stamp['fit_count'], rf_selected_actions=0),
        xapp_library=dict(instances=[dict(implementation_id=ENERGY,
            writable_actions=['ue_offload_fraction', 'ue_drx_configs'], active_cell_ids=['cell_2'])]))
    features = [1., 0., 0., 0., 10., 20., 0., 40., 4., 8., .5]
    prediction = OnlineEnergyRF.load(model['path'], ENERGY_FEATURES).predict([features])[0]
    events = [dict(kind='decision', mode='frozen', ue_id='ue_4',
        reason='no_predicted_energy_gain_baseline', trained_samples=stamp['trained_samples']),
        dict(kind='executed_policy_confirmed', mode='frozen', ue_id='ue_4'),
        dict(kind='measured_sample', mode='frozen', features=features,
            prediction_before_outcome=prediction)]
    jsonl(folder/'energy_rf_events.jsonl', events)
    jsonl(folder/'control_receipts.jsonl', [receipt(ENERGY, .5)])
    return folder, summary, model, events


def test_actual_rf_baseline_inference_is_not_confused_with_no_model_execution(configured):
    folder, summary, model, _ = energy_fixture(configured)
    actual = energy_controller_evidence(folder, summary, model)
    assert actual['valid_control_evidence']
    assert actual['genuine_model_inference']
    assert actual['independent_pre_outcome_prediction_replays'] == 1
    summary['xapp_library']['instances'][0]['active_cell_ids'] = []
    assert not energy_controller_evidence(folder, summary, model)['valid_control_evidence']


def test_prediction_audit_rejects_fabricated_before_outcome_forecast(configured):
    folder, summary, model, events = energy_fixture(configured)
    events[-1]['prediction_before_outcome'][0] += 1.
    jsonl(folder/'energy_rf_events.jsonl', events)
    with pytest.raises(ValueError, match='prediction differs'):
        energy_controller_evidence(folder, summary, model)


def test_random_draw_to_actual_uid_policy_is_independently_replayed(tmp_path):
    rng = random.Random('1701:direct-conflict-mitigation:ue_offload_fraction:ue_4')
    selections = []; receipts = []
    for index in range(1, 4):
        draw = rng.random(); winner = sorted([ENERGY, BUDGET])[int(draw*2)]
        value = 1. if winner == ENERGY else .5
        row = dict(reason='direct_conflict_random_selection', field='ue_offload_fraction',
            ue_id='ue_4', cell_id='cell_2', window_index=index, generation=index,
            mitigation_seed=1701, stream_key='ue_offload_fraction:ue_4', draw_index=index,
            probability=.5, draw=draw, winner=winner, selected_value=value,
            candidates=[dict(source=ENERGY, requested_value=1.), dict(source=BUDGET, requested_value=.5)])
        selections.append(row); applied = receipt(winner, value)
        applied['mitigation_records'] = [dict(row, status='applied')]; receipts.append(applied)
    jsonl(tmp_path/'rule_decisions.jsonl', selections); jsonl(tmp_path/'control_receipts.jsonl', receipts)
    actual = arbitration_evidence(tmp_path)
    assert actual['valid'] and actual['seeded_rng_replay_passed']
    assert actual['unique_random_epochs'] == 3
    receipts[-1]['after']['cell_2']['ue_service_policies']['ue_4']['offload_fraction'] = .123
    jsonl(tmp_path/'control_receipts.jsonl', receipts)
    assert not arbitration_evidence(tmp_path)['valid']


def base_recording(configured):
    _, _, plan = configured; case = plan['cases'][0]; folder = Path(case['output']); folder.mkdir(parents=True)
    service = json.loads(Path(plan['service_config']['path']).read_text()); service['task_cohort_ms'] = 100
    manifest = dict(total_slots=600, slot_duration_ms=.5, rng_seed=plan['seed'],
        ues=[dict(ue_id=u) for u in sorted(POPULATION)], cells=[{}, {}, {}])
    cfg = dict(source_sha256=plan['source_sha256'], manifest=manifest,
        configs=dict(phy=dict(rt_sys=dict(ue_services=service)), ue={},
            ric=dict(xapps=[], direct_conflict_mitigation='priority', mitigation_seed=1701, mitigation_window_ms=100)))
    history = TaskEventCollector(.3, target_uids=TARGETS); births = []; terminal = []
    for index, uid in enumerate(TARGETS):
        event = dict(kind='task_generated', task_id=index, ue_id=uid, time_s=.1,
            created_s=.1, deadline_s=.1, cohort_index=1, offloaded=True)
        history.observe(event); births.append(event)
        if uid == 'ue_0':
            # UE4 remains pending; completed-only delay cannot hide it.
            done = dict(event, kind='task_completed', time_s=.15, latency_ms=50.)
            history.observe(done); terminal.append(done)
    exact = history.summary()
    energy = {u:dict(ue_energy_total_j=1.) for u in POPULATION}
    for uid in TARGETS: energy[uid]['ue_task_metrics'] = exact['per_ue'][uid]['whole_run']
    packet = {u:dict(initial_bytes=0, offered_bytes=10, delivered_bytes=5,
        pending_bytes=3, overflow_bytes=2) for u in POPULATION}
    summary = dict(total_slots=600, ue_services=dict(per_ue=energy), packet_metrics=packet)
    dump(folder/'config_snapshot.json', cfg); dump(folder/'final_summary.json', summary)
    dump(folder/'run_status.json', dict(completed=True, source_sha256=plan['source_sha256'], replay=dict(audits_passed=True)))
    (folder/'network-replay.html').write_text('<html>unit-test fixture</html>')
    with gzip.open(folder/'rt_sys_trace.jsonl.gz', 'wt') as stream:
        for slot in range(1, 601):
            events = births if slot == 200 else terminal if slot == 300 else []
            services = {u:dict(ue_task_cohorts=[dict(exact['per_ue'][u]['cohorts'][0],
                mean_offload_fraction=.5)]) for u in TARGETS} if slot == 600 else {}
            row = dict(slot_id=slot, ues=manifest['ues'], traffic_arrivals={u:dict(exogenous_bytes=0) for u in POPULATION},
                ue_service_events=events, ue_services=services)
            stream.write(json.dumps(row)+'\n')
    return case, plan, folder, summary


def test_full_recording_audit_retains_all_task_outcomes_and_raw_hashes(configured):
    case, plan, folder, _ = base_recording(configured)
    audit = audit_case(case, plan)
    assert audit['metrics']['tasks']['pending'] == 1
    assert audit['metrics']['tasks']['on_time_rate'] == .5
    assert audit['metrics']['target_ue_energy_j'] == 2.
    assert audit['metrics']['expected_mec_gflops'] == .5
    assert set(audit['metrics']['per_ue_packet_qos']) == POPULATION
    assert audit['files_sha256']['rt_sys_trace.jsonl.gz'] == sha256(folder/'rt_sys_trace.jsonl.gz')
    assert len(audit['task_birth_inputs_sha256']) == 64
    assert compare({'base':audit}, plan)['matched_inputs_verified']


def test_final_task_ledger_cannot_hide_missing_or_late_outcomes(configured):
    case, plan, folder, summary = base_recording(configured)
    summary['ue_services']['per_ue']['ue_4']['ue_task_metrics']['pending'] = 0
    dump(folder/'final_summary.json', summary)
    with pytest.raises(ValueError, match='Full task events disagree'):
        audit_case(case, plan)
