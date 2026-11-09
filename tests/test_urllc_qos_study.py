"""Matched URLLC study bookkeeping; synthetic records make no RF/PHY claim."""
from copy import deepcopy
import json

import pytest

from experiments import run_urllc_qos_study as study


def test_invalid_duration_seed_and_output_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    valid = ['--output', str(tmp_path/'outputs/trial')]
    args = study.parse_args(valid)
    assert args.train_seconds == 30.
    for extra in (['--pilot-seconds', '.1'], ['--eval-seeds', '179'],
                  ['--eval-seeds', '55', '55'], ['--min-free-gb', '.1']):
        with pytest.raises(ValueError):
            study.parse_args(valid + extra)
    with pytest.raises(ValueError):
        study.parse_args(['--output', str(tmp_path/'elsewhere')])


def test_train_test_seeds_are_disjoint_and_pilot_is_not_reported_as_training(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    args = study.parse_args(['--output', str(tmp_path/'outputs/trial')])
    train = study.cases_for(args, 'train')
    pilots = study.cases_for(args, 'pilot')
    evaluations = study.cases_for(args, 'evaluate')
    assert len(train) == 2 and all(c[2] == 'train' for c in train)
    assert [c[1] for c in train] == ['native', 'burst']
    assert [c[3] for c in train] == [10., 20.]
    assert sum(c[3] for c in train) == args.train_seconds
    assert {c[4] for c in train}.isdisjoint(c[4] for c in pilots + evaluations)
    assert {c[2] for c in pilots} == {'rr', 'qos1', 'qos2', 'qos4'}
    assert {c[2] for c in evaluations} == {'rr', 'qos1', 'qos2', 'qos4', 'rf'}
    assert {c[1] for c in evaluations} == {'native', 'burst', 'alternating'}
    assert evaluations[0][1] == 'burst'
    assert all(c[3] == 3.2 for c in evaluations)
    assert len(evaluations) == 15
    assert {c[1] for c in pilots} == {'native', 'burst'}


def test_commands_use_explicit_policy_and_scenario_with_shared_matched_cache(tmp_path):
    out = tmp_path/'study'; model = out/'models/model.json'
    commands = {arm: study.command_for(out, arm, 'burst', arm, .8, 55, model)
                for arm in ('rr', 'qos1', 'qos2', 'qos4', 'rf', 'train')}
    caches = {tuple(cmd[cmd.index('--channel-cache-root')+1:cmd.index('--channel-cache-root')+2])
              for cmd in commands.values()}
    assert len(caches) == 1
    expected_modes = dict(rr='off', qos1='fixed', qos2='fixed', qos4='fixed', rf='frozen', train='online')
    for arm, cmd in commands.items():
        assert cmd[cmd.index('--urllc-xapp-policy-version')+1] == 'qos_v2'
        assert cmd[cmd.index('--urllc-xapp-mode')+1] == expected_modes[arm]
        assert cmd[cmd.index('--urllc-scenario')+1] == 'burst'
    assert commands['train'][commands['train'].index('--urllc-xapp-balanced-start-windows')+1] == '40'
    assert '--traffic-profile' not in study.command_for(out, 'native', 'native', 'rr', .8, 55, model)


def test_alternating_selection_requires_complete_cycles_but_pressure_can_use_short_pilot(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    base = ['--output', str(tmp_path/'outputs/trial')]
    with pytest.raises(ValueError, match='complete 1.6 s'):
        study.parse_args(base+['--eval-seconds', '.8'])
    args = study.parse_args(base+['--eval-seconds', '.8', '--eval-scenarios', 'burst', 'native'])
    assert {c[1] for c in study.cases_for(args,'evaluate')} == {'burst','native'}
    with pytest.raises(ValueError, match='unique'):
        study.parse_args(base+['--eval-scenarios', 'burst', 'burst'])


def test_unknown_scenario_or_policy_does_not_silently_run_native(tmp_path):
    with pytest.raises(ValueError, match='scenario'):
        study.command_for(tmp_path,'wrong','brust','rr',.8,55,tmp_path/'model')
    with pytest.raises(ValueError, match='policy'):
        study.command_for(tmp_path,'wrong','burst','oops',.8,55,tmp_path/'model')


def test_alternating_arms_share_scenario_duration_seed_and_channel_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    args=study.parse_args(['--output',str(tmp_path/'outputs/trial'), '--eval-scenarios','alternating'])
    cases=study.cases_for(args,'evaluate')
    commands=[study.command_for(args.output,*c[:3],c[3],c[4],tmp_path/'model') for c in cases]
    assert len(commands)==5
    for key in ('--urllc-scenario','--duration-s','--seed','--channel-cache-root','--traffic-load'):
        assert len({cmd[cmd.index(key)+1] for cmd in commands}) == 1
    assert commands[0][commands[0].index('--urllc-xapp-mode')+1] == 'off'
    assert commands[0][commands[0].index('--urllc-scenario')+1] == 'alternating'


def test_capped_cost_includes_overflow_and_uncompleted_mature_packets(monkeypatch, tmp_path):
    events = [
        dict(kind='arrival', ue_id='ue_2', packet_id='ok', slot_id=1),
        dict(kind='arrival', ue_id='ue_2', packet_id='late', slot_id=2),
        dict(kind='arrival', ue_id='ue_6', packet_id='overflow', slot_id=3, admitted=False),
        dict(kind='arrival', ue_id='ue_6', packet_id='pending', slot_id=4),
        dict(kind='arrival', ue_id='ue_6', packet_id='censored', slot_id=90),
        dict(kind='delivery', ue_id='ue_2', packet_id='ok', slot_id=5, completed=True, latency_ms=2.),
        dict(kind='delivery', ue_id='ue_2', packet_id='late', slot_id=60, completed=True, latency_ms=29.),
        dict(kind='delivery', ue_id='ue_6', packet_id='pending', slot_id=10, completed=False, latency_ms=3.),
        dict(kind='delivery', ue_id='ue_6', packet_id='censored', slot_id=92, completed=True, latency_ms=1.),
    ]
    monkeypatch.setattr(study, 'jsonl', lambda *_: iter(events))
    result = study.cohort_latency(tmp_path, {'manifest': dict(total_slots=100, slot_duration_ms=.5)})
    assert result['mature_packet_count'] == 4
    assert result['deadline_capped_delay_cost_ms'] == 8.


def paired_case(arm='rr'):
    return dict(scenario='burst', seed=55, arm=arm, source_sha256='same-source',
        arrivals=dict(exogenous_slot_inputs_sha256='same-inputs'),
        per_ue={u: dict(mature_offered_inputs_sha256='same-cohort-'+u) for u in study.URLLC},
        urllc=dict(failure_fraction=.2 if arm == 'rr' else .1),
        network_goodput_mbps=40. if arm == 'rr' else 36.)


@pytest.mark.parametrize('change', ['source', 'arrivals', 'cohort'])
def test_comparison_rejects_unmatched_source_arrivals_or_packet_cohorts(change):
    a, b = paired_case(), paired_case('qos1')
    if change == 'source': b['source_sha256'] = 'different'
    elif change == 'arrivals': b['arrivals']['exogenous_slot_inputs_sha256'] = 'different'
    else: b['per_ue']['ue_6']['mature_offered_inputs_sha256'] = 'different'
    with pytest.raises(ValueError): study.compare([a, b])


def test_comparison_reports_absolute_failure_pp_and_goodput_cost_without_filtering():
    cases = study.compare([paired_case(), paired_case('qos1')])
    assert len(cases) == 2
    assert cases[1]['vs_rr']['failure_change_pp'] == pytest.approx(-10.)
    assert cases[1]['vs_rr']['network_goodput_change_pct'] == pytest.approx(-10.)


def test_comparison_requires_each_scenario_rr_and_rejects_duplicate_arms():
    with pytest.raises(ValueError, match='RR reference'):
        study.compare([paired_case('rf')])
    with pytest.raises(ValueError, match='Duplicate'):
        study.compare([paired_case(),paired_case()])


def test_case_audit_pools_packet_counts_instead_of_mean_of_ue_rates(tmp_path, monkeypatch):
    for name, value in {
        'config_snapshot.json': dict(source_sha256='s', manifest=dict(total_slots=200, slot_duration_ms=.5),
                                    configs=dict(ric=dict(xapps=[]), cell={}, ue={}, phy={})),
        'final_summary.json': {},
        'run_status.json': dict(completed=True, source_sha256='s', replay=dict(audits_passed=True)),
        'performance.json': dict(simulation_s=1.),
    }.items():
        (tmp_path/name).write_text(json.dumps(value))
    users = {
        'ue_2': dict(mature_offered_packets=100, mature_failed_packets=10, mature_failure_fraction=.1, goodput_mbps=2.),
        'ue_6': dict(mature_offered_packets=10, mature_failed_packets=5, mature_failure_fraction=.5, goodput_mbps=1.),
    }
    monkeypatch.setattr(study, 'actual_packet_metrics', lambda *_: deepcopy(users))
    monkeypatch.setattr(study, 'arrival_audit', lambda *_: {})
    monkeypatch.setattr(study, 'cohort_latency', lambda *_: dict(mature_packet_count=110))
    monkeypatch.setattr(study, 'controller_audit', lambda *_: {})
    result = study.audit_case(tmp_path, 'rr', tmp_path/'unused-model.json')
    assert result['urllc']['failure_fraction'] == pytest.approx(15/110)
    assert result['urllc']['failure_fraction'] != pytest.approx(.3)
    assert result['urllc']['worst_ue_failure_fraction'] == .5
    monkeypatch.setattr(study, 'cohort_latency', lambda *_: dict(mature_packet_count=109))
    with pytest.raises(ValueError, match='denominator'):
        study.audit_case(tmp_path, 'rr', tmp_path/'unused-model.json')


def test_public_ownership_without_applied_ack_cannot_count_as_control_success(tmp_path, monkeypatch):
    snapshot = dict(cell_0=dict(scheduler_mode='qos_deadline', ue_scheduler_weights=dict(ue_2=1., ue_6=1.)))
    confirmed = dict(kind='executed_policy_confirmed', action=1, produced_slot=20,
                     executed_snapshot=snapshot, owners=dict(cell_0=dict(scheduler_mode='latency_urllc_rf',
                         ue_scheduler_weights='latency_urllc_rf')))
    events = [dict(kind='decision', action=1, status='fixed'), confirmed]
    receipt = dict(status='applied', apply_slot=21, provenance=[dict(implementation_id='latency_urllc_rf')],
                   after=snapshot, fields=[dict(field=k, value=v) for k,v in snapshot['cell_0'].items()])
    monkeypatch.setattr(study, 'jsonl', lambda folder, kind: iter(events if kind == 'urllc_rf_events' else [receipt]))
    with pytest.raises(ValueError, match='earlier matching applied'):
        study.controller_audit(tmp_path, {}, {}, 'qos1', tmp_path/'unused-model.json')


def test_native_rr_cannot_silently_include_xapps(tmp_path):
    with pytest.raises(ValueError, match='no active xApp'):
        study.controller_audit(tmp_path, dict(configs=dict(ric=dict(xapps=[{'implementation_id': 'other'}]))),
                               {}, 'rr', tmp_path/'unused-model.json')


def measured_row(action=0, index=0):
    from ric.near_rt.pools.chongyu.latency_urllc_qos_rf import FEATURES
    from ric.near_rt.pools.chongyu.online_urllc_qos_rf import TARGETS
    from experiments.run_ue_energy_study import POPULATION
    zero = {u: 0 for u in study.URLLC}
    one = {u: 1 for u in study.URLLC}
    snapshot = {'cell_0': dict(scheduler_mode='round_robin' if action == 0 else 'qos_deadline',
        ue_scheduler_weights={u: ((1., 1., 2., 4.)[action] if u in study.URLLC else 1.) for u in POPULATION}),
        'cell_1': dict(scheduler_mode='round_robin' if action == 0 else 'qos_deadline', ue_scheduler_weights={}),
        'cell_2': dict(scheduler_mode='round_robin' if action == 0 else 'qos_deadline', ue_scheduler_weights={})}
    return dict(kind='measured_sample', action=action,
        features=[.1]+[0.]*(len(FEATURES)-5)+[float(action == i) for i in range(4)],
        targets=[0., 20., 20., .048, 0., 0.], duration_s=.1,
        produced_start_slot=60+240*index, produced_end_slot=260+240*index,
        snapshot=snapshot, membership={u: 'cell_0' for u in POPULATION},
        delivered_bits_by_ue={u: 800 for u in POPULATION}, offered_bits_by_ue={u: 800 for u in POPULATION},
        deadline_failed_packets_by_ue=deepcopy(zero), overflow_packets_by_ue=deepcopy(zero),
        completed_packets_by_ue=deepcopy(one), admitted_packets_by_ue=deepcopy(one),
        on_time_completed_packets_by_ue=deepcopy(one), late_completed_packets_by_ue=deepcopy(zero),
        completion_latency_ms_by_ue={u: 1. for u in study.URLLC},
        completion_mean_ms_by_ue={u: 1. for u in study.URLLC},
        original_latency_budgets_ms={u: 10 for u in study.URLLC},
        target_order=TARGETS, retained_for_training=True)


@pytest.mark.parametrize('change', ['target', 'action', 'mode', 'weight', 'deadline'])
def test_measured_labels_reject_arithmetic_and_control_mapping_tampering(change):
    row = measured_row(2)
    study.verify_qos_sample(row)
    if change == 'target': row['targets'][0] = 1.
    elif change == 'action': row['features'][-4:] = [1., 0., 0., 0.]
    elif change == 'mode': row['snapshot']['cell_0']['scheduler_mode'] = 'latency_aware'
    elif change == 'weight': row['snapshot']['cell_0']['ue_scheduler_weights']['ue_2'] = 1.
    else: row['original_latency_budgets_ms']['ue_6'] = 1.
    with pytest.raises(ValueError): study.verify_qos_sample(row)


def training_fixture(folder):
    """Synthetic complete recording + valid constant-tree schema; not a model fit."""
    from ric.near_rt.pools.chongyu.latency_urllc_qos_rf import FEATURES
    from ric.near_rt.pools.chongyu.online_urllc_qos_rf import FIT_PARAMETERS, TARGETS, FORMAT, VERSION
    from experiments.run_scheduler_conflict_study import _f32
    from experiments.run_ue_energy_study import POPULATION
    samples = [measured_row(i%4, i) for i in range(40)]
    tree = dict(children_left=[-1], children_right=[-1], feature=[-2], threshold=[-2.], value=[[0.]*6])
    payload = dict(format=FORMAT, version=VERSION, feature_names=FEATURES, target_names=TARGETS,
        target_scales=[1., 20., 20., 1., 1., 1.], seed=179,
        x=[_f32(r['features']) for r in samples], y=[r['targets'] for r in samples],
        forest=[deepcopy(tree) for _ in range(64)], trained_samples=40, fit_count=1,
        prediction_batches=0, fit_parameters=FIT_PARAMETERS, fit_s=0., predict_s=0.)
    model = folder/'model.json'; model.write_text(json.dumps(payload))
    checkpoint = folder/'checkpoint.json'; checkpoint.write_bytes(model.read_bytes())
    events, packets = [], []
    for i, row in enumerate(samples):
        owners = {cid: dict(scheduler_mode='latency_urllc_rf') for cid in row['snapshot']}
        owners['cell_0']['ue_scheduler_weights'] = 'latency_urllc_rf'
        events += [dict(kind='executed_policy_confirmed', action=row['action'],
            produced_slot=row['produced_start_slot']-40, executed_snapshot=row['snapshot'], owners=owners), row]
        for uid in POPULATION:
            packets += [dict(kind='arrival', ue_id=uid, packet_id=f'{uid}:{i}', bytes=100,
                admitted=True, slot_id=row['produced_start_slot']+2),
                dict(kind='delivery', ue_id=uid, packet_id=f'{uid}:{i}', bytes=100,
                completed=True, latency_ms=1., slot_id=row['produced_start_slot']+4)]
    events += [dict(kind='rf_fit', fit_count=1, samples=40),
               dict(kind='model_checkpoint', fit_count=1, trained_samples=40, path=str(checkpoint),
                    sha256=study.sha256(checkpoint))]
    for name, values in [('urllc_rf_events', events), ('packet_events', packets)]:
        (folder/(name+'.jsonl')).write_text(''.join(json.dumps(v)+'\n' for v in values))
    cfg = dict(manifest=dict(total_slots=10000, slot_duration_ms=.5), configs=dict(
        ue=dict(ues=[dict(ue_id=u, latency_budget_ms=10 if u in study.URLLC else 100) for u in POPULATION]),
        ric=dict(xapps=[dict(implementation_id='latency_urllc_rf', parameters=dict(mode='online', policy_version='qos_v2'))])))
    summary = dict(urllc_xapp=dict(samples=40, trained_samples=40, fit_count=1))
    return model, cfg, summary, events, packets


def test_training_lineage_matches_measured_packets_and_fitted_prefix(tmp_path):
    model, cfg, summary, _, _ = training_fixture(tmp_path)
    audit = study.training_audit(tmp_path, cfg, summary, model)
    assert audit['actual_measured_rows'] == audit['fitted_rows'] == 40
    assert audit['fitted_action_counts'] == {0: 10, 1: 10, 2: 10, 3: 10}
    assert audit['raw_packet_counter_arithmetic_verified']
    assert audit['saved_measured_prefix_verified']


@pytest.mark.parametrize('change', ['raw_packet', 'persisted_row', 'checkpoint'])
def test_training_lineage_rejects_tampered_raw_recording_row_or_checkpoint(tmp_path, change):
    model, cfg, summary, _, packets = training_fixture(tmp_path)
    if change == 'raw_packet':
        packets[1]['bytes'] = 101
        (tmp_path/'packet_events.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in packets))
    elif change == 'persisted_row':
        payload = json.loads(model.read_text()); payload['x'][0][0] = .9
        model.write_text(json.dumps(payload))
    else:
        with (tmp_path/'checkpoint.json').open('a') as stream: stream.write(' ')
    with pytest.raises(ValueError): study.training_audit(tmp_path, cfg, summary, model)


@pytest.mark.parametrize('change', [None, 'old_feature', 'old_label', 'fit_offset'])
def test_second_training_episode_appends_to_the_unchanged_prior_prefix(tmp_path, change):
    from experiments.run_scheduler_conflict_study import _f32
    model, cfg, summary, original_events, _ = training_fixture(tmp_path)
    previous = json.loads(model.read_text())
    samples = [deepcopy(e) for e in original_events if e['kind'] == 'measured_sample']
    for row in samples:
        row['features'][0] = .2  # Distinguish this synthetic episode's observed context.
    payload = deepcopy(previous)
    payload['x'] += [_f32(r['features']) for r in samples]
    payload['y'] += [r['targets'] for r in samples]
    payload.update(trained_samples=80, fit_count=3)
    events = []
    for index, sample in enumerate(samples):
        confirmation = deepcopy(original_events[index*2])
        events += [confirmation, sample]
        if index in (19, 39):
            count = 40+index+1; fit = 2 if index == 19 else 3
            checkpoint_payload = dict(payload, x=payload['x'][:count], y=payload['y'][:count],
                                      trained_samples=count, fit_count=fit)
            checkpoint = tmp_path/f'append-checkpoint-{fit}.json'
            checkpoint.write_text(json.dumps(checkpoint_payload))
            events += [dict(kind='rf_fit', fit_count=fit, samples=count),
                dict(kind='model_checkpoint', fit_count=fit, trained_samples=count,
                    path=str(checkpoint), sha256=study.sha256(checkpoint))]
    model.write_text(json.dumps(payload))
    (tmp_path/'urllc_rf_events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    summary['urllc_xapp'].update(samples=80, trained_samples=80, fit_count=3)
    before = deepcopy(previous)
    if change == 'old_feature': before['x'][0][0] = .9
    elif change == 'old_label': before['y'][0][0] = 1.
    elif change == 'fit_offset': before['fit_count'] = 0
    if change:
        with pytest.raises(ValueError):
            study.training_audit(tmp_path, cfg, summary, model, previous=before)
    else:
        result = study.training_audit(tmp_path, cfg, summary, model, previous=before)
        assert result['actual_measured_rows'] == result['prior_measured_rows'] == 40
        assert result['fitted_rows'] == 80 and result['fits'] == result['checkpoints'] == 2
