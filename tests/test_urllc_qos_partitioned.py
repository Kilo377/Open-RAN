"""Process-boundary orchestration tests; no simulated outcome is fabricated."""
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1]/'scripts/orin/run_urllc_qos_partitioned.py'
spec = importlib.util.spec_from_file_location('qos_partitions', PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_declared_physical_time_and_seeds_are_exact():
    episodes = runner.episodes()
    assert len(episodes) == 6
    assert sum(e['seconds'] for e in episodes) == 30.
    assert sum(e['seconds'] for e in episodes if e['scenario']=='native') == 10.
    assert [e['seed'] for e in episodes] == list(range(179, 185))
    assert [e['balanced_start_windows'] for e in episodes] == [40, 8, 8, 8, 8, 8]
    assert episodes[0]['before_model'] is None
    assert all(e['before_model'] for e in episodes[1:])


def test_warm_start_override_is_last_cli_value(tmp_path):
    episode = runner.episodes()[1]
    cmd = runner.episode_command(tmp_path, episode, tmp_path/'models/current.json')
    assert cmd[-2:] == ['--urllc-xapp-balanced-start-windows', '8']
    assert cmd[cmd.index('--seed')+1] == '180'
    assert cmd[cmd.index('--duration-s')+1] == '4.0'
    assert cmd[cmd.index('--urllc-scenario')+1] == 'burst'
    assert '--traffic-profile' not in cmd


def test_seed_separation_and_output_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner.study, 'ROOT', tmp_path)
    base = ['--output', str(tmp_path/'outputs/study')]
    assert runner.parse_args(base+['--eval-seeds', '56']).eval_seeds == [56]
    for extra in (['--eval-seeds','179'], ['--eval-seeds','184'], ['--eval-seeds','56','56'], ['--native-pid','1']):
        with pytest.raises(ValueError):
            runner.parse_args(base+extra)
    with pytest.raises(ValueError):
        runner.parse_args(['--output', str(tmp_path/'other')])


def test_partitioned_plan_uses_shared_study_schema_and_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner.study, 'ROOT', tmp_path)
    args = runner.parse_args(['--output', str(tmp_path/'outputs/study')])
    shared = runner.study.parse_args(['--output', str(args.output), '--train-seed', '179',
        '--eval-seeds', '56'])
    assert args.eval_seconds == shared.eval_seconds == 3.2
    assert args.eval_scenarios == shared.eval_scenarios == ['burst', 'native', 'alternating']
    assert runner.expected_plan_for(args) == runner.study.plan_for(shared)
    declared = runner.manifest(runner.expected_plan_for(args))
    assert declared['evaluation_scenarios'] == args.eval_scenarios
    assert declared['eval_seconds'] == 3.2
    assert set(declared['profiles_sha256']) == {'burst', 'alternating'}
    assert declared['episodes'] == runner.episodes()


@pytest.mark.parametrize('extra', [
    ['--eval-seconds', '.8'],
    ['--eval-scenarios', 'burst', 'burst'],
])
def test_partitioned_arguments_obey_shared_scenario_validation(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner.study, 'ROOT', tmp_path)
    with pytest.raises(ValueError):
        runner.parse_args(['--output', str(tmp_path/'outputs/study'), *extra])


def test_evaluate_only_preserves_custom_scenarios_duration_and_pinned_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner.study, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'verify_source', lambda plan: None)
    cli = ['--output', str(tmp_path/'outputs/study'), '--eval-seeds', '57',
        '--eval-seconds', '6.4', '--eval-scenarios', 'alternating', 'burst',
        '--resume', '--evaluate-only']
    args = runner.parse_args(cli)
    expected = runner.expected_plan_for(args)
    args.output.mkdir(parents=True)
    (args.output/'plan.json').write_text(json.dumps(expected))
    captured = []
    monkeypatch.setattr(runner.study, 'main', lambda: captured.append(
        runner.study.plan_for(runner.study.parse_args(runner.sys.argv[1:]))))
    # pytest owns sys.argv; isolate the CLI handoff rather than leaking it.
    monkeypatch.setattr(runner.sys, 'argv', ['partitioned-test'])
    runner.main(cli)
    assert captured == [expected]
    assert expected['evaluation_seeds'] == [57]
    assert expected['evaluation_scenarios'] == ['alternating', 'burst']
    assert expected['eval_seconds'] == 6.4


@pytest.mark.parametrize('mutation', ['legacy_schema', 'scenarios', 'duration', 'profiles'])
def test_evaluate_only_refuses_old_or_drifted_plan(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner.study, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'verify_source', lambda plan: None)
    cli = ['--output', str(tmp_path/'outputs/study'), '--resume', '--evaluate-only']
    args = runner.parse_args(cli)
    saved = runner.expected_plan_for(args)
    if mutation == 'legacy_schema':
        saved.pop('evaluation_scenarios'); saved.pop('profiles_sha256')
    elif mutation == 'scenarios':
        saved['evaluation_scenarios'] = ['native', 'burst']
    elif mutation == 'duration':
        saved['eval_seconds'] = 1.6
    else:
        saved['profiles_sha256']['alternating'] = 'changed-profile'
    args.output.mkdir(parents=True)
    (args.output/'plan.json').write_text(json.dumps(saved))
    monkeypatch.setattr(runner.study, 'main', lambda: pytest.fail('Drifted plan reached evaluation'))
    with pytest.raises(ValueError, match='identical existing study plan'):
        runner.main(cli)


def test_source_verification_pins_alternating_profile_too(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner.study, 'ROOT', tmp_path)
    args = runner.parse_args(['--output', str(tmp_path/'outputs/study')])
    plan = runner.expected_plan_for(args)
    plan['source_sha256'] = runner.PINNED_SOURCE
    monkeypatch.setattr(runner, 'source_fingerprint', lambda root: runner.PINNED_SOURCE)
    original_sha256 = runner.sha256
    monkeypatch.setattr(runner, 'sha256', lambda path:
        'changed-alternating' if path == runner.study.PROFILES['alternating'] else original_sha256(path))
    with pytest.raises(ValueError, match='traffic profiles changed'):
        runner.verify_source(plan)


def test_exact_model_checkpoint_copy_preserves_bytes(tmp_path):
    source=tmp_path/'source.json'; target=tmp_path/'sub/after.json'
    source.write_bytes(b'{"x": [1.0], "y": [2.0]}\n')
    runner.exact_copy(source,target)
    assert target.read_bytes() == source.read_bytes()
    assert runner.sha256(source) == runner.sha256(target)


def recorded(folder, *, complete=True, seed=179, balanced=40):
    folder.mkdir(exist_ok=True)
    (folder/'run_status.json').write_text(json.dumps(dict(completed=complete)))
    (folder/'config_snapshot.json').write_text(json.dumps(dict(
        source_sha256=runner.PINNED_SOURCE,
        manifest=dict(total_slots=20000,slot_duration_ms=.5,rng_seed=seed),
        configs=dict(sim=dict(rng_seed=seed),ric=dict(xapps=[dict(parameters=dict(balanced_start_windows=balanced,seed=seed))])))))


@pytest.mark.parametrize('mutation', ['partial', 'seed', 'balanced'])
def test_completed_input_checks_reject_partial_or_changed_run(tmp_path, mutation):
    recorded(tmp_path, complete=mutation!='partial', seed=180 if mutation=='seed' else 179,
        balanced=8 if mutation=='balanced' else 40)
    with pytest.raises(ValueError):
        runner.check_recorded_inputs(tmp_path,runner.episodes()[0],dict(source_sha256=runner.PINNED_SOURCE))


def test_completed_native_accepts_exact_inputs(tmp_path):
    recorded(tmp_path)
    runner.check_recorded_inputs(tmp_path,runner.episodes()[0],dict(source_sha256=runner.PINNED_SOURCE))


def test_physical_episode_seed_varies_but_loaded_learner_seed_remains_native(tmp_path):
    recorded(tmp_path,seed=180,balanced=8)
    path=tmp_path/'config_snapshot.json'; cfg=json.loads(path.read_text())
    cfg['manifest']['total_slots']=8000;path.write_text(json.dumps(cfg))
    runner.check_recorded_inputs(tmp_path,runner.episodes()[1],dict(source_sha256=runner.PINNED_SOURCE))
    model=tmp_path/'model.json';model.write_text(json.dumps(dict(seed=179)))
    runner.require_learner_seed(model)
    model.write_text(json.dumps(dict(seed=180)))
    with pytest.raises(ValueError,match='learner seed'):
        runner.require_learner_seed(model)


def test_resume_refuses_tampered_after_checkpoint_before_expensive_audit(tmp_path):
    episode=runner.episodes()[0]
    audit=tmp_path/episode['audit']; audit.parent.mkdir()
    after=tmp_path/episode['after_model']; after.write_text('{}')
    audit.write_text(json.dumps(dict(source_sha256=runner.PINNED_SOURCE,episode_model_sha256='tampered')))
    with pytest.raises(ValueError,match='lineage/hash'):
        runner.verify_completed(tmp_path,episode,None,dict(source_sha256=runner.PINNED_SOURCE),None)


def membership_noop():
    snapshot=dict(scheduler_mode='qos_deadline',ue_scheduler_weights={'ue_0':1.,'ue_6':2.,'ue_8':1.})
    decision=dict(kind='decision',slot_id=8600,snapshot={'cell_0':deepcopy(snapshot)})
    before={'cell_0':dict(scheduler_mode='qos_deadline',ue_scheduler_weights={'ue_6':2.,'ue_8':1.})}
    rejected=dict(status='rejected',reason='ue_scheduler_weight_target_outside_cell_scope',
        slot_id=8692,issued_at_slot=8691,apply_scope='cell',target_cell_id='cell_0',target_cell_ids=['cell_0'],
        before=before,after=deepcopy(before),fields=[dict(field=k,value=v) for k,v in snapshot.items()],
        provenance=[dict(implementation_id='latency_urllc_rf')])
    return decision,rejected


def test_membership_rejection_is_explicitly_counted_and_raw_input_unchanged():
    decision,rejected=membership_noop(); original=deepcopy(rejected)
    result=runner.classify_stale_rejections([rejected],[decision])
    assert result[0]['stale_ue_ids']==['ue_0'] and result[0]['atomic_noop']
    assert result[0]['matched_decision_slot']==8600
    assert rejected==original


@pytest.mark.parametrize('mutation', ['other_reason','mutated','bad_weight','unknown_uid','wrong_decision','future_decision','no_absent_uid','wrong_scope'])
def test_membership_exception_cannot_hide_other_control_failures(mutation):
    decision,rejected=membership_noop()
    if mutation=='other_reason':rejected['reason']='invalid_ue_scheduler_weight'
    elif mutation=='mutated':rejected['after']['cell_0']['scheduler_mode']='round_robin'
    elif mutation=='bad_weight':rejected['fields'][1]['value']['ue_6']=9.
    elif mutation=='unknown_uid':rejected['fields'][1]['value']['ghost']=1.
    elif mutation=='wrong_decision':decision['snapshot']['cell_0']['ue_scheduler_weights']['ue_6']=1.
    elif mutation=='future_decision':decision['slot_id']=9000
    elif mutation=='no_absent_uid':
        rejected['before']['cell_0']['ue_scheduler_weights']['ue_0']=1.
        rejected['after']=deepcopy(rejected['before'])
    elif mutation=='wrong_scope':rejected['target_cell_ids']=['cell_1']
    with pytest.raises(ValueError):
        runner.classify_stale_rejections([rejected],[decision])


def test_native_membership_auditor_preserves_original_missing_ack_rejection(tmp_path,monkeypatch):
    from experiments import run_urllc_qos_study as study
    original=study.controller_audit
    monkeypatch.setattr(study,'controller_audit',original)
    decision,rejected=membership_noop()
    decision.update(action=1,status='fixed')
    confirmation=dict(kind='executed_policy_confirmed',action=1,produced_slot=8695,
        executed_snapshot=deepcopy(decision['snapshot']))
    # There is an applied action, but it is AFTER the claimed confirmation.
    applied=deepcopy(rejected);applied.update(status='applied',apply_slot=9000)
    applied['after']=deepcopy(decision['snapshot'])
    events=[decision,confirmation]
    monkeypatch.setattr(study,'jsonl',lambda folder,kind:iter([rejected,applied] if kind=='control_receipts' else events))
    runner.require_membership_rejection_audit()
    with pytest.raises(ValueError,match='earlier matching applied action'):
        study.controller_audit(tmp_path,{}, {}, 'qos1',tmp_path/'unused.json')
