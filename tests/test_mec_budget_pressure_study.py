"""The pressure comparison must preserve its scenario and delayed outcomes."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.run_mec_budget_pressure_study import (
    ARMS, build_plan, command_for, compare, parse_args, task_metrics,
)
from experiments.task_event_metrics import TaskEventCollector
from experiments.run_xapp_long_validation import sha256
from ric.near_rt.pools.chongyu.mec_offload_budget_qgbdt import FEATURES
from ric.near_rt.pools.chongyu.online_qgbdt import OnlineQGBDT, parameters

ROOT = Path(__file__).resolve().parents[1]


def fitted_test_model(path):
    learner = OnlineQGBDT(FEATURES,137)
    for p in [0.]*6+[.5]*6:
        learner.add([20.,0.,0.,0.,10.,0.,20.,p],.8)
    # A schema-valid fitted constant forest is sufficient for input validation;
    # these synthetic unit-test rows are never used in a simulation experiment.
    tree = dict(children_left=[-1],children_right=[-1],feature=[-2],threshold=[-2.],value=[0.])
    learner._forest=[deepcopy(tree) for _ in range(32)]
    learner.initial_prediction=.8;learner.fit_parameters=parameters(137)
    learner.trained_samples=12;learner.fit_count=1;learner.save(path)


@pytest.fixture
def configured(tmp_path):
    config=tmp_path/'configs/rt_sys';config.mkdir(parents=True)
    native=json.loads((ROOT/'configs/rt_sys/ue_services.json').read_text())
    (config/'ue_services.json').write_text(json.dumps(native))
    pressure=deepcopy(native);pressure['mec']['tasks_by_ue']['ue_4']['offload_fraction']=1.
    service=config/'ue_services_mec_offload_first.json';service.write_text(json.dumps(pressure))
    scripts=tmp_path/'scripts/orin';scripts.mkdir(parents=True)
    for name in ('run-six-ue-cuda-nr.sh','activate-gpu.sh','activate.sh'):
        (scripts/name).write_text('#!/bin/sh\n')
    model=tmp_path/'outputs/input/model.json';model.parent.mkdir(parents=True)
    fitted_test_model(model)
    args=parse_args(['--output','outputs/study','--model-path',str(model),
        '--expected-model-sha256',sha256(model),'--duration-s','10','--seed','211'],root=tmp_path)
    return tmp_path,args,service


def test_all_three_arms_share_offload_pressure_with_only_controller_mode_changed(configured):
    root,args,_=configured;plan=build_plan(args,root)
    assert plan['slots']==20000
    assert plan['model']['stamp']['samples']==12
    commands={a:command_for(args,a,root) for a in ARMS}
    for arm,cmd in commands.items():
        assert cmd[cmd.index('--ue-services-config')+1]==str(args.service_config)
        assert cmd[cmd.index('--seed')+1]=='211'
        assert cmd[cmd.index('--duration-s')+1]=='10.0'
        assert cmd[cmd.index('--budget-xapp-mode')+1]=={'base':'off','rules':'rules','qgbdt':'frozen'}[arm]
        for kind in ('energy','latency','throughput','urllc','ap'):
            assert cmd[cmd.index('--'+kind+'-xapp-mode')+1]=='off'
    assert '--budget-xapp-model' not in commands['base']
    assert all('--budget-xapp-model' in commands[a] for a in ('rules','qgbdt'))


def test_pressure_plan_rejects_silent_task_workload_or_radio_profile_changes(configured):
    root,args,path=configured
    profile=json.loads(path.read_text());profile['mec']['tasks_by_ue']['ue_4']['compute_gflop']=.06
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError,match='only in UE4'):
        build_plan(args,root)


def test_declared_model_checksum_and_real_coverage_are_required(configured):
    root,args,_=configured
    args.expected_model_sha256='0'*64
    with pytest.raises(ValueError,match='checksum'):
        build_plan(args,root)
    args.expected_model_sha256=sha256(args.model_path)
    payload=json.loads(args.model_path.read_text())
    payload['x'][-1][-1]=0.
    args.model_path.write_text(json.dumps(payload));args.expected_model_sha256=sha256(args.model_path)
    with pytest.raises(ValueError,match='coverage'):
        build_plan(args,root)


def test_scoring_keeps_late_completions_after_public_policy_bucket_stops_updating():
    history=TaskEventCollector(1.,target_uids=('ue_0','ue_4'))
    policies={u:{} for u in ('ue_0','ue_4')}
    for tid,uid in enumerate(policies):
        event=dict(kind='task_generated',task_id=tid,ue_id=uid,time_s=.1,
            created_s=.1,deadline_s=.1,cohort_index=1,offloaded=True)
        history.observe(event)
        # The historical public snapshot still says pending. Only its executed
        # p and birth count are reused; the later event supplies final outcomes.
        policies[uid][(.1,.2)]=dict(generated=1,mean_offload_fraction=.5,pending=1)
        history.observe(dict(event,kind='task_completed',time_s=.85,latency_ms=750.))
    exact=history.summary()
    summary={'ue_services':{'per_ue':{uid:{'ue_task_metrics':v['whole_run']}
        for uid,v in exact['per_ue'].items()}}}
    result=task_metrics(history,policies,summary,dict(interval_ms=50.,compute_gflop=.05))
    assert result['tasks']['completed']==2
    assert result['tasks']['pending']==0
    assert result['tasks']['on_time_completed']==0
    assert result['tasks']['completed_task_mean_latency_ms']==750.
    assert result['expected_mec_gflops']==.5
    assert result['realized_mec_gflops']==.5


def test_missing_executed_policy_cannot_masquerade_as_zero_compute_demand():
    history=TaskEventCollector(1.,target_uids=('ue_0','ue_4'))
    for tid,uid in enumerate(('ue_0','ue_4')):
        history.observe(dict(kind='task_generated',task_id=tid,ue_id=uid,time_s=.1,
            created_s=.1,deadline_s=.1,cohort_index=1,offloaded=False))
    exact=history.summary()
    summary={'ue_services':{'per_ue':{uid:{'ue_task_metrics':v['whole_run']}
        for uid,v in exact['per_ue'].items()}}}
    with pytest.raises(ValueError,match='executed-generation policy'):
        task_metrics(history,{'ue_0':{},'ue_4':{}},summary,dict(interval_ms=50.,compute_gflop=.05))


def test_pairing_rejects_exogenous_changes_but_allows_endogenous_route_outcomes():
    plan=dict(duration_s=10.,seed=211,source_sha256='s',model={'sha256':'m','stamp':{}},
        scenario='mec_offload_first',scope='test')
    base=dict(source_sha256='s',model_sha256='m',physical_inputs_sha256='p',
        exogenous_slot_inputs_sha256='e',task_birth_inputs_sha256='b',
        controller={'valid_control_evidence':True},
        metrics=dict(mature_birth_counts={'ue_4':[[1,2]]},realized_mec_gflops=1.))
    on=deepcopy(base);on['metrics']['realized_mec_gflops']=.5
    result=compare({'base':base,'rules':on},plan)
    assert result['matched_inputs_verified']
    on['exogenous_slot_inputs_sha256']='different'
    with pytest.raises(ValueError,match='exogenous'):
        compare({'base':base,'rules':on},plan)


@pytest.mark.parametrize('seed',[55,137,138])
def test_historical_calibration_and_training_seeds_are_not_new_evaluation(configured,seed):
    root,args,_=configured
    with pytest.raises(ValueError,match='fresh'):
        parse_args(['--output','outputs/new','--model-path',str(args.model_path),
            '--seed',str(seed)],root=root)
