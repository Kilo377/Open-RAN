"""Short ablation audit/driver tests; synthetic records make no PHY claim."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from experiments import run_mec_stage_ablation as study


def trained_test_artifact(path):
    """A tiny explicit tree fixture for serializer/plan checks, never a result."""
    learner=study.OnlineEnergyRF(study.FEATURES,min_samples=2,trees=1,seed=179)
    for _ in range(2):learner.add([0.]*len(study.FEATURES),[1.,1.,0.,10.])
    learner._forest=[dict(children_left=[-1],children_right=[-1],feature=[-2],
        threshold=[-2.],value=[[1.,1.,0.,10.]])]
    learner.trained_samples=2;learner.fit_count=1;learner.fit_parameters={'max_features':1.}
    path.parent.mkdir(parents=True,exist_ok=True);learner.save(path)
    return path


def arguments(tmp_path,monkeypatch,extra=()):
    monkeypatch.setattr(study,'ROOT',tmp_path)
    model=trained_test_artifact(tmp_path/'outputs/models/frozen.json')
    return study.parse_args(['--output','outputs/diagnostic','--model-path',str(model),*extra])


def test_default_commands_share_native_inputs_and_read_only_model(tmp_path,monkeypatch):
    args=arguments(tmp_path,monkeypatch)
    assert args.duration_s==.8 and args.seed==57 and args.min_free_gb==.5
    commands={arm:study.command_for(args,arm) for arm in study.ARMS}
    for arm,cmd in commands.items():
        assert '--traffic-profile' not in cmd and '--urllc-scenario' not in cmd
        assert '--mec-stage-metrics' in cmd
        for key,value in [('--duration-s','.8'),('--traffic-load','1'),('--ue-services','drx-mec'),
                          ('--xapp-pool','chongyu'),('--record-compression','gzip')]:
            actual=cmd[cmd.index(key)+1]
            assert float(actual)==float(value) if key=='--duration-s' else actual==value
        assert cmd[cmd.index('--latency-xapp-model')+1]==str(args.model_path)
        assert cmd[cmd.index('--latency-xapp-mode')+1]==('off' if arm=='off' else 'frozen' if arm=='rf' else 'fixed')
    assert len({cmd[cmd.index('--channel-cache-root')+1] for cmd in commands.values()})==1
    assert [commands[a][commands[a].index('--latency-xapp-fixed-action')+1] for a in study.FIXED]==['1','2','3']


@pytest.mark.parametrize('extra',[
    ['--duration-s','nan'],['--duration-s','.2'],['--duration-s','.85'],['--duration-s','1.7'],
    ['--seed','-1'],['--min-free-gb','.1'],['--output','elsewhere'],
    ['--model-path','outputs/diagnostic/model.json']])
def test_invalid_or_unsafe_study_arguments_are_rejected(tmp_path,monkeypatch,extra):
    with pytest.raises(ValueError):arguments(tmp_path,monkeypatch,extra)


def timing(created,observed,*,stage='downlink',elapsed=None,terminal=True):
    elapsed=(observed-created)*1000 if elapsed is None else elapsed
    stages=dict.fromkeys(study.STAGES,0.);stages[stage]=elapsed
    service=dict.fromkeys(study.SERVICE,0.)
    waiting=dict.fromkeys(study.SERVICE,0.)
    if stage in waiting:waiting[stage]=elapsed
    return dict(schema=study.SCHEMA,elapsed_ms=elapsed,stage_ms=stages,service_ms=service,
        wait_or_quantization_ms=waiting,upload_blocked_ms=0.,upload_blocked_by_reason_ms={},
        terminal_stage=stage if terminal else None,current_stage=stage,observed_until_s=observed)


@pytest.mark.parametrize('mutation',['sum','clock','schema','negative','service','blocked','terminal'])
def test_stage_conservation_rejects_bad_timing(mutation):
    value=timing(.1,.12)
    if mutation=='sum':value['stage_ms']['downlink']+=1
    elif mutation=='clock':value['observed_until_s']+=.1
    elif mutation=='schema':value['schema']='other'
    elif mutation=='negative':value['stage_ms']['upload']=-1
    elif mutation=='service':value['service_ms']['edge_compute']=1
    elif mutation=='blocked':value['upload_blocked_ms']=1
    else:value['terminal_stage']=None
    with pytest.raises(ValueError):study.validate_timing(value,created_s=.1,observed_s=.12,terminal=True)


def task_history():
    history=study.TaskEventCollector(.8,start_s=.1,target_uids=study.TARGETS)
    events=[]
    def add(uid,tid,birth,route,finish=None,kind='task_completed',stage='downlink'):
        event=dict(kind='task_generated',ue_id=uid,task_id=tid,created_s=birth,time_s=birth,
            deadline_s=.1,cohort_index=round(birth/.1) if abs(birth/.1-round(birth/.1))<1e-8 else int(birth/.1),offloaded=route)
        events.append(event)
        if finish is not None:
            events.append(dict(event,kind=kind,time_s=finish,latency_ms=(finish-birth)*1000,
                task_stage_timing=timing(birth,finish,stage=stage)))
    add('ue_0',1,.1,True,.12)
    add('ue_0',2,.15,False,.175,stage='local_compute')
    add('ue_4',3,.2,True)
    add('ue_4',4,.25,True,.26,kind='task_result_overflow')
    add('ue_0',5,0.,True,.03)  # Startup outcome is not a scored mature cohort.
    add('ue_4',6,.75,True)     # Right-censored final cohort is not scored.
    for e in sorted(events,key=lambda e:(e['time_s'],e['kind']!='task_generated')):history.observe(e)
    exact=history.summary()
    pending=[]
    per_ue={}
    for uid in study.TARGETS:
        whole=exact['per_ue'][uid]['whole_run']
        per_ue[uid]=dict(ue_task_metrics=whole,ue_task_stage_metrics=dict(schema=study.SCHEMA,
            **{k:dict(count=whole[k]) for k in ('completed','failed','pending')}))
    for (uid,tid),t in history.tasks.items():
        if 'terminal' not in t:
            pending.append(dict(task_id=tid,ue_id=uid,created_s=t['created_s'],offloaded=t['offloaded'],
                task_stage_timing=timing(t['created_s'],.8,stage='upload',terminal=False)))
    summary=dict(ue_services=dict(per_ue=per_ue,task_stage_metrics=dict(pending=dict(tasks=pending))))
    return history,summary


def test_stage_metrics_separate_routes_failures_pending_and_censored_tasks():
    history,summary=task_history()
    result=study.task_stage_metrics(history,summary)
    assert result['all_tasks']['generated']==4
    off=result['per_route']['offloaded'];local=result['per_route']['local']
    assert off['generated']==3 and off['completed']['count']==1
    assert off['completed']['elapsed_mean_ms']==pytest.approx(20.)
    assert off['failed']['count']==1 and off['failure_stage_counts']=={'downlink':1}
    assert off['pending']['count']==1 and off['pending']['elapsed_mean_ms']==pytest.approx(600.)
    assert local['completed']['elapsed_mean_ms']==pytest.approx(25.)
    assert result['offloaded_tasks']['completed_task_mean_latency_ms']==pytest.approx(20.)
    assert result['whole_run_pending_by_ue']['ue_4']['count']==2


@pytest.mark.parametrize('mutation',['missing_pending','wrong_pending_identity','counter','stage_counter'])
def test_stage_metrics_cannot_hide_pending_or_disagree_with_runtime(mutation):
    history,summary=task_history()
    if mutation=='missing_pending':summary['ue_services']['task_stage_metrics']['pending']['tasks'].pop()
    elif mutation=='wrong_pending_identity':summary['ue_services']['task_stage_metrics']['pending']['tasks'][0]['offloaded']=False
    elif mutation=='counter':summary['ue_services']['per_ue']['ue_0']['ue_task_metrics']['completed']+=1
    else:summary['ue_services']['per_ue']['ue_4']['ue_task_stage_metrics']['pending']['count']=0
    with pytest.raises(ValueError):study.task_stage_metrics(history,summary)


def controller_records(arm):
    index=study.FIXED.get(arm,2);mode='frozen' if arm=='rf' else 'fixed'
    model=dict(path='/tmp/model.json',stamp=dict(samples=28,trained_samples=28,fit_count=3,prediction_batches=10))
    cfg=dict(configs=dict(ric=dict(xapps=[dict(implementation_id=study.APP,instance_id=study.APP,
        parameters=dict(mode=mode,model_path=model['path'],fixed_action=index))])))
    summary=dict(latency_xapp=dict(mode=mode,samples=28,trained_samples=28,fit_count=3,prediction_batches=12 if arm=='rf' else 10))
    snapshot=dict(scheduler_mode=study.ACTIONS[index]['scheduler_mode'],ue_scheduler_weights={'ue_0':study.ACTIONS[index]['weight']})
    decision=dict(kind='decision',cell_id='cell_0',slot_id=20,action=index,
        scheduler_mode=snapshot['scheduler_mode'],ue_scheduler_weight=study.ACTIONS[index]['weight'],
        ue_ids=['ue_0'],reason='rf_minimum_predicted_e2e_risk' if arm=='rf' else 'predeclared_fixed_scheduler_action')
    confirm=dict(kind='executed_policy_confirmed',cell_id='cell_0',action=index,produced_slot=24,
        executed_config=snapshot,owners={k:study.APP for k in snapshot})
    receipt=dict(status='applied',apply_slot=21,after={'cell_0':dict(snapshot,extra_true_telemetry=1)},
        provenance=[dict(implementation_id=study.APP)])
    return cfg,summary,model,[decision,confirm],[receipt]


@pytest.mark.parametrize('arm',['fixed05','fixed1','fixed2','rf'])
def test_control_audit_requires_real_action_ack_and_frozen_history(tmp_path,monkeypatch,arm):
    cfg,summary,model,events,receipts=controller_records(arm)
    monkeypatch.setattr(study,'jsonl',lambda folder,stem:iter(events if stem=='latency_rf_events' else receipts))
    result=study.controller_audit(tmp_path,cfg,summary,arm,model)
    assert result['execution_ack_and_ownership_verified']
    assert result['genuine_frozen_rf_used']==(arm=='rf')


@pytest.mark.parametrize('mutation',['future_ack','reason','owner','mutated_weight','rejection','fit','history','no_rf'])
def test_control_audit_rejects_false_fixed_or_rf_execution(tmp_path,monkeypatch,mutation):
    arm='rf' if mutation=='no_rf' else 'fixed05'
    cfg,summary,model,events,receipts=controller_records(arm)
    if mutation=='future_ack':receipts[0]['apply_slot']=30
    elif mutation=='reason':events[0]['reason']='invented'
    elif mutation=='owner':events[1]['owners']['scheduler_mode']='other'
    elif mutation=='mutated_weight':receipts[0]['after']['cell_0']['ue_scheduler_weights']={'ue_0':2.}
    elif mutation=='rejection':receipts.append(dict(receipts[0],status='rejected'))
    elif mutation=='fit':events.append(dict(kind='rf_fit'))
    elif mutation=='history':summary['latency_xapp']['samples']+=1
    else:summary['latency_xapp']['prediction_batches']=10
    monkeypatch.setattr(study,'jsonl',lambda folder,stem:iter(events if stem=='latency_rf_events' else receipts))
    with pytest.raises(ValueError):study.controller_audit(tmp_path,cfg,summary,arm,model)


def paired_case(arm='off'):
    history,summary=task_history();tasks=study.task_stage_metrics(history,summary)
    return dict(arm=arm,source_sha256='source',model_sha256='model',physical_inputs_sha256='physical',
        exogenous_slot_inputs_sha256='exogenous',tasks=tasks,network_goodput_mbps=20.)


@pytest.mark.parametrize('mutation',['physical','exogenous','birth','cohort','source','model'])
def test_comparison_refuses_unmatched_inputs_or_task_cohorts(mutation):
    off,rf=paired_case(),paired_case('rf')
    if mutation=='birth':rf['tasks']['mature_birth_identity_sha256']='different'
    elif mutation=='cohort':rf['tasks']['mature_creation_cohorts']['ue_0']=[]
    else:
        key=dict(physical='physical_inputs_sha256',exogenous='exogenous_slot_inputs_sha256',
            source='source_sha256',model='model_sha256')[mutation]
        rf[key]='different'
    with pytest.raises(ValueError):study.compare([off,rf])


def test_resume_reaudits_completed_cases_and_keeps_original_plan(tmp_path,monkeypatch):
    args=arguments(tmp_path,monkeypatch,['--resume'])
    plan=dict(source_sha256='source',model=dict(sha256='model'),config_and_launcher_sha256={})
    monkeypatch.setattr(study,'build_plan',lambda args:deepcopy(plan))
    monkeypatch.setattr(study,'verify_inputs',lambda *args:None)
    args.output.mkdir(parents=True)
    (args.output/'plan.json').write_text(json.dumps(plan))
    cases=[]
    for arm in study.ARMS:
        (args.output/arm).mkdir()
        case=paired_case(arm);cases=study.compare(cases+[case])
        (args.output/(arm+'.audit.json')).write_text(json.dumps(case))
    reaudited=[]
    monkeypatch.setattr(study,'audit_case',lambda folder,arm,args,plan:(reaudited.append(arm) or paired_case(arm)))
    monkeypatch.setattr(study,'launch',lambda *args:pytest.fail('Completed case must not launch'))
    study.main(['--output',str(args.output),'--model-path',str(args.model_path),'--resume'])
    assert reaudited==list(study.ARMS)
    assert json.loads((args.output/'results.json').read_text())['status']=='complete'
    assert json.loads((args.output/'plan.json').read_text())==plan
    assert (args.output/'comparison.html').is_file()


def test_resume_refuses_plan_drift_without_touching_recordings(tmp_path,monkeypatch):
    args=arguments(tmp_path,monkeypatch,['--resume'])
    args.output.mkdir(parents=True);marker=args.output/'existing.log';marker.write_text('retain')
    (args.output/'plan.json').write_text(json.dumps(dict(source_sha256='old')))
    monkeypatch.setattr(study,'build_plan',lambda args:dict(source_sha256='new'))
    monkeypatch.setattr(study,'launch',lambda *args:pytest.fail('Changed plan must not launch'))
    with pytest.raises(ValueError,match='identical plan'):
        study.main(['--output',str(args.output),'--model-path',str(args.model_path),'--resume'])
    assert marker.read_text()=='retain'


def test_resume_requires_existing_pinned_plan_for_existing_recording(tmp_path,monkeypatch):
    args=arguments(tmp_path,monkeypatch,['--resume'])
    (args.output/'off').mkdir(parents=True)
    marker=args.output/'off/raw.json';marker.write_text('original')
    monkeypatch.setattr(study,'build_plan',lambda args:dict(source_sha256='new'))
    with pytest.raises(ValueError,match='without its pinned plan retained'):
        study.main(['--output',str(args.output),'--model-path',str(args.model_path),'--resume'])
    assert not (args.output/'plan.json').exists() and marker.read_text()=='original'


def test_frozen_model_and_config_hashes_are_checked_before_launch(tmp_path,monkeypatch):
    args=arguments(tmp_path,monkeypatch)
    config=tmp_path/'config.json';config.write_text('{}')
    plan=dict(source_sha256=study.source_fingerprint(tmp_path),model=dict(sha256=study.sha256(args.model_path)),
        config_and_launcher_sha256={'config.json':study.sha256(config)})
    study.verify_inputs(args,plan)
    args.model_path.write_text(args.model_path.read_text()+'\n')
    with pytest.raises(ValueError,match='Frozen model changed'):study.verify_inputs(args,plan)


def test_partial_case_is_not_replayed_or_log_overwritten(tmp_path,monkeypatch):
    args=arguments(tmp_path,monkeypatch)
    args.output.mkdir(parents=True);log=args.output/'off.log';log.write_text('failed original log')
    monkeypatch.setattr(study.subprocess,'Popen',lambda *args,**kwargs:pytest.fail('Partial case cannot restart'))
    with pytest.raises(ValueError,match='partial recording/log retained'):study.launch(args,'off',{})
    assert log.read_text()=='failed original log'


def test_legacy_completed_recording_without_opt_in_cannot_be_reused(tmp_path,monkeypatch):
    args=arguments(tmp_path,monkeypatch)
    folder=tmp_path/'legacy';folder.mkdir()
    (folder/'network-replay.html').write_text('historical replay')
    values={
        'run_status.json':dict(completed=True,source_sha256='source',replay=dict(audits_passed=True)),
        'config_snapshot.json':dict(source_sha256='source',manifest=dict(total_slots=1600,
            slot_duration_ms=.5,rng_seed=57),configs=dict(sim=dict(rng_seed=57),phy=dict(rt_sys=dict(ue_services={})))),
        'final_summary.json':dict(total_slots=1600),
    }
    for name,value in values.items():(folder/name).write_text(json.dumps(value))
    with pytest.raises(ValueError,match='opt-in MEC stage metrics'):
        study.audit_case(folder,'off',args,dict(slots=1600,source_sha256='source'))
    assert (folder/'network-replay.html').read_text()=='historical replay'
