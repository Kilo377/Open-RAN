import gzip
import hashlib
import json
from pathlib import Path

import pytest

from experiments import indexed_physical_parity_v4 as p


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def test_no_physical_pointer_is_metadata():
    forbidden={'/cache_key','/system_gain_epoch','/slot_id','/request_id','/source','/action','/throughput_mbps'}
    assert not forbidden & {r['pointer'] for r in p.metadata_rules()}
    a=p.Audit(Path('/tmp'),p.OLD_SOURCE,'own')
    v={'cache_key':'old','request_id':'same','control':{'action':3},'system_gain_epoch':2}
    assert a.transform('rt_sys_trace.jsonl',v)==v


def test_source_rules_are_specific_and_verified():
    a=p.Audit(Path('/tmp'),p.OLD_SOURCE,'own')
    assert a.transform('performance.json',{'source_sha256':p.OLD_SOURCE}) != {'source_sha256':p.OLD_SOURCE}
    assert a.transform('rt_sys_trace.jsonl',{'source_sha256':p.OLD_SOURCE})=={'source_sha256':p.OLD_SOURCE}
    with pytest.raises(ValueError,match='source pin'):
        a.transform('performance.json',{'source_sha256':'anything'})


def test_metadata_type_and_presence_still_matter():
    a=p.Audit(Path('/tmp'),p.OLD_SOURCE,'own')
    with pytest.raises(ValueError):a.transform('performance.json',{'simulation_s':'0.1'})
    assert list(p.exact._differences(a.transform('performance.json',{'simulation_s':1}),{}))
    assert a.exclusions()


def test_instrumentation_schema_correction_is_one_exact_proven_pointer():
    a=p.Audit(Path('/tmp'),p.OLD_SOURCE,'own')
    value=p.exact._load('{"ue_services":{"module_wall_s":0.100000000000000001,"latency_ms":5,"sim_time_s":1}}')
    out=a.transform('final_summary.json',value)
    assert out['ue_services']['latency_ms']==5 and out['ue_services']['sim_time_s']==1
    assert a.scalar_metadata_originals['final_summary.json:/ue_services/module_wall_s']==['0.100000000000000001']
    assert a.transform('rt_sys_trace.jsonl',value)==value
    assert p.policy()['schema_correction']['prospective_first_trial_claimed'] is False


@pytest.mark.parametrize('file,prefix', [('final_summary.json',('xapp_library','instances','0','learning')),
    ('final_summary.json',('xapp_test','app','learning')),('live.json',('xapp_test','app','learning'))])
def test_v4_only_three_performance_aliases_do_not_hide_training_or_predictions(file,prefix):
    a=p.Audit(Path('/tmp'),p.OLD_SOURCE,'own')
    for key in ('fit_s','predict_s','decision_s'):
        assert a.transform(file,p.exact._load('0.100000000000000001'),prefix+(key,)) != p.exact._load('0.100000000000000001')
        assert a.scalar_metadata_originals[file+':/'+ '/'.join(prefix+(key,))]==['0.100000000000000001']
    for key in ('fit_count','fit_calls','trained_samples','prediction_batches','latency_ms','deadline_ms','sim_time_s','prediction','confidence','action'):
        assert a.transform(file,9,prefix+(key,))==9
    assert len(p.LEARNING_TIMING_RULES)==9


def test_v3_code_is_a_separate_preserved_file():
    original=Path(p.__file__).with_name('indexed_physical_parity.py')
    assert p.sha(original)==p.V3_COMPARATOR_SHA


def test_v4_preserves_all_admission_and_physical_comparison_functions():
    import ast
    files=[Path(p.__file__).with_name('indexed_physical_parity.py'),Path(p.__file__)]
    defs=[]
    for file in files:
        tree=ast.parse(file.read_text())
        defs.append({n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))})
    for name in ('verify','inventory','compare_study'):
        assert ast.dump(defs[0][name],include_attributes=False)==ast.dump(defs[1][name],include_attributes=False)
    methods=[{n.name:n for n in d['Audit'].body if isinstance(n,ast.FunctionDef)}for d in defs]
    assert ast.dump(methods[0]['transform'],include_attributes=False)==ast.dump(methods[1]['transform'],include_attributes=False)
    # Only the explanatory metadata classification string differs in the pair
    # driver.  Raw records, typed Decimal equality, sequence order, full seals,
    # physical flags and pre/post hash guards remain identical.
    class Label(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str) and node.value.startswith('Pure host instrumentation;'):
                return ast.copy_location(ast.Constant(value='INSTRUMENTATION_DESCRIPTION'),node)
            return node
    nodes=[Label().visit(d['compare_pair'])for d in defs]
    assert ast.dump(nodes[0],include_attributes=False)==ast.dump(nodes[1],include_attributes=False)


def test_model_paths_are_content_addressed_not_dropped():
    a=p.Audit(Path('/tmp'),p.OLD_SOURCE,'own')
    with pytest.raises(ValueError,match='Unpinned'):
        a.transform('final_summary.json',{'ap_xapp':{'model_path':'unknown'}})
    a.model_paths['model-a']='a'*64;a.model_paths['model-b']='b'*64
    x=a.transform('final_summary.json',{'ap_xapp':{'model_path':'model-a'}})
    y=a.transform('final_summary.json',{'ap_xapp':{'model_path':'model-b'}})
    assert list(p.exact._differences(x,y))
    for file in ('final_summary.json','live.json'):
        x=a.transform(file,{'xapp_test':{'app':{'learning':{'model_path':'model-a'}}}})
        y=a.transform(file,{'xapp_test':{'app':{'learning':{'model_path':'model-b'}}}})
        assert list(p.exact._differences(x,y))


def test_only_communication_nonce_paths_are_exempt():
    a=p.Audit(Path('/tmp'),p.OLD_SOURCE,'own')
    for key in ('request_id','run_id','tb_id','ue_id'):
        x={key:'a'*32};assert a.transform('control_receipts.jsonl',x)==x
    assert a.transform('xapp_hot_start.json',{'timeline':[{'run_id':'a'*32}]}) != {'timeline':[{'run_id':'a'*32}]}
    with pytest.raises(ValueError):a.transform('xapp_hot_start.json',{'timeline':[{'run_id':'bad'}]})


def make_inventory(root,apps=()):
    needed=(p.CORE_STREAMS|{s+'.csv' for s in p.exact.CSV}
        |{s+'.json' for s in p.exact.JSON if s!='evaluation_origin_verified'}
        |{'xapp_model_checkpoints/'+s+'.json' for s in p.exact.MODEL_IDS}
        |{p.APP_STREAMS[x] for x in apps})
    if apps:needed.add('evaluation_origin_verified.json')
    for s in needed:put(root/s,{})


def test_lazy_stream_schema_and_unknown_extra_fail(tmp_path):
    make_inventory(tmp_path)
    assert 'energy_rf_events.jsonl' not in p.inventory(tmp_path,[])
    with pytest.raises(ValueError,match='Missing recordings'):
        p.inventory(tmp_path,['xapp:ue_energy_rf'])
    put(tmp_path/'unregistered.json',{})
    with pytest.raises(ValueError,match='Unknown recording'):p.inventory(tmp_path,[])


@pytest.fixture
def pair(tmp_path,monkeypatch):
    sides=[]
    for i,side in enumerate(('reference','candidate')):
        run=tmp_path/side;run.mkdir();source=(p.OLD_SOURCE,p.NEW_SOURCE)[i]
        files={}
        records={
            'rt_sys_trace.jsonl':[{'slot_id':j,'cache_key':'same','kpi':{'power':1.0000000000000002},'control':{'request_id':'r'+str(j)}} for j in range(1,2841)],
            'control_receipts.jsonl':[{'request_id':'r1','status':'applied','source':'controller','action':{'weight':2}},{'request_id':'r2','status':'applied','action':{'weight':3}}],
            'packet_events.jsonl':[{'packet_id':'p','ue_id':'ue_0','bytes':100}],
            'performance.json':{'source_sha256':source,'simulation_s':10+i,'cache_key':'same'},
        }
        for name,v in records.items():
            f=run/name
            if name.endswith('jsonl'):f.write_text(''.join(json.dumps(x)+'\n' for x in v))
            else:put(f,v)
            files[name]=f
        case=dict(arm='OFF',scene='s',seed=14,app_ids=[],slots=2840,baseline_slot=2040,origin_observe_slots=2000,observe_slots=400,traffic_load=1.5,
            traffic_profile='sameprofile',traffic_profile_sha256='a'*64,ue_services_config='same',ue_services_config_sha256='b'*64,
            ap_services_profile_sha256='c'*64,base_config_sha256='d'*64,id='job'+side)
        seal=dict(worker_id='agx1',case=case,source_sha256=source,source_archive_sha256=(p.OLD_ARCHIVE,p.NEW_ARCHIVE)[i],runner_sha256='e'*64,
                  frozen_models={'m':'p'},origin_id='own'+side,actual_system_hostname='physical-agx1',files_sha256={})
        cp=tmp_path/(side+'.json');pp=tmp_path/(side+'plan.json');put(cp,seal);put(pp,dict(cases=[case]))
        sides.append((seal,{},run,files,p.Audit(run,source,seal['origin_id']),cp,pp))
    # This fixture exercises comparison/normalization.  The real verify()
    # admission tests are separate.  Placeholders contain no job files.
    for side in sides:
        (side[5].parent.parent/'jobs'/side[0]['case']['id']).mkdir(parents=True,exist_ok=True)
    monkeypatch.setattr(p,'verify',lambda cp,plan,side:sides[0 if side=='reference' else 1][:5])
    return sides


def compare(pair):
    return p.compare_pair(pair[0][5],pair[1][5],pair[0][6],pair[1][6])


def test_physical_equality_with_wall_source_metadata_only(pair):
    r=compare(pair)
    assert r['passed'] and all(r['checks'].values())
    assert r['metadata_excluded']['candidate']['performance.json:/source_sha256']['count']==1


def test_missing_real_execution_hostname_is_rejected(pair):
    for side in pair:side[0].pop('actual_system_hostname')
    assert not compare(pair)['passed']


def test_checkpoint_changing_mid_comparison_is_rejected(pair,monkeypatch):
    original=p.exact._records
    def changed(name,path):
        yield from original(name,path)
        if name=='rt_sys_trace.jsonl':pair[1][5].write_text('{}')
    monkeypatch.setattr(p.exact,'_records',changed)
    assert not compare(pair)['passed']


@pytest.mark.parametrize('field', ['request_id','status','action'])
def test_ack_identity_action_and_status_are_not_normalized(pair,field):
    f=pair[1][3]['control_receipts.jsonl'];rows=[json.loads(x) for x in f.read_text().splitlines()]
    rows[0][field]={'weight':9} if field=='action' else 'changed'
    f.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    assert not compare(pair)['passed']


def test_ack_order_is_exact(pair):
    f=pair[1][3]['control_receipts.jsonl'];f.write_text('\n'.join(reversed(f.read_text().splitlines()))+'\n')
    assert not compare(pair)['passed']


def test_very_small_kpi_difference_is_not_rounded(pair):
    f=pair[1][3]['rt_sys_trace.jsonl'];raw=f.read_text();f.write_text(raw.replace('1.0000000000000002','1.0000000000000003',1))
    assert not compare(pair)['passed']


def test_cache_identity_must_remain_exact(pair):
    f=pair[1][3]['rt_sys_trace.jsonl'];f.write_text(f.read_text().replace('"cache_key": "same"','"cache_key": "different"',1))
    assert not compare(pair)['passed']


def test_missing_absolute_slot_fails(pair):
    f=pair[1][3]['rt_sys_trace.jsonl'];f.write_text('\n'.join(f.read_text().splitlines()[1:])+'\n')
    r=compare(pair);assert not r['passed'] and 'absolute slot' in r['non_metadata_differences'][-1]['error']


def test_inventory_mismatch_fails_even_empty_new_stream(pair):
    f=pair[1][2]/'latency_rf_events.jsonl';f.write_text('');pair[1][3][f.name]=f
    assert not compare(pair)['passed']


def test_decimal_exact_not_python_float_rounding():
    x=p.exact._load('{"x":0.100000000000000001}')
    y=p.exact._load('{"x":0.100000000000000002}')
    assert list(p.exact._differences(x,y))
    assert not list(p.exact._differences(p.exact._load('1.0'),1))
    assert list(p.exact._differences(True,1))


def test_wrong_source_or_incomplete_seal_is_not_read_as_parity(tmp_path):
    plan={'schema':'test','cases':[]};plan['plan_sha256']=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    pp=tmp_path/'plan.json';put(pp,plan);cp=tmp_path/'checkpoint.json';put(cp,{'completed':False})
    with pytest.raises(ValueError,match='Complete nonempty seal'):p.verify(cp,pp,'reference')


def test_no_full_10s_claim():
    v=p.policy();assert v['full_run_parity_claimed'] is False and '2840' in v['physical_scope']
    assert v['unknown_files']=='fail' and v['unknown_fields']=='compare' and v['tolerance']==0


def full_admission_fixture(tmp_path):
    plan_path=tmp_path/'plan.json';root=tmp_path/'worker';cp=root/'checkpoints/job.json';run=root/'jobs/job/data/run'
    case=dict(id='job',arm='OFF',worker_id='agx1',seed=14,origin_key='o',app_ids=[],slots=2840,
              baseline_slot=2040,origin_observe_slots=2000,observe_slots=400)
    models={};hot={};configured=[];release={};argv={}
    make_inventory(run)
    for app,leaf in p.MODEL_FILES.items():
        path=run/'xapp_model_checkpoints'/(app+'.json');put(path,{'model':app});pin=p.sha(path)
        models['artifacts/xapps/chongyu/20261117/models/'+leaf+'.json']=pin
        pathstr='/source/outputs/models/'+leaf+'.json'
        hot[app]=dict(sha256=pin,mode='frozen',checkpoint='xapp_model_checkpoints/'+app+'.json',path=pathstr)
        configured.append(dict(implementation_id=app,parameters=dict(mode='frozen',model_path=pathstr)))
        preview=run/'preview_models/models'/(leaf+'.json');preview.parent.mkdir(parents=True,exist_ok=True);preview.write_bytes(path.read_bytes())
        release[leaf]=dict(implementation_id=app,cli_mode=leaf,sha256=pin)
        argv[leaf]=['--'+leaf+'-xapp-mode','frozen','--'+leaf+'-xapp-model','outputs/preview_models/models/'+leaf+'.json']
    put(run/'preview_models/models.json',{})
    manifest={str(x.relative_to(run/'preview_models')):p.sha(x)for x in(run/'preview_models').rglob('*')if x.is_file()}
    put(run/'preview_models/release.json',dict(files_sha256=manifest,models=release))
    put(run/'preview_models/runtime-arguments.json',argv)
    put(run/'xapp_hot_start.json',dict(models=hot))
    put(run/'config_snapshot.json',dict(source_sha256=p.OLD_SOURCE,manifest=dict(total_slots=2840,rng_seed=14),
        configs=dict(ric=dict(freeze_learning=True,persistent_memory_enabled=False,xapps=configured))))
    put(run/'run_status.json',dict(source_sha256=p.OLD_SOURCE,completed=True,simulation_completed=True,stage='completed',replay=dict(audits_passed=True)))
    plan=dict(worker_id='agx1',source_sha256=p.OLD_SOURCE,source_archive_sha256=p.OLD_ARCHIVE,runner_sha256=p.OLD_RUNNER,frozen_models=models,cases=[case])
    plan['plan_sha256']=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest();put(plan_path,plan)
    origin=dict(source_job_id='job',seed=14,source_sha256=p.OLD_SOURCE,source_archive_sha256=p.OLD_ARCHIVE,
                baseline_slot=2040,observe_slots=2000,model_sha256={app:hot[app]['sha256']for app in hot},after_kpis_used=False)
    origin['origin_id']=hashlib.sha256(json.dumps(origin,sort_keys=True,separators=(',',':')).encode()).hexdigest();put(root/'origins/o.json',origin)
    seal=dict(completed=True,case=case,worker_id='agx1',actual_system_hostname='agx-one',plan_sha256=plan['plan_sha256'],
              source_sha256=p.OLD_SOURCE,source_archive_sha256=p.OLD_ARCHIVE,runner_sha256=p.OLD_RUNNER,
              frozen_models=models,origin_id=origin['origin_id'])
    put(root/'jobs/job/data/request.json',seal)
    seal['files_sha256']={str(x.relative_to(root/'jobs/job')):p.sha(x)for x in(root/'jobs/job').rglob('*')if x.is_file()}
    put(cp,seal)
    return cp,plan_path,root,run


def test_complete_sealed_model_source_admission(tmp_path):
    cp,plan,_,_=full_admission_fixture(tmp_path)
    assert len(p.verify(cp,plan,'reference')[4].model_pins)==6


def test_known_but_unsealed_recording_fails(tmp_path):
    cp,plan,_,run=full_admission_fixture(tmp_path)
    seal=json.loads(cp.read_text());seal['files_sha256'].pop('data/run/rt_sys_trace.jsonl');put(cp,seal)
    with pytest.raises(ValueError,match='inventories differ'):p.verify(cp,plan,'reference')


def test_actual_hostname_not_only_optional_equality(tmp_path):
    cp,plan,_,_=full_admission_fixture(tmp_path)
    seal=json.loads(cp.read_text());seal.pop('actual_system_hostname');put(cp,seal)
    with pytest.raises(ValueError,match='hostname'):p.verify(cp,plan,'reference')


def test_model_binding_cannot_swap_individual_models(tmp_path):
    cp,plan,root,run=full_admission_fixture(tmp_path)
    value=json.loads(plan.read_text());pins=value['frozen_models'];keys=list(pins);pins[keys[0]],pins[keys[1]]=pins[keys[1]],pins[keys[0]]
    value.pop('plan_sha256');value['plan_sha256']=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest();put(plan,value)
    seal=json.loads(cp.read_text());seal['frozen_models']=pins;seal['plan_sha256']=value['plan_sha256'];request=root/'jobs/job/data/request.json';req=json.loads(request.read_text());req['frozen_models']=pins;req['plan_sha256']=value['plan_sha256'];put(request,req);seal['files_sha256']['data/request.json']=p.sha(request);put(cp,seal)
    with pytest.raises(ValueError,match='Specific app-to-model pins'):p.verify(cp,plan,'reference')
