"""Analytic control chains: these numbers are not measured xApp benefits."""
from copy import deepcopy
import json
import pytest

from smo.conflict_quantification import quantify_catalog,cancellation,interval,response,memberships


def fixture():
    base=dict(ue_offload_fraction={'u':.5},ue_drx_configs={'u':dict(enabled=True,cycle_ms=40,on_duration_ms=4,inactivity_ms=8)},
        scheduler_mode={'c':'round_robin'},ue_scheduler_weights={'u':1.},cell_sleep_mode={'c':'active'})
    frame=dict(slot=100,time_ms=50,states={**{'rcp:'+p:dict(by_entity=v)for p,v in base.items()},'kpi:k':dict(value=10.)})
    rows=[]
    def edge(p,gain,component,domain):
        row=dict(parameter=p,kpi='k',population_control_eligible=True,kind='range_linear_fit',
            parameter_component=component,fit_domain=domain,estimate=gain,confidence_interval=[gain,gain],
            contrast_id=p,scoring_window_ms=500,horizon_ms=600)
        if p=='ue_drx_configs':row['scalar_component']='inactivity_ms'
        return dict(id='gain:'+p,source='rcp:'+p,target='kpi:k',kind='influences',sensitivities=[row])
    rows=[edge('ue_offload_fraction',10,'mec_workload',[0,1]),edge('ue_drx_configs',1,'drx_enabled_population',[0,40])]
    for a in('a','b'):
        for p in base:rows.append(dict(id=a+':'+p,kind='controls',source=a,target='rcp:'+p))
    graph=dict(edges=rows)
    def action(p,v):
        return dict(parameter=p,cell_id='c',ue_id='u'if p not in('scheduler_mode','cell_sleep_mode')else None,
            current=deepcopy(base[p]['u'if p not in('scheduler_mode','cell_sleep_mode')else'c']),proposed=v)
    preview=dict(slot=100,apps=[dict(app_id='a',status='proposed',actions=[action('ue_offload_fraction',.7)]),
        dict(app_id='b',status='proposed',actions=[action('ue_drx_configs',dict(base['ue_drx_configs']['u'],inactivity_ms=7))])])
    catalog=dict(required_kpis=['k'],ranges=[dict(kpi='k',lower=10,upper=None,lower_inclusive=False,upper_inclusive=True)],
        groups=[dict(id='ab',app_ids=['a','b'])])
    return graph,frame,catalog,preview,action


def run(g,f,c,p,**kw):return quantify_catalog(g,f,c,p,horizon_ms=500,**kw)


def test_opposition_and_net_interval_effect_without_ranking_or_mutation():
    g,f,c,p,_=fixture();original=deepcopy((g,f,c,p));r=run(g,f,c,p);q=r['groups'][0]
    assert q['direct']==[] and q['indirect'][0]['ratio']==pytest.approx(2/3)
    assert q['indirect'][0]['amount']==1
    assert q['kpis'][0]['predicted_min']==pytest.approx(11)
    assert q['kpis'][0]['violation_probability']==0
    assert q['indirect'][0]['opposition_robust_to_gain_intervals']
    assert not any(r[k]for k in('automatic_selection','ranking_performed','pruning_performed','deployment_enabled'))
    assert (g,f,c,p)==original
    json.dumps(r,allow_nan=False)


def test_same_direction_and_tiny_opposition_are_distinguished():
    assert cancellation(2,3)==dict(ratio=0,amount=0)
    assert cancellation(.001,-.001)==dict(ratio=1,amount=.001)
    assert cancellation(None,1)==dict(ratio=None,amount=None)


def test_direct_actions_are_arbitrated_not_added_and_equal_writes_are_compatible():
    g,f,c,p,a=fixture();p['apps'][1]['actions']=[a('ue_offload_fraction',.3)]
    q=run(g,f,c,p)['groups'][0]
    assert len(q['direct'])==1 and not q['indirect']
    assert q['scenario_count']==2
    assert q['kpis'][0]['predicted_min']==pytest.approx(8)
    assert q['kpis'][0]['predicted_max']==pytest.approx(12)
    assert q['kpis'][0]['violation_probability']==.5
    p['apps'][1]['actions']=[a('ue_offload_fraction',.7)]
    q=run(g,f,c,p)['groups'][0]
    assert q['direct']==[] and q['compatible_shared_resources']==1
    assert q['scenario_count']==1 and q['kpis'][0]['predicted_max']==pytest.approx(12) # not 14


def test_unequal_source_counts_preserve_uniform_per_app_probabilities():
    g,f,c,p,a=fixture();p['apps'][1]['actions']=[a('ue_offload_fraction',.3)]
    p['apps'].append(dict(app_id='z',status='proposed',actions=[a('ue_offload_fraction',.7)]))
    g['edges'].append(dict(id='z:off',kind='controls',source='z',target='rcp:ue_offload_fraction'))
    c['groups'][0]['app_ids'].append('z')
    q=run(g,f,c,p)['groups'][0]
    assert q['scenario_count']==2
    assert q['kpis'][0]['violation_probability']==pytest.approx(1/3)


def test_no_self_conflict_and_internal_chains_are_aggregated():
    g,f,c,p,a=fixture();p['apps'][0]['actions'].append(p['apps'][1]['actions'][0])
    p['apps'][1]['actions']=[a('ue_drx_configs',dict(a('ue_drx_configs',{})['current'],inactivity_ms=9))]
    c['groups']=[dict(id='a',app_ids=['a'])]
    r=run(g,f,c,p)
    assert not r['groups'][0]['direct']and not r['groups'][0]['indirect']
    assert r['apps'][0]['effects'][0]['delta']==pytest.approx(1)


def test_partial_population_step_is_unavailable_not_divided_by_ue_count():
    g,f,c,p,a=fixture();f['states']['rcp:ue_offload_fraction']['by_entity']['v']=.5
    q=run(g,f,c,p)['groups'][0]
    assert q['kpis'][0]['predicted_min']is None and q['kpis'][0]['violation_probability']is None
    assert 'nonuniform_population_action'in q['kpis'][0]['issues']


def test_disjoint_ue_writes_do_not_create_direct_conflict():
    g,f,c,p,a=fixture();f['states']['rcp:ue_offload_fraction']['by_entity']['v']=.5
    p['apps'][1]['actions']=[dict(a('ue_offload_fraction',.3),ue_id='v')]
    assert run(g,f,c,p)['groups'][0]['direct']==[]


def test_categorical_steps_are_named_contrasts_and_scheduler_is_atomic():
    g,f,c,p,a=fixture()
    g['edges'].append(dict(id='mode',kind='influences',source='rcp:scheduler_mode',target='kpi:k',sensitivities=[
        dict(parameter='scheduler_mode',population_control_eligible=True,kind='finite_contrast',
             parameter_component='all_cells',estimate=2,confidence_interval=[1,3],scoring_window_ms=500,**{'from':'round_robin','to':'latency_aware'})]))
    p['apps'][0]['actions']=[a('scheduler_mode','latency_aware'),a('ue_scheduler_weights',1)]
    p['apps'][1]['actions']=[a('scheduler_mode','round_robin'),a('ue_scheduler_weights',1)]
    q=run(g,f,c,p)['groups'][0]
    assert len(q['direct'])==1 and q['direct'][0]['resource']==['scheduler_bundle','c']
    assert q['scenario_count']==2 # never four independent mode/weight selections
    assert q['kpis'][0]['predicted_max']==12 and q['kpis'][0]['violation_probability']==.5 # equality fails strict increase
    p['apps'][0]['actions'][0]['proposed']='light_sleep'
    assert run(g,f,c,p)['groups'][0]['kpis'][0]['predicted_min']is None


def test_no_new_action_or_missing_gain_is_not_assumed_zero():
    g,f,c,p,_=fixture();p['apps'][0].update(status='no_action',actions=[])
    q=run(g,f,c,p)['groups'][0]
    assert q['missing_proposal_apps']==['a']and q['kpis'][0]['violation_probability']is None
    g,f,c,p,_=fixture();g['edges']=[e for e in g['edges']if e['id']!='gain:ue_offload_fraction']
    assert run(g,f,c,p)['groups'][0]['kpis'][0]['predicted_min']is None


def test_uncertainty_visibility_domain_and_horizon_are_explicit():
    g,f,c,p,_=fixture();g['edges'][0]['sensitivities'][0]['confidence_interval']=[-1,12]
    assert run(g,f,c,p)['groups'][0]['kpis'][0]['predicted_min']is None
    q=run(g,f,c,p,show_dashed=True)['groups'][0]
    assert not q['indirect'][0]['opposition_robust_to_gain_intervals']
    assert 'gain_direction_uncertain'in q['kpis'][0]['issues']
    g,f,c,p,_=fixture()
    r=quantify_catalog(g,f,c,p,horizon_ms=250)
    assert 'response_window_mismatch'in r['groups'][0]['kpis'][0]['issues']
    assert r['groups'][0]['kpis'][0]['predicted_max']==pytest.approx(11) # no arbitrary time scaling
    p['apps'][0]['actions'][0]['proposed']=1.1
    assert run(g,f,c,p)['groups'][0]['kpis'][0]['predicted_min']is None


def test_preview_freshness_and_input_readings_are_verified():
    g,f,c,p,_=fixture();p['slot']=99
    assert run(g,f,c,p)['status']=='unavailable'
    p['slot']=100;p['apps'][0]['actions'][0]['current']=.4
    assert run(g,f,c,p)['reason']=='parameter_observation_mismatch'


def test_missing_compact_graph_reading_can_use_actual_captured_public_reading():
    g,f,c,p,_=fixture();del f['states']['rcp:ue_offload_fraction']['by_entity']['u']
    r=run(g,f,c,p)
    assert r['supplemented_parameter_readings']==[dict(parameter='ue_offload_fraction',entity='u',source='captured_public_report')]
    assert r['groups'][0]['kpis'][0]['predicted_max']==pytest.approx(11)


def test_manifest_prevents_missing_cell_from_shrinking_population_calibration():
    g,f,c,p,_=fixture()
    m=memberships(f,dict(manifest=dict(cells=[dict(cell_id='c'),dict(cell_id='unobserved')])) )
    assert m['cells']==['c','unobserved']


def test_scenario_limit_reports_unknown_rather_than_truncating_probability():
    g,f,c,p,a=fixture();p['apps'][1]['actions']=[a('ue_offload_fraction',.3)]
    q=run(g,f,c,p,max_scenarios=1)['groups'][0]
    assert q['scenario_count']==2 and not q['scenario_enumeration_complete']
    assert q['kpis'][0]['violation_probability']is None


def test_both_interval_boundaries_and_strict_equality_are_checked():
    row=dict(lower=10,upper=20,lower_inclusive=False,upper_inclusive=True)
    assert interval(10,row)['status']=='outside'
    assert interval(20,row)['status']=='inside'
    assert interval(21,row)['gap']==1


def test_unsupported_shared_resource_is_not_given_fabricated_random_mitigation():
    g,f,c,p,a=fixture();p['apps'][0]['actions']=[a('ue_drx_configs',dict(a('ue_drx_configs',{})['current'],inactivity_ms=9))]
    q=run(g,f,c,p)['groups'][0]
    assert q['direct']and q['direct'][0]['mitigation_supported']is False
    assert q['direct'][0]['candidates'][0]['probability']is None
    assert q['kpis'][0]['violation_probability']is None
    assert 'unsupported_resource_arbitration'in q['kpis'][0]['issues']


def test_partial_atomic_scheduler_bundle_is_not_treated_as_executable():
    g,f,c,p,a=fixture();p['apps'][0]['actions']=[a('scheduler_mode','latency_aware')]
    assert run(g,f,c,p)['groups'][0]['missing_proposal_apps']==['a']


def test_current_method_refuses_a_different_execution_arbiter():
    g,f,c,p,_=fixture()
    r=run(g,f,c,p,config=dict(configs=dict(ric=dict(direct_conflict_mitigation='priority'))))
    assert r['reason']=='unsupported_mitigation_mode'


def test_real_quantification_renderer_distinguishes_model_risk_and_observation():
    import shutil,subprocess
    from pathlib import Path
    node=shutil.which('node')
    if not node:pytest.skip('Node required for renderer check')
    g,f,c,p,a=fixture();p['apps'][1]['actions']=[a('ue_offload_fraction',.3)]
    source=(Path(__file__).resolve().parents[1]/'smo/web/app.js').read_text()
    fn=source[source.index('  function renderGroupQuantification(group){'):source.index('  function renderCombinationCatalog() {')]
    script='''const vm=require('vm');const input=DATA;
const context={structuredClone,
 conflictQuantificationBusy:false,quantificationReasons:{},chineseKpis:{},find:id=>({label:id,unit:'W'}),
 matchingBatch:()=>null,fmt:(v,n)=>String(Number(v.toFixed(n))),esc:v=>String(v).replaceAll('<','&lt;'),rangeText:()=>'(10, ∞)'};
vm.createContext(context);vm.runInContext(SOURCE,context);
const data=context.SmoConflictRisk.review(input.g,input.f,input.c.groups,input.c.ranges,input.p);
context.quantifiedGroup=()=>data.groups[0];context.currentQuantification=()=>data;
vm.runInContext(FN+';result=renderGroupQuantification(dataGroup)',Object.assign(context,{dataGroup:data.groups[0]}));
console.log(context.result);
'''.replace('DATA',json.dumps(dict(g=g,f=f,c=c,p=p))).replace('SOURCE',json.dumps(source)).replace('FN',json.dumps(fn))
    html=subprocess.check_output([node],input=script,text=True)
    assert '综合冲突风险' in html and '<strong>100</strong>'in html
    assert '不是 KPI 变化预测、达标程度或失败概率'in html and '量化详情'in html
    summary=html.split('<details>')[0]
    assert '直接动作分歧'not in summary and '已知方向对抗'not in summary
    assert '意图加权控制动力对抗'in summary and '直接冲突：'not in summary and '间接冲突：'not in summary
    assert '无法估计净效果'not in html and '仲裁越界占比'not in html and '当前 10 → 预测'not in html


def test_replay_quantification_uses_exact_earlier_decision_without_touching_deployment():
    import shutil,subprocess
    from pathlib import Path
    node=shutil.which('node')
    if not node:pytest.skip('Node required for replay alignment check')
    source=(Path(__file__).resolve().parents[1]/'smo/web/app.js').read_text()
    fn=source[source.index('  async function requestConflictQuantification('):source.index('  function ensureConflictQuantification(')]
    script='''const assert=require('assert');
FN
(async()=>{
 const request={slot:1420,active_intent_ids:['frozen'],observe_slots:500,show_dashed:false};
 const frames=[{slot:1400,time_ms:700},{slot:1420,time_ms:710}];
 const unavailable={source_id:'owned-run',reason:'matching_decision_preview_unavailable',suggested_slot:1400};
 const valid={source_id:'owned-run',preview_slot:1400,snapshot:{slot:1400,time_ms:700},groups:[{id:'ab'}]};
 const calls=[];const post=async(path,data)=>{assert.equal(path,'./api/conflicts/quantify');calls.push({...data});return calls.length===1?unavailable:valid;};
 const result=await requestConflictQuantification(request,frames,post);
 assert.deepEqual(calls,[request,{...request,slot:1400}]);
 assert.equal(request.slot,1420);assert.equal(result.requested_snapshot.slot,1420);assert.equal(result.snapshot.slot,1400);
 assert.equal(result.snapshot_alignment.lag_ms,10);assert.equal(result.snapshot_alignment.interpolation_performed,false);
 assert.deepEqual(result.groups,valid.groups);
 for(const suggestion of [1440,null,1399]){
   let n=0;await requestConflictQuantification(request,frames,async()=>{n++;return {...unavailable,suggested_slot:suggestion};});assert.equal(n,1);
 }
 for(const bad of [{...valid,source_id:'different-run'},{...valid,preview_slot:1420},{...valid,snapshot:{slot:1420}}]){
   let n=0;const r=await requestConflictQuantification(request,frames,async()=>++n===1?unavailable:bad);assert.equal(r,unavailable);
 }
 let n=0;const exact=await requestConflictQuantification({...request,slot:1400},frames,async()=>{n++;return valid;});
 assert.equal(n,1);assert.equal(exact,valid);
})().catch(e=>{console.error(e);process.exitCode=1;});
'''.replace('FN',fn)
    subprocess.run([node],input=script,text=True,check=True,capture_output=True)


def test_http_consumes_only_trusted_saved_intents_graph_and_recorded_proposals(monkeypatch):
    from http.server import ThreadingHTTPServer
    from threading import Thread
    from urllib.request import Request,urlopen
    from urllib.error import HTTPError
    from smo.__main__ import GraphService,handler
    g,f,c,p,_=fixture();service=GraphService()
    service.graph=g
    monkeypatch.setattr(service,'intent_context',lambda slot:(dict(source=dict(name='fixture',frames=[f],slot_duration_ms=.5)),f))
    monkeypatch.setattr(service.intent_ledger,'select',lambda ids:[dict(id='saved')])
    monkeypatch.setattr('smo.__main__.intersect_intents',lambda *args:{})
    monkeypatch.setattr('smo.__main__.graph_intent_review',lambda *args:dict(combination_catalog=c,combination_catalog_with_uncertain=c))
    monkeypatch.setattr(service,'preview_xapps',lambda slot:p)
    monkeypatch.setattr('smo.xapp_preview.preview_one',lambda *args:pytest.fail('must not invoke a model'))
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(service));thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    def post(data,origin=None):
        headers={'Content-Type':'application/json'}
        if origin:headers['Origin']=origin
        return urlopen(Request(f'http://127.0.0.1:{server.server_port}/api/conflicts/quantify',
            data=json.dumps(data).encode(),headers=headers))
    try:
        request=dict(active_intent_ids=['saved'],slot=100,observe_slots=1000)
        result=json.load(post(request))
        assert result['method']=='proposal_first_order_v1'and len(result['groups'])==1
        assert not result['deployment_enabled']
        for extra in({'actions':[]},{'gains':[99]},{'group_ids':['ab']}):
            with pytest.raises(HTTPError)as error:post({**request,**extra})
            assert error.value.code==400
        with pytest.raises(HTTPError)as error:post(request,'https://unrelated.example')
        assert error.value.code==403
    finally:server.shutdown();server.server_close();thread.join()
