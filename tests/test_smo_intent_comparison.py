"""Analytic scalar traces; these fixtures are not simulated xApp benefits."""
from copy import deepcopy
import json
import pytest
from smo.contracts import build_graph
from smo.population import graph_view
from smo.intent_language import IntentError,resolve_ranges
from smo.intent_comparison import IntentComparison,application_evidence
from smo.window_metrics import WindowIndex


def frame(slot,receipt=None):
    states={'kpi:ue_power_w':dict(value=6.),'kpi:mec_expected_gflops':dict(value=2.),
        'kpi:network_goodput_mbps':dict(value=-999.),
        'xapp:ue_energy_rf':dict(enabled=True,current_proposals=[dict(action=999)])}
    if receipt:states['xapp:ue_energy_rf']['last_applied']=receipt
    return dict(slot=slot,time_ms=slot*.5,states=states)


def scalar_index(source):
    cfg=dict(manifest=dict(slot_duration_ms=.5,ues=[dict(ue_id='u')]),configs=dict(phy=dict(rt_sys=dict(
        ue_services=dict(mec=dict(tasks_by_ue={'u':dict(interval_ms=50,compute_gflop=.1,deadline_ms=100)}))))))
    w=WindowIndex(cfg)
    for slot in range(1,1401):
        power=6. if slot<640 else source.get('after_power',5.5)
        energy=(min(slot,639)*6.+max(0,slot-639)*source.get('after_power',5.5))*.0005
        if source.get('bad_counter')==slot:energy=None
        load=2. if slot<640 else source.get('after_load',1.7)
        services={'u':dict(ue_energy_total_j=energy,ue_service_policy=dict(offload_fraction=load/2))}
        w.observe(slot,[dict(ue_id='u')],services,{})
    return w


def setup(path=None,*,power_upper=5.8,inclusive=True):
    graph=graph_view(build_graph());before=frame(600)
    raw=dict(questions=[],ranges=[dict(kpi='ue_power_w',lower=0,upper=power_upper,lower_inclusive=True,upper_inclusive=inclusive),
        dict(kpi='mec_expected_gflops',lower=0,upper=1.8,lower_inclusive=True,upper_inclusive=True)])
    records=[dict(id='test-intent',text='Fixture only: power and compute bounds',snapshot={'slot':600},ranges=resolve_ranges(raw,graph,before)['ranges'])]
    receipt=dict(status='applied',slot=640,fields=['ue_offload_fraction'])
    source=dict(mode='recorded',name='explicit_test_fixture',frames=[before,frame(640,receipt),frame(900,receipt),frame(1140,receipt),frame(1400,receipt)])
    store=IntentComparison({'fixture':'not_network_data'},path)
    return store,store.start(records,graph,before,500),graph,source,records


def review(store,ident,graph,source,current):
    return store.review(ident,graph,source,current,windows=scalar_index(source))


def test_only_requested_kpis_are_compared_all_required_without_mutation():
    store,ident,graph,source,records=setup();original=deepcopy((source,records))
    r=review(store,ident,graph,source,source['frames'][3])
    assert [x['kpi']for x in r['rows']]==['ue_power_w','mec_expected_gflops']
    assert set(store.sessions[ident]['before']['states'])=={'kpi:ue_power_w','kpi:mec_expected_gflops'}
    assert r['before']['satisfied']==0 and r['after']['satisfied']==2
    assert r['after']['all_satisfied'] and r['intents'][0]['after']['all_satisfied']
    assert (source,records)==original
    assert not r['deployment_enabled'] and not r['selection_performed']
    assert r['quality']['state']=='reached_intent'
    assert r['quality']['improvement_percentage_points']==pytest.approx(10.)


def test_exact_500_slots_include_first_applied_slot_not_preview_or_baseline():
    store,ident,graph,source,_=setup()
    waiting=review(store,ident,graph,source,source['frames'][2])
    assert waiting['status']=='observing' and waiting['remaining_slots']==239
    assert waiting['after']is None and waiting['application_slot']==640
    assert waiting['target_end_slot']==1139
    end=review(store,ident,graph,source,source['frames'][3])
    assert end['status']=='complete' and end['actual_elapsed_slots']==500
    assert end['baseline_window']==dict(window_start_slot=100,slot=600,window_slots=500)
    assert end['after_window']==dict(window_start_slot=639,slot=1139,window_slots=500)
    assert end['rows'][0]['delta']==pytest.approx(-.5)
    assert end['observed_after_slot']==1139 # UI sampled 1140; raw endpoint stays exact.
    assert review(store,ident,graph,source,source['frames'][4])['after']==end['after']


def test_later_apps_are_not_attributed_to_fixed_window():
    store,ident,graph,source,_=setup()
    source['frames'][-1]['states']['xapp:mec_offload_budget_qgbdt']=dict(last_applied=dict(status='applied',slot=1300))
    r=review(store,ident,graph,source,source['frames'][-1])
    assert {x['app_id']for x in r['application_evidence']}=={'xapp:ue_energy_rf'}


def test_enabled_or_previewed_apps_do_not_become_deployment_evidence():
    store,ident,graph,source,_=setup()
    for f in source['frames']:f['states']['xapp:ue_energy_rf'].pop('last_applied',None)
    r=review(store,ident,graph,source,source['frames'][-1])
    assert r['status']=='awaiting_deployment' and r['after']is None
    assert all(x['after']is None and x['delta']is None for x in r['rows'])


def test_one_pass_does_not_satisfy_two_target_intent():
    store,ident,graph,source,_=setup();source['after_load']=2.
    r=review(store,ident,graph,source,source['frames'][3])
    assert r['after']['satisfied']==1 and not r['after']['all_satisfied']
    assert not r['intents'][0]['after']['all_satisfied']
    assert r['quality']['state']=='unchanged' # the worst gap remains 10%.


def test_strict_endpoint_equality_zero_numeric_gap_still_fails():
    _,_,_,source,_=setup()
    exact=scalar_index(source).window(['ue_power_w'],639,1139,through=1140)['states']['kpi:ue_power_w']['value']
    store,ident,graph,source,_=setup(power_upper=exact,inclusive=False)
    r=review(store,ident,graph,source,source['frames'][3])
    assert r['rows'][0]['after']['status']=='above_range'
    assert r['rows'][0]['after']['normalized_gap']==0
    assert r['rows'][0]['after']['strict_boundary_violation']
    assert not r['after']['all_satisfied']


def test_contradictory_intents_report_invalid_constraints():
    store,_,graph,source,records=setup()
    extra=resolve_ranges(dict(questions=[],ranges=[dict(kpi='ue_power_w',lower=6.5,upper=None,lower_inclusive=True,upper_inclusive=True)]),graph,source['frames'][0])['ranges']
    ident=store.start(records+[dict(id='contradiction',text='Analytic conflicting bound',snapshot={'slot':600},ranges=extra)],graph,source['frames'][0],500)
    r=review(store,ident,graph,source,source['frames'][3])
    assert r['status']=='invalid_constraints' and r['rows'][0]['after']['status']=='empty_range'


@pytest.mark.parametrize('bad',[640,900,1139])
def test_missing_scalar_in_window_cannot_be_hidden_by_valid_endpoint(bad):
    store,ident,graph,source,_=setup();source['bad_counter']=bad
    r=review(store,ident,graph,source,source['frames'][3])
    assert r['status']=='complete_with_missing_kpis' and r['after']['valid']==1
    assert r['rows'][0]['delta']is None and not r['quality']['complete']


def test_insufficient_before_history_and_no_index_have_no_point_fallback():
    store,ident,graph,source,_=setup();store.sessions[ident]['before']['slot']=200
    r=review(store,ident,graph,source,source['frames'][3])
    assert r['before']['valid']==0 and not r['quality']['complete']
    r=store.review(ident,graph,source,source['frames'][3])
    assert r['after']['valid']==0
    assert r['rows'][0]['before']['measurement']['reason']=='exact_window_source_unavailable'


def test_future_rejected_unknown_receipts_do_not_count():
    rows=[frame(700,dict(status='rejected',slot=640)),frame(700,dict(status='applied',slot=900))]
    rows[0]['states']['xapp:unregistered']=dict(last_applied=dict(status='applied',slot=640))
    assert application_evidence(rows,600,1400)==[]


def test_archive_survives_reload_frozen_target_and_point_snapshot_unchanged(tmp_path):
    path=tmp_path/'comparisons.json';store,ident,graph,source,_=setup(path)
    r=review(store,ident,graph,source,source['frames'][3]);r['rows'][0]['before']['value']=999
    again=review(IntentComparison({'fixture':'not_network_data'},path),ident,graph,source,source['frames'][-1])
    assert again['rows'][0]['before']['value']==pytest.approx(6.)
    assert again['baseline']['slot']==600 and again['rows'][0]['upper']==5.8
    assert json.loads(path.read_text())['sessions'][ident]['observe_slots']==500


def test_changed_run_id_rejects_cross_run_comparison():
    store,ident,graph,source,_=setup();source['run_id']='another_run'
    with pytest.raises(IntentError,match='运行已变化'):review(store,ident,graph,source,source['frames'][-1])


def test_no_overlapping_point_fraction_or_independent_trial_claim():
    store,ident,graph,source,_=setup();r=review(store,ident,graph,source,source['frames'][3])
    assert 'window_score'not in r
    assert r['interpretation']=='short_term_response_including_initial_transition'


def test_live_bounded_ui_history_keeps_exact_fixed_endpoint():
    store,ident,graph,source,_=setup();source['mode']='live';source['frames']=[source['frames'][3]]
    first=review(store,ident,graph,source,source['frames'][0])
    later=frame(1400,dict(status='applied',slot=1300));source['frames']=[later]
    assert review(store,ident,graph,source,later)['after']==first['after']


@pytest.mark.parametrize('slots',[0,-1,True,1.5,20001])
def test_invalid_budgets_rejected(slots):
    store,_,graph,source,records=setup()
    with pytest.raises(IntentError):store.start(records,graph,source['frames'][0],slots)


def test_http_trusted_records_and_index_no_client_values_or_actions(monkeypatch):
    from http.server import ThreadingHTTPServer
    from threading import Thread
    from urllib.request import Request,urlopen
    from urllib.error import HTTPError
    from smo.__main__ import GraphService,handler
    _,_,_,source,records=setup();service=GraphService();service.source=source;service.window_source=scalar_index(source)
    service.intent_ledger.records={r['id']:deepcopy(r)for r in records}
    def forbidden(*a,**k):raise AssertionError('Comparison attempted model execution')
    monkeypatch.setattr('smo.xapp_preview.preview_one',forbidden)
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(service));thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    def post(path,data):
        return json.load(urlopen(Request(f'http://127.0.0.1:{server.server_port}'+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})))
    try:
        r=post('/api/intent/comparison/start',dict(active_intent_ids=['test-intent'],slot=600,observe_slots=500))
        assert r['status']=='awaiting_deployment'
        after=post('/api/intent/comparison/review',dict(comparison_id=r['id'],slot=1140))
        assert after['after']['all_satisfied']
        for extra in ({'ue_offload_fraction':1},{'after':{'value':0}},{'measurements':{}}):
            with pytest.raises(HTTPError)as error:post('/api/intent/comparison/start',dict(active_intent_ids=['test-intent'],slot=600,**extra))
            assert error.value.code==400
        assert service.intent_ledger.records=={r['id']:r for r in records}
    finally:server.shutdown();server.server_close();thread.join()


def test_browser_renderer_shows_completed_quality_and_missing_reasons(tmp_path):
    import shutil,subprocess
    from pathlib import Path
    node=shutil.which('node')
    if not node:pytest.skip('Node unavailable for standalone renderer check')
    store,ident,graph,source,_=setup()
    complete=review(store,ident,graph,source,source['frames'][3])
    source['bad_counter']=900
    missing=review(store,ident,graph,source,source['frames'][3])
    source.pop('bad_counter')
    waiting=review(store,ident,graph,source,source['frames'][0])
    app=(Path(__file__).resolve().parents[1]/'smo/web/app.js').read_text()
    window_helpers=app[app.index('  const fixedBeforeSlots ='):app.index('  const find =')]
    render=window_helpers+app[app.index('  function quantificationText(row){'):app.index('  function renderMeasuredMember(member){')]+app[app.index('  function resultScoreValue(value){'):app.index('  function renderObservedResult(evaluation){')]+app[app.index('  function renderComparison() {'):app.index('  let decisionPreview=')]
    data=tmp_path/'comparison.json';data.write_text(json.dumps([complete,missing,waiting]))
    script=tmp_path/'render.cjs'
    script.write_text('''const fs=require('fs'), vm=require('vm');
const results=JSON.parse(fs.readFileSync(process.argv[2]));
for(const result of results){
 const elements=new Map();const $=id=>{if(!elements.has(id))elements.set(id,{dataset:{},value:'500',closest:()=>({})});return elements.get(id)};
 const c={$,state:{activeIntents:[{}],bundle:{source:{slot_duration_ms:.5}}},comparisonResult:{...result,reviewed_slot:1140},
 comparisonContext:'fixture',comparisonScope:()=> 'fixture',frame:()=>({slot:1140}),comparisonBusy:false,comparisonError:'',comparisonSession:result.id,
 location:{protocol:'http:'},fmt:(v,n)=>v==null?'—':String(Number(v.toFixed(n))),esc:v=>String(v),
 SmoGraph:{evaluationPresentation:e=>({label:e.no_regression===false?'存在目标退步':e.after.complete?'全部目标达标':'数据不足'})},metricValue:(k,v)=>v==null?'—':String(v),metricUnit:(k,u)=>u,chineseKpis:{},rangeText:r=>String(r.upper),find:()=>null,managedComparison:()=>null,matchingBatch:()=>null,renderDeployment:()=>{}};
 vm.createContext(c);vm.runInContext(RENDER+';renderComparison();',c);
 console.log(JSON.stringify({state:$('comparison-state').textContent,summary:$('comparison-summary').innerHTML,body:$('comparison-body').innerHTML}));
}
'''.replace('RENDER',json.dumps(render)))
    output=subprocess.check_output([node,str(script),str(data)],text=True)
    done,missing_ui,pending=map(json.loads,output.splitlines())
    assert '意图改善分'in done['summary'] and '<strong>+11.67</strong>'in done['summary']
    assert '部署前基准分'in done['summary'] and '<strong>0</strong>'in done['summary']
    assert '剩余违背'in done['body'] and '改善'in done['body']
    assert '计数器缺失或重置'in missing_ui['body'] and '数据不足'in missing_ui['summary']
    assert pending['state']=='尚未部署' and '尚无部署后观测'in pending['body']
