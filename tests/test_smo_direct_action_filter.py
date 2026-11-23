"""Confirmed actuator opposition, not target-direction or gain-sign pruning."""
from copy import deepcopy
from itertools import combinations

from smo.direct_action_filter import filter_catalog
from test_smo_goal_obstruction import fixture


def setup():
    g,f,c,p,a=fixture();p['apps'][1]['actions']=[a('ue_offload_fraction',.3)]
    c.update(groups=[dict(id='a',app_ids=['a']),dict(id='ab',app_ids=['a','b'])],
        reserve_groups=[dict(id='abz',app_ids=['a','b','z'])])
    return g,f,c,p,a


def apply(g,f,c,p):return filter_catalog(c,g,dict(preview=p,frame=f,requested_slot=f['slot'],lag_ms=0))


def test_opposite_same_ue_writes_are_removed_from_primary_and_reserve_without_mutation():
    g,f,c,p,_=setup();original=deepcopy((g,f,c,p));r=apply(g,f,c,p)
    assert [x['id']for x in r['groups']]==['a'] and r['reserve_groups']==[]
    assert {x['id']for x in r['excluded_groups']}=={'ab','abz'}
    pair=r['direct_action_filter']['rejected_pairs'][0]
    assert pair['apps']==['a','b'] and pair['evidence'][0]['parameter']=='ue_offload_fraction'
    assert [x['direction']for x in pair['evidence'][0]['actions']]==['increase','decrease']
    assert (g,f,c,p)==original
    assert not r['direct_action_filter']['observed_outcomes_used']


def test_all_opposed_candidates_can_be_removed_without_a_fallback():
    g,f,c,p,_=setup();c['groups']=c['groups'][1:]
    r=apply(g,f,c,p);assert r['group_count']==0 and not r['groups']


def test_same_direction_different_magnitudes_and_unchanged_do_not_count_as_opposite():
    for value in (.5,.6,.7):
        g,f,c,p,a=setup();p['apps'][1]['actions']=[a('ue_offload_fraction',value)]
        assert len(apply(g,f,c,p)['groups'])==2


def test_different_ues_cells_and_different_parameters_are_not_direct_opposition():
    g,f,c,p,a=setup();p['apps'][1]['actions'][0]['ue_id']='v'
    f['states']['rcp:ue_offload_fraction']['by_entity']['v']=.5
    assert len(apply(g,f,c,p)['groups'])==2
    p['apps'][1]['actions']=[a('ue_drx_configs',dict(a('ue_drx_configs',{})['current'],inactivity_ms=7))]
    assert len(apply(g,f,c,p)['groups'])==2
    p['apps'][1]['actions']=[dict(a('ue_offload_fraction',.3),cell_id='another_cell')]
    r=apply(g,f,c,p);assert len(r['groups'])==2 and r['direct_action_filter']['unresolved_shared_actions']


def test_drx_components_use_own_current_timer_and_never_form_self_conflict():
    g,f,c,p,a=setup();base=a('ue_drx_configs',{})['current']
    p['apps'][0]['actions']=[a('ue_drx_configs',dict(base,inactivity_ms=4))]
    p['apps'][1]['actions']=[a('ue_drx_configs',dict(base,inactivity_ms=12))]
    r=apply(g,f,c,p);assert [x['id']for x in r['groups']]==['a']
    assert r['direct_action_filter']['rejected_pairs'][0]['evidence'][0]['component']=='inactivity_ms'


def test_categorical_disagreement_is_not_an_invented_opposite_direction():
    g,f,c,p,a=setup()
    for app,mode in zip(p['apps'],['latency_aware','qos_deadline']):app['actions']=[a('scheduler_mode',mode)]
    r=apply(g,f,c,p)
    assert len(r['groups'])==2 and not r['direct_action_filter']['rejected_pairs']
    assert r['direct_action_filter']['unresolved_shared_actions'][0]['reason']=='no_ordered_numeric_direction'


def test_unknown_invalid_stale_and_mixed_observations_never_cause_rejection():
    for mode in ('missing','invalid','current_mismatch','stale','mixed'):
        g,f,c,p,_=setup()
        if mode=='missing':p['apps'].pop()
        elif mode=='invalid':p['apps'][1]['action_contract_failures']=['bad']
        elif mode=='current_mismatch':p['apps'][1]['actions'][0]['current']=.9
        elif mode=='stale':p['slot']-=1
        else:
            p['apps'][0]['observation_slot']=98;p['apps'][1]['observation_slot']=99
        assert len(apply(g,f,c,p)['groups'])==2


def test_every_combination_containing_a_confirmed_pair_is_excluded():
    g,f,c,p,_=setup();ids=['a','b','z','w'];c['reserve_groups']=[]
    c['groups']=[dict(id='|'.join(group),app_ids=list(group))for n in range(1,5)for group in combinations(ids,n)]
    r=apply(g,f,c,p)
    assert len(r['excluded_groups'])==4 and len(r['groups'])==11
    assert all(not {'a','b'}<=set(x['app_ids'])for x in r['groups'])


def test_public_review_and_deployment_admission_use_the_same_filter(monkeypatch):
    from smo.__main__ import GraphService
    from smo.intent_language import IntentError
    import pytest
    g,f,_,p,_=setup();s=GraphService();s.graph=g
    g['nodes'] += [dict(id=a,kind='xapp',label=a)for a in ('a','b')]
    s.source=dict(mode='recorded',name='fixture',frames=[f])
    monkeypatch.setattr(s,'intent_context',lambda slot=None:(dict(source=s.source),f))
    monkeypatch.setattr(s,'preview_xapps',lambda slot:p)
    records=[dict(id='saved')]
    monkeypatch.setattr(s.intent_ledger,'select',lambda ids:records)
    monkeypatch.setattr('smo.__main__.intersect_intents',lambda *args:dict(ranges=[dict(kpi='k',lower=10,upper=None,lower_inclusive=False,upper_inclusive=True)],questions=[]))
    # This test uses the real action filter on a small fixed structural catalog.
    monkeypatch.setattr('smo.candidate_selection.enumerate_combinations',lambda *args,**kw:dict(
        candidates=[],ranges=[],groups=[dict(id='a',app_ids=['a']),dict(id='ab',app_ids=['a','b'])]))
    r=s.review_intents(['saved'],100)
    assert [x['id']for x in r['combination_catalog']['groups']]==['a']
    assert r['runtime_proposals_used'] and not r['model_inference_invoked']
    with pytest.raises(IntentError,match='过滤'):s.deploy_combination(['saved'],'ab',100)
    s.evaluation_origin=dict(origin_id='fixture-origin',observe_slots=500)
    f['evaluation_origin']=dict(origin_id='fixture-origin')
    captured={}
    monkeypatch.setattr(s.deployments.batch,'start',lambda **kw:captured.update(kw)or kw)
    s.evaluate_all_combinations(['saved'],100)
    assert [x['id']for x in captured['groups']]==['a']


def test_nearest_earlier_preview_uses_its_own_frame_and_never_a_future_decision(tmp_path):
    import json
    from smo.__main__ import GraphService
    from smo.xapp_preview import PreviewStore
    g,f,c,p,_=setup();requested=deepcopy(f);requested.update(slot=120,time_ms=60)
    g['metric_scope']='ue_population.v1';p.update(schema='oran.xapp-decision-preview.v1',deployment_enabled=False)
    requested['states']['rcp:ue_offload_fraction']['by_entity']['u']=.9
    future=deepcopy(f);future.update(slot=130,time_ms=65)
    later=deepcopy(p);later['slot']=130
    (tmp_path/'xapp_decision_previews.jsonl').write_text('\n'.join(json.dumps(x)for x in(p,later)))
    s=GraphService();s.graph=g;s.preview_store=PreviewStore(run=tmp_path)
    s.source=dict(mode='recorded',name='fixture',frames=[f,requested,future])
    context=s.selection_decision_context(dict(source=s.source),requested)
    assert context['preview']['slot']==100 and context['lag_ms']==10
    assert context['frame']==f
    assert [x['id']for x in filter_catalog(c,g,context)['groups']]==['a']


def test_ui_reports_exclusions_and_preserves_escaped_direction_evidence():
    import json,shutil,subprocess
    from pathlib import Path
    import pytest
    node=shutil.which('node')
    if not node:pytest.skip('Node required for UI renderer')
    g,f,c,p,_=setup();catalog=apply(g,f,c,p)
    app=(Path(__file__).resolve().parents[1]/'smo/web/app.js').read_text()
    fn=app[app.index('  function renderActionExclusions(catalog){'):app.index('  function quantificationText(row){')]
    script="""
const fs=require('fs'),input=JSON.parse(fs.readFileSync(0,'utf8')),nodes={};
const $=id=>nodes[id]??={};const fmt=x=>String(x),time=x=>String(x/2)+' ms';
const esc=x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const find=id=>({label:id==='a'?'A<script>':id});
"""+fn+"\nrenderActionExclusions(input);process.stdout.write(JSON.stringify(nodes));"
    result=json.loads(subprocess.check_output([node,'-e',script],text=True,input=json.dumps(catalog)))
    assert '剔除 2 组'in result['action-filter-note']['textContent']
    assert result['action-exclusions']['hidden']is False
    body=result['action-exclusions-body']['innerHTML']
    assert '↑'in body and '↓'in body and '0.5'in body
    assert 'A&lt;script&gt;'in body and '<script>'not in body
