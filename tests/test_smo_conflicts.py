"""Read-only pair analysis using analytic fixtures, never simulation evidence."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from smo.conflict_catalog import action_catalog, catalog_source
from smo.contracts import build_graph
from smo.population import graph_view
from smo.intent_match import match_intent
from smo.intent_language import resolve_ranges
from test_smo_intent import snapshot, interval, graph_with_gains

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which('node') or '/Applications/ChatGPT.app/Contents/Resources/cua_node/bin/node'
APP = ROOT/'smo/web/app.js'


def evaluate(graph, frame, plan, selected=None):
    js = "const fs=require('fs'),c=require('./smo/web/app.js'),i=JSON.parse(fs.readFileSync(0,'utf8'));const s=i.selected===null?c.defaults(i.plan):i.selected;process.stdout.write(JSON.stringify({selected:s,result:c.review(i.graph,i.frame,i.plan,s)}));"
    payload = dict(graph=graph,frame=frame,plan=plan,selected=selected)
    return json.loads(subprocess.run([NODE,'-e',js],input=json.dumps(payload),text=True,capture_output=True,cwd=ROOT,check=True).stdout)


def candidate(graph, key, index):
    app=next(n for n in graph['nodes'] if n.get('key')==key and n['kind']=='xapp')
    profile=action_catalog()['profiles'][app['id']][index]
    return dict(app_id=app['id'],paper_id=app['paper_id'],label=app['label'],variants=[dict(
        id=app['id']+':preview:'+str(index),name=profile['name'],matches=[],side_effects=[],actions=[])])


def plan_for(*candidates, ranges=None):
    return dict(snapshot=dict(slot=1600,time_ms=800),candidates=list(candidates),ranges=ranges or [])


def select(plan):
    return {c['app_id']:c['variants'][0]['id'] for c in plan['candidates']}


def effect(parameter,kpi,direction, evidence='resolved_average',approximate=False):
    return dict(parameter=parameter,kpi=kpi,direction=direction,evidence=evidence,scope_approximation=approximate,
                contrast_id='analytic_fixture',edge_id='influence:'+parameter+':'+kpi)


def primary(c,kpi,direction,paths):
    c['variants'][0]['matches']=[dict(kpi=kpi,direction=direction,policy_goal_aligned=True,paths=paths,opposing_paths=[])]
    return c


def test_browser_catalog_is_generated_from_canonical_released_profiles():
    # Retained solely for auditing old action-space experiments. The active
    # browser UI and graph-only service never load or enumerate these profiles.
    assert catalog_source() in APP.with_name('legacy_action_review.js').read_text()


def test_direct_offload_collision_uses_actual_member_intersection():
    graph=graph_view(build_graph());frame=snapshot()
    plan=plan_for(candidate(graph,'x01',2),candidate(graph,'x13',0))
    before=deepcopy((graph,frame,plan));result=evaluate(graph,frame,plan,select(plan))['result']
    direct=[c for c in result['conflicts'] if c['type']=='direct']
    assert len(direct)==1 and direct[0]['parameters']==['ue_offload_fraction']
    assert [o['entity'] for o in direct[0]['overlaps']]==['ue_4']
    assert direct[0]['status']=='incompatible_writes'
    assert len(direct[0]['edge_ids'])==2
    assert result['deployment_enabled'] is False and before==(graph,frame,plan)


def test_equal_writes_are_compatible_and_disjoint_groups_do_not_collide():
    graph=graph_view(build_graph());frame=snapshot()
    plan=plan_for(candidate(graph,'x01',0),candidate(graph,'x13',0))
    assert evaluate(graph,frame,plan,select(plan))['result']['conflicts']==[]
    # Disjoint priority groups, with a compatible reset to one, do not collide
    # merely because their shared RCP is called scheduling weights.
    a=candidate(graph,'x12_mec',2);b=candidate(graph,'x12_urllc',1)
    result=evaluate(graph,frame,plan_for(a,b),select(plan_for(a,b)))['result']
    assert not any(c['parameters']==['ue_scheduler_weights'] for c in result['conflicts'])
    assert any(c['parameters']==['scheduler_mode'] for c in result['conflicts'])


def test_native_unchanged_reset_writes_still_conflict_with_another_proposal():
    graph=graph_view(build_graph());frame=snapshot()
    # X-08 resets all priorities to one, even though they are already one now.
    # Its delta-only candidate actions would omit this write.
    plan=plan_for(candidate(graph,'x08',0),candidate(graph,'x12_mec',3))
    result=evaluate(graph,frame,plan,select(plan))['result']
    assert any(c['parameters']==['ue_scheduler_weights'] and c['status']=='incompatible_writes' for c in result['conflicts'])
    assert all('ue_drx_configs'not in c['parameters'] for c in result['conflicts'])


def test_runtime_selected_scope_is_possible_overlap_not_certainty():
    graph=graph_view(build_graph());frame=snapshot()
    plan=plan_for(candidate(graph,'x08',2),candidate(graph,'x12_mec',3))
    result=evaluate(graph,frame,plan,select(plan))['result']
    weights=next(c for c in result['conflicts'] if c['parameters']==['ue_scheduler_weights'])
    assert weights['status']=='scope_uncertain'


def test_cross_parameter_adverse_path_marks_potential_indirect_conflict():
    graph=graph_view(build_graph());frame=snapshot()
    a=primary(candidate(graph,'x01',2),'ue_energy_j','decrease',[effect('ue_offload_fraction','ue_energy_j','decrease')])
    b=primary(candidate(graph,'x12_mec',3),'mec_offloaded_latency_ms','decrease',[effect('ue_scheduler_weights','mec_offloaded_latency_ms','decrease')])
    a['variants'][0]['side_effects']=[effect('ue_offload_fraction','mec_offloaded_latency_ms','increase')]
    plan=plan_for(a,b)
    result=evaluate(graph,frame,plan,select(plan))['result']
    indirect=next(c for c in result['conflicts'] if c['type']=='indirect')
    assert indirect['status']=='measured_path' and indirect['kpis']==['mec_offloaded_latency_ms']
    assert 'influence:ue_offload_fraction:mec_offloaded_latency_ms' in indirect['edge_ids']
    assert 'influence:ue_scheduler_weights:mec_offloaded_latency_ms' in indirect['edge_ids']
    assert 'joint-run validation' in result['note']
    a['variants'][0]['side_effects'][0]['evidence']='uncertain_direction'
    assert next(c for c in evaluate(graph,frame,plan,select(plan))['result']['conflicts'] if c['type']=='indirect')['status']=='uncertain_path'


def test_no_indirect_conflict_for_zero_gain_or_a_same_parameter_collision():
    graph=graph_view(build_graph());frame=snapshot()
    a=primary(candidate(graph,'x01',2),'ue_energy_j','decrease',[effect('ue_offload_fraction','ue_energy_j','decrease')])
    b=primary(candidate(graph,'x13',0),'mec_expected_gflops','decrease',[effect('ue_offload_fraction','mec_expected_gflops','decrease')])
    a['variants'][0]['side_effects']=[effect('ue_offload_fraction','mec_expected_gflops','increase')]
    plan=plan_for(a,b)
    assert {c['type'] for c in evaluate(graph,frame,plan,select(plan))['result']['conflicts']}=={'direct'}
    b=primary(candidate(graph,'x12_mec',3),'mec_offloaded_latency_ms','decrease',[effect('ue_scheduler_weights','mec_offloaded_latency_ms','decrease')])
    a['variants'][0]['side_effects']=[effect('ue_offload_fraction','mec_offloaded_latency_ms','increase','no_observed_change')]
    plan=plan_for(a,b)
    assert not evaluate(graph,frame,plan,select(plan))['result']['conflicts']


def test_coupled_offsets_and_scope_mismatch_are_not_resolved_net_conflicts():
    graph=graph_view(build_graph());frame=snapshot()
    a=primary(candidate(graph,'x08',2),'network_goodput_mbps','increase',[effect('scheduler_mode','network_goodput_mbps','increase')])
    b=primary(candidate(graph,'x01',2),'ue_energy_j','decrease',[effect('ue_offload_fraction','ue_energy_j','decrease')])
    a['variants'][0]['side_effects']=[effect('scheduler_mode','ue_energy_j','increase'),effect('ue_scheduler_weights','ue_energy_j','decrease')]
    plan=plan_for(a,b)
    conflict=next(c for c in evaluate(graph,frame,plan,select(plan))['result']['conflicts'] if c['type']=='indirect')
    assert conflict['status']=='uncertain_path' and conflict['coupled_response'] is True
    a['variants'][0]['side_effects']=[effect('scheduler_mode','ue_energy_j','increase',approximate=True)]
    assert next(c for c in evaluate(graph,frame,plan,select(plan))['result']['conflicts'] if c['type']=='indirect')['status']=='uncertain_path'


def test_no_default_selection_and_explicit_reselection_recomputes(tmp_path):
    graph=graph_with_gains(tmp_path);frame=snapshot()
    interpreted=resolve_ranges(dict(ranges=[interval('ue_energy_j'),interval('mec_expected_gflops')],questions=[]),graph,frame)
    plan=match_intent(graph,frame,interpreted)
    evaluated=evaluate(graph,frame,plan)
    assert evaluated['selected']=={} and not evaluated['result']['selected']
    chosen={'xapp:ue_energy_rf','xapp:mec_offload_budget_qgbdt'}
    selection={c['app_id']:c['variants'][0]['id']for c in plan['candidates']if c['app_id']in chosen}
    # Historical action-space audit: explicitly pick incompatible offload
    # profiles instead of relying on today's power-objective candidate ranking.
    selection.update({'xapp:ue_energy_rf':'xapp:ue_energy_rf:preview:2',
                      'xapp:mec_offload_budget_qgbdt':'xapp:mec_offload_budget_qgbdt:preview:0'})
    evaluated=evaluate(graph,frame,plan,selection)
    assert any(c['type']=='direct' for c in evaluated['result']['conflicts'])
    selected={k:v for k,v in evaluated['selected'].items() if k!='xapp:mec_offload_budget_qgbdt'}
    assert not evaluate(graph,frame,plan,selected)['result']['conflicts']
    assert any(r['kpi']=='mec_expected_gflops' for r in evaluated['result']['range_risks'])


@pytest.mark.parametrize('change',['contract','profile','snapshot','selection'])
def test_stale_or_unknown_inputs_fail_closed(change):
    graph=graph_view(build_graph());frame=snapshot();plan=plan_for(candidate(graph,'x01',2),candidate(graph,'x13',0));selected=select(plan)
    if change=='contract': graph['contract_sha256']['fixture']='changed'
    if change=='profile': plan['candidates'][0]['variants'][0]['name']='Changed action'
    if change=='snapshot':frame['slot']=1
    if change=='selection':selected['unknown']='unknown'
    result=evaluate(graph,frame,plan,selected)['result']
    assert result['unavailable'] and not result['conflicts']


def test_physical_domain_bounds_do_not_create_a_spurious_guard_risk():
    graph=graph_view(build_graph());frame=snapshot()
    a=candidate(graph,'x01',2)
    a['variants'][0]['side_effects']=[effect('ue_offload_fraction','ue_energy_j','decrease')]
    guard=dict(kpi='ue_energy_j',status='within_range',needed_direction='hold',current=3.,lower=0.,upper=None,
               lower_inclusive=True,upper_inclusive=True,domain=[0.,None])
    plan=plan_for(a,ranges=[guard])
    assert not evaluate(graph,frame,plan,select(plan))['result']['range_risks']
    guard['lower']=2.
    risk=evaluate(graph,frame,plan,select(plan))['result']['range_risks'][0]
    assert risk['at_boundary'] is False and 'no net bundle' in risk['note']


@pytest.mark.parametrize('delta,band,pressure', [
    (-5,[-6,-4],False),(-15,[-16,-14],True),(1,[.5,1.5],True),
    (-5,[-10,0],True),(-5,[],False),
])
def test_action_pressure_uses_frozen_band_not_opposite_objective_words(delta,band,pressure):
    graph=graph_view(build_graph());frame=snapshot();app=candidate(graph,'x01',2)
    app['variants'][0]['side_effects']=[dict(effect('ue_offload_fraction','network_goodput_mbps','decrease' if delta<0 else 'increase'),delta=delta,delta_interval=band)]
    target=dict(kpi='network_goodput_mbps',lower=20,upper=30,lower_inclusive=False,upper_inclusive=False,
                current=30,status='change_needed',needed_direction='decrease',domain=[0,None])
    result=evaluate(graph,frame,plan_for(app,ranges=[target]),select(plan_for(app)))['result']
    assert bool(result['range_risks']) is pressure


def test_uncalibrated_victim_path_does_not_get_measured_conflict_status():
    graph=graph_view(build_graph());frame=snapshot()
    a=primary(candidate(graph,'x01',2),'ue_energy_j','decrease',[effect('ue_offload_fraction','ue_energy_j','decrease')])
    b=primary(candidate(graph,'x12_mec',3),'mec_offloaded_latency_ms','decrease',[])
    a['variants'][0]['side_effects']=[effect('ue_offload_fraction','mec_offloaded_latency_ms','increase')]
    plan=plan_for(a,b)
    assert next(c for c in evaluate(graph,frame,plan,select(plan))['result']['conflicts'] if c['type']=='indirect')['status']=='uncertain_path'


def display_eval(payload):
    js="const c=require('./smo/web/app.js'),fs=require('fs'),i=JSON.parse(fs.readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(i.edges?i.edges.map(x=>c.parameterKpiDashed(x.edge,x.estimate,x.options)):c.project(i.review,i.show,i.hidden)));"
    return json.loads(subprocess.run([NODE,'-e',js],input=json.dumps(payload),text=True,capture_output=True,cwd=ROOT,check=True).stdout)


def test_parameter_kpi_filter_preserves_all_control_contracts_and_lock_markers():
    assert display_eval(dict(edges=[
        dict(edge=dict(kind='controls')),
        dict(edge=dict(kind='controls',deployment='native_locked')),
        dict(edge=dict(kind='influences')),
        dict(edge=dict(kind='influences'),estimate=dict(confidence_interval=[0,2])),
        dict(edge=dict(kind='influences'),estimate=dict(confidence_interval=[-2,-1])),
        dict(edge=dict(kind='influences'),estimate=dict(confidence_interval=[1,2]),options=dict(uncalibrated=True)),
        dict(edge=dict(kind='controls'),options=dict(uncertain=True)),
    ]))==[False,False,True,True,False,True,False]


def test_solid_view_excludes_uncertain_conflicts_and_hidden_support_paths():
    raw=dict(conflicts=[
        dict(id='direct',type='direct',status='incompatible_writes',edge_ids=['solid']),
        dict(id='uncertain_scope',type='direct',status='scope_uncertain',edge_ids=['solid']),
        dict(id='measured',type='indirect',status='measured_path',edge_ids=['solid']),
        dict(id='uncertain_gain',type='indirect',status='uncertain_path',edge_ids=['solid']),
        dict(id='hidden_support',type='indirect',status='measured_path',edge_ids=['solid','dashed']),
    ],range_risks=[],selected=['preserved'],deployment_enabled=False)
    before=deepcopy(raw)
    visible=display_eval(dict(review=raw,hidden=['dashed']))
    assert [c['id'] for c in visible['conflicts']]==['direct','uncertain_scope','measured']
    assert visible['view']==dict(show_dashed=False) and visible['selected']==['preserved']
    assert raw==before
    restored=display_eval(dict(review=raw,hidden=['dashed'],show=True))
    assert restored['conflicts']==raw['conflicts'] and restored['deployment_enabled'] is False


def test_solid_view_does_not_report_hidden_adverse_paths_as_visible_risks():
    confident=effect('p','k','increase');confident['edge_id']='solid'
    uncertain={**confident,'evidence':'uncertain_direction'}
    scoped={**confident,'scope_approximation':True}
    hidden={**confident,'edge_id':'dashed'}
    raw=dict(conflicts=[],range_risks=[dict(kpi='k',effects=[confident,uncertain,scoped,hidden]),dict(kpi='hidden',effects=[hidden])])
    result=display_eval(dict(review=raw,hidden=['dashed']))
    assert len(result['range_risks'])==1 and result['range_risks'][0]['effects']==[confident]
    assert display_eval(dict(review=raw,hidden=['dashed'],show=True))['range_risks']==raw['range_risks']


def test_direct_conflicts_do_not_depend_on_parameter_kpi_visibility():
    graph=graph_view(build_graph());frame=snapshot()
    plan=plan_for(candidate(graph,'x08',2),candidate(graph,'x12_mec',3))
    raw=evaluate(graph,frame,plan,select(plan))['result']
    assert any(c['status']=='scope_uncertain' for c in raw['conflicts'])
    solid=display_eval(dict(review=raw,hidden=[]))
    all_paths=display_eval(dict(review=raw,hidden=[],show=True))
    assert [c for c in solid['conflicts'] if c['type']=='direct']==[c for c in all_paths['conflicts'] if c['type']=='direct']
