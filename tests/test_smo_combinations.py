"""Exhaustive enumeration against the established full-review oracle.

Analytic fixtures check combinatorics and evidence handling, not simulated KPI
attainment. No physical-network result is inferred from these tests.
"""
from copy import deepcopy
import itertools
import json
import subprocess

import pytest

from smo.contracts import build_graph
from smo.population import graph_view
from test_smo_conflicts import NODE, ROOT, candidate, primary, effect, plan_for
from test_smo_intent import snapshot


def enumerate_plan(graph, frame, plan, *, show=False, compare=False):
    code = r"""
const fs=require('fs'),c=require('./smo/web/app.js'),i=JSON.parse(fs.readFileSync(0,'utf8'));
const before=JSON.stringify(i),result=c.enumerateCombinations(i.graph,i.frame,i.plan,{showDashed:i.show}),differences=[];
if(i.compare)for(const g of result.groups)for(const a of g.assignments){
  const variants=Object.entries(a.selection).map(([id,v])=>i.plan.candidates.find(x=>x.app_id===id).variants.find(x=>x.id===v));
  const raw=c.review(i.graph,i.frame,i.plan,a.selection);
  const hidden=i.graph.edges.filter(e=>c.presentationForVariants(e,variants).dashed).map(e=>e.id);
  const visible=c.project(raw,i.show,hidden);
  const counts=[visible.conflicts.filter(x=>x.type==='direct').length,visible.conflicts.filter(x=>x.type==='indirect').length,visible.range_risks.length];
  if(JSON.stringify(counts)!==JSON.stringify([a.direct_count,a.indirect_count,a.range_risk_count]))differences.push({id:a.id,counts,actual:a});
}
process.stdout.write(JSON.stringify({result,differences,unchanged:before===JSON.stringify(i),defaults:c.defaults(i.plan)}));
"""
    payload = dict(graph=graph, frame=frame, plan=plan, show=show, compare=compare)
    return json.loads(subprocess.run([NODE, '-e', code], input=json.dumps(payload),
        cwd=ROOT, text=True, capture_output=True, check=True).stdout)


def alternative(graph, key, indices, kpi='ue_energy_j', direction='decrease'):
    rows = [primary(candidate(graph, key, index), kpi, direction, []) for index in indices]
    first = rows[0]
    first['variants'] = [row['variants'][0] for row in rows]
    return first


@pytest.mark.parametrize('show', [False, True])
def test_every_nonempty_app_and_action_assignment_matches_full_review(show):
    graph = graph_view(build_graph()); frame = snapshot()
    apps = [alternative(graph, 'x01', [0, 2]), alternative(graph, 'x13', [0, 1]), alternative(graph, 'x08', [1, 2])]
    plan = plan_for(*apps, ranges=[dict(kpi='ue_energy_j', status='change_needed', needed_direction='decrease', current=3, lower=0, upper=3)])
    before = deepcopy(plan)
    checked = enumerate_plan(graph, frame, plan, show=show, compare=True); result = checked['result']
    assert result['group_count'] == 7 and result['assignment_count'] == 26
    assert checked['defaults'] == {} and result['selected_by_default'] is False
    assert checked['unchanged'] and plan == before and not checked['differences']
    actual = {tuple(sorted(a['selection'].items())) for g in result['groups'] for a in g['assignments']}
    expected = {tuple((c['app_id'], v) for c, v in zip(apps, choices) if v is not None)
        for choices in itertools.product(*[[None]+[v['id'] for v in c['variants']] for c in apps]) if any(v is not None for v in choices)}
    expected = {tuple(sorted(x)) for x in expected}
    assert actual == expected and len(actual) == 26
    assert all(len(a['selection']) == len(g['app_ids']) for g in result['groups'] for a in g['assignments'])
    assert result['deployment_enabled'] is False


def test_mutually_exclusive_actions_are_not_merged_into_a_false_full_coverage():
    graph = graph_view(build_graph()); frame = snapshot()
    c = alternative(graph, 'x08', [0, 1], 'network_goodput_mbps', 'increase')
    c['variants'][1]['matches'][0].update(kpi='ue_energy_j', direction='decrease')
    plan = plan_for(c, ranges=[dict(kpi=k, needed_direction=d, status='change_needed', current=1) for k,d in
        [('network_goodput_mbps','increase'),('ue_energy_j','decrease')]])
    result = enumerate_plan(graph, frame, plan)['result']
    assert result['assignment_count'] == 2 and result['group_count'] == 1
    assert result['groups'][0]['full_coverage_count'] == 0
    assert all(not a['full_coverage'] and len(a['covered_kpis']) == 1 for a in result['groups'][0]['assignments'])


def test_dashed_filter_changes_uncertain_indirect_statistics_without_dropping_assignments():
    graph = graph_view(build_graph()); frame = snapshot()
    a = primary(candidate(graph,'x01',2),'ue_energy_j','decrease',[effect('ue_offload_fraction','ue_energy_j','decrease')])
    b = primary(candidate(graph,'x12_mec',3),'mec_offloaded_latency_ms','decrease',[effect('ue_scheduler_weights','mec_offloaded_latency_ms','decrease')])
    a['variants'][0]['side_effects'] = [effect('ue_offload_fraction','mec_offloaded_latency_ms','increase','uncertain_direction')]
    plan = plan_for(a,b, ranges=[dict(kpi=k,needed_direction='decrease',status='change_needed',current=1)for k in ['ue_energy_j','mec_offloaded_latency_ms']])
    solid = enumerate_plan(graph,frame,plan,compare=True)
    dashed = enumerate_plan(graph,frame,plan,show=True,compare=True)
    assert not solid['differences'] and not dashed['differences']
    assert solid['result']['assignment_count'] == dashed['result']['assignment_count'] == 3
    both_solid = next(g for g in solid['result']['groups'] if len(g['app_ids']) == 2)
    both_all = next(g for g in dashed['result']['groups'] if len(g['app_ids']) == 2)
    assert both_solid['indirect_assignments'] == 0 and both_all['indirect_assignments'] == 1
    assert both_solid['assignments'][0]['raw_indirect_count'] == 1
    assert not both_solid['conflict_summaries']
    detail = both_all['conflict_summaries'][0]
    assert detail['type'] == 'indirect'
    assert detail['kpis'] == ['mec_offloaded_latency_ms']
    assert detail['parameters'] == ['ue_offload_fraction']
    assert detail['certain_assignments'] == 0


@pytest.mark.parametrize('left,right,total,certain,dependent', [
    ([0, 1], [3], 2, 2, False),
    ([1], [2, 3], 1, 1, True),
    ([0], [2], 1, 0, False),
])
def test_group_summary_distinguishes_certain_partial_and_scope_conditional_conflicts(left,right,total,certain,dependent):
    graph=graph_view(build_graph());frame=snapshot()
    plan=plan_for(alternative(graph,'x08',left),alternative(graph,'x12_mec',right))
    result=enumerate_plan(graph,frame,plan)['result']
    group=next(g for g in result['groups']if len(g['app_ids'])==2)
    assert len(group['conflict_summaries'])==1
    detail=group['conflict_summaries'][0]
    assert detail['type']=='direct' and detail['assignments']==total
    assert detail['certain_assignments']==certain and detail['action_dependent'] is dependent
    assert detail['parameters'] and detail['apps']==sorted(['xapp:throughput_scheduler_mlp','xapp:latency_scheduler_rf'])


@pytest.mark.parametrize('change',['snapshot','contract','profile'])
def test_stale_or_invalid_context_cannot_generate_a_catalog(change):
    graph = graph_view(build_graph()); frame = snapshot(); c = candidate(graph,'x01',2); plan=plan_for(c)
    if change == 'snapshot': frame['slot'] += 1
    if change == 'contract': graph['contract_sha256']['changed'] = 'invalid'
    if change == 'profile': c['variants'][0]['name'] = 'wrong'
    result = enumerate_plan(graph,frame,plan)['result']
    assert result['unavailable'] and not result['groups'] and not result['assignment_count']


def test_empty_or_contradictory_plan_has_no_enumerated_assignments():
    graph = graph_view(build_graph()); frame=snapshot()
    assert enumerate_plan(graph,frame,plan_for())['result']['assignment_count'] == 0
    plan = plan_for(candidate(graph,'x01',2),ranges=[dict(kpi='ue_energy_j',empty=True,current=3)])
    assert enumerate_plan(graph,frame,plan)['result']['assignment_count'] == 0


@pytest.mark.parametrize('show', [False, True])
def test_multi_kpi_list_requires_all_primary_objectives_and_no_unrelated_apps(show):
    graph = graph_view(build_graph()); frame = snapshot()
    apps = [candidate(graph, key, index) for key, index in
        [('x01', 2), ('x02', 1), ('x08', 1), ('x12_mec', 3), ('x12_urllc', 2)]]
    requests = [('network_goodput_mbps', 'increase'), ('ap_power_w', 'decrease')]
    # Even if every returned preview has favorable paths to both KPIs, only
    # the declared primary owners qualify. This reproduces the old false list.
    for app in apps:
        app['variants'][0]['matches'] = [dict(kpi=k, direction=d,
            policy_goal_aligned=False, paths=[], opposing_paths=[]) for k,d in requests]
    plan = plan_for(*apps, ranges=[dict(kpi=k, needed_direction=d,
        status='change_needed', current=1) for k,d in requests])
    checked = enumerate_plan(graph, frame, plan, show=show, compare=True)
    result = checked['result']
    assert checked['unchanged'] and not checked['differences']
    assert result['group_count'] == result['assignment_count'] == 31
    eligible = [g for g in result['groups'] if g['intent_coverage']['eligible']]
    assert result['eligible_group_count'] == len(eligible) == 1
    x02, x08 = apps[1]['app_id'], apps[2]['app_id']
    assert set(eligible[0]['app_ids']) == {x02, x08}
    coverage = eligible[0]['intent_coverage']
    assert coverage['contributors'] == {'network_goodput_mbps': [x08], 'ap_power_w': [x02]}
    assert coverage['complete'] and not coverage['missing_kpis'] and not coverage['unrelated_apps']
    assert coverage['basis'] == 'kpi_ownership_and_intersected_ranges'
    assert coverage['attainment_verified'] is False and result['deployment_enabled'] is False
    assert all(not g['intent_coverage']['eligible'] for g in result['groups'] if len(g['app_ids']) == 1)
    superset = next(g for g in result['groups'] if set(g['app_ids']) == {apps[0]['app_id'], x02, x08})
    assert superset['intent_coverage']['complete'] and superset['intent_coverage']['unrelated_apps'] == [apps[0]['app_id']]
    incidental = next(g for g in result['groups'] if g['app_ids'] == [apps[0]['app_id']])
    assert incidental['full_coverage_count'] == 1  # Raw variant evidence remains for audit.
    assert not incidental['intent_coverage']['complete']


def test_single_controller_can_cover_its_two_declared_primary_kpis():
    graph = graph_view(build_graph()); frame = snapshot()
    app = candidate(graph, 'x12_urllc', 2)
    plan = plan_for(app, ranges=[dict(kpi=k, needed_direction='decrease',
        status='change_needed', current=1) for k in
        ['urllc_failure_fraction', 'urllc_packet_latency_ms']])
    result = enumerate_plan(graph, frame, plan)['result']
    assert result['eligible_group_count'] == 1
    coverage = result['groups'][0]['intent_coverage']
    assert coverage['eligible'] and all(ids == [app['app_id']] for ids in coverage['contributors'].values())
    assert coverage['attainment_verified'] is False


@pytest.mark.parametrize('kpi,direction,key,index,related', [
    ('network_goodput_mbps', 'decrease', 'x08', 1, True),
    ('ap_power_w', 'increase', 'x02', 1, True),
    ('ap_power_w', 'decrease', 'x01', 2, False),
])
def test_reverse_kpi_relation_does_not_hard_compare_objective_labels(kpi,direction,key,index,related):
    graph = graph_view(build_graph()); frame = snapshot()
    app = primary(candidate(graph,key,index),kpi,direction,[])
    plan = plan_for(app,ranges=[dict(kpi=kpi,needed_direction=direction,status='change_needed',current=1)])
    result = enumerate_plan(graph,frame,plan)['result']
    assert result['eligible_group_count'] == int(related)
    coverage = result['groups'][0]['intent_coverage']
    assert coverage['eligible'] is related
    assert coverage['missing_kpis'] == ([] if related else [kpi])
    assert coverage['attainment_verified'] is False


def test_unknown_controller_cannot_supply_primary_coverage():
    graph = graph_view(build_graph()); frame = snapshot()
    app = primary(candidate(graph,'x08',1),'network_goodput_mbps','increase',[])
    graph['nodes'] = [n for n in graph['nodes'] if n['id'] != app['app_id']]
    plan = plan_for(app,ranges=[dict(kpi='network_goodput_mbps',needed_direction='increase',current=1)])
    result = enumerate_plan(graph,frame,plan)['result']
    assert result['eligible_group_count'] == 0
