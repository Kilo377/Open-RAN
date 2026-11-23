"""Structural graph risks are neither policy actions nor KPI attainment."""
from copy import deepcopy

import pytest

from smo.contracts import build_graph
from smo.graph_combinations import enumerate_combinations
from smo.population import graph_view


def app(name, targets, **extra):
    return dict(id='xapp:'+name,kind='xapp',label=name,paper_id=name,target_kpis=targets,**extra)


def control(name, parameter, **extra):
    return dict(id='control:'+name+':'+parameter,kind='controls',source='xapp:'+name,
                target='rcp:'+parameter,deployment='controllable',**extra)


def influence(parameter, kpi, *, resolved=True, **extra):
    return dict(id='influence:'+parameter+':'+kpi,kind='influences',source='rcp:'+parameter,target='kpi:'+kpi,
        sensitivities=[dict(contrast_id='unselected_tested_context',estimate=1,
            confidence_interval=[.5,1.5] if resolved else [-.5,1.5],population_control_eligible=True,**extra)])


def pair_group(result):
    return next(g for g in result['groups'] if set(g['app_ids'])=={'xapp:A','xapp:B'})


def fixture():
    graph=dict(metric_scope='ue_population.v1',nodes=[app('A',{'energy':'minimize'}),app('B',{'latency':'minimize'})],
        edges=[control('A','p'),control('B','q'),influence('p','energy'),influence('p','latency'),influence('q','latency')])
    frame=dict(slot=1,time_ms=.5,states={'xapp:A':dict(enabled=False),'xapp:B':dict(enabled=True)})
    plan=dict(snapshot=dict(slot=1),ranges=[dict(kpi=k,lower=0,upper=10,lower_inclusive=True,
        upper_inclusive=False,current=10,anchors=[dict(current=10)])for k in ('energy','latency')])
    return graph,frame,plan


def test_related_groups_cover_all_kpis_without_action_profiles_or_model_inference():
    graph,frame,plan=fixture();before=deepcopy((graph,frame,plan))
    graph['nodes'].append(app('C',{'energy':'maximize'}))
    graph['edges'].append(control('C','r'))
    graph['edges'].append(influence('r','energy'))
    graph['nodes'].append(app('unrelated',{'other':'maximize'}))
    graph['edges'].append(control('unrelated','z'))
    original=deepcopy((graph,frame,plan))
    result=enumerate_combinations(graph,frame,plan)
    groups={frozenset(g['app_ids']) for g in result['groups']}
    assert groups=={frozenset(['xapp:A']),frozenset(['xapp:A','xapp:B']),frozenset(['xapp:A','xapp:C']),frozenset(['xapp:C','xapp:B']),frozenset(['xapp:A','xapp:B','xapp:C'])}
    assert result['group_count']==5 and not result['missing_kpis']
    assert all('variants' not in candidate for candidate in result['candidates'])
    assert all('assignments' not in group for group in result['groups'])
    assert result['analysis_basis']=='knowledge_graph'
    assert not result['automatic_selection'] and not result['deployment_enabled'] and not result['runtime_proposals_used']
    assert result['candidates'][0]['enabled'] is False
    assert all(g['intent_coverage']['eligible'] and not g['intent_coverage']['attainment_verified']for g in result['groups'])
    assert (graph,frame,plan)==original
    assert before[2]['ranges']==plan['ranges']


def test_single_multi_objective_app_is_allowed_and_missing_owner_is_explicit():
    graph,frame,plan=fixture()
    graph['nodes']=[app('A',{'energy':'minimize','latency':'minimize'})]
    result=enumerate_combinations(graph,frame,plan)
    assert result['group_count']==1 and result['groups'][0]['app_ids']==['xapp:A']
    assert not result['groups'][0]['raw_conflicts']
    graph['nodes'][0]['target_kpis'].pop('latency')
    result=enumerate_combinations(graph,frame,plan)
    assert result['group_count']==1  # the measured side-effect path still covers latency
    graph['edges']=[edge for edge in graph['edges'] if edge['target']!='kpi:latency']
    result=enumerate_combinations(graph,frame,plan)
    assert result['group_count']==0 and result['missing_kpis']==['latency']


def test_direct_shared_control_is_potential_even_if_declared_action_values_are_equal():
    graph,frame,plan=fixture()
    graph['edges'][1]=control('B','p',declared_values=dict(values=[1]))
    graph['edges'][0]['declared_values']=dict(values=[1])
    result=enumerate_combinations(graph,frame,plan)
    direct=[c for c in pair_group(result)['conflicts']if c['type']=='direct']
    assert len(direct)==1 and direct[0]['status']=='potential_unknown_values'
    assert direct[0]['scope_status']=='possible_overlap'
    assert not direct[0]['action_values_known'] and not direct[0]['actual_conflict_verified']
    graph['edges'][0]['declared_values']['values']=[0,2,10]
    assert enumerate_combinations(graph,frame,plan)['groups']==result['groups']


def test_explicit_disjoint_write_scopes_do_not_collide_but_scheduler_target_labels_do_not_prove_disjointness():
    graph,frame,plan=fixture()
    graph['nodes'][0]['target_ues']=['ue_0'];graph['nodes'][1]['target_ues']=['ue_1']
    graph['edges'][0]=control('A','ue_scheduler_weights')
    graph['edges'][1]=control('B','ue_scheduler_weights')
    graph['edges']=graph['edges'][:2]+[influence('ue_scheduler_weights','energy'),influence('ue_scheduler_weights','latency')]
    result=enumerate_combinations(graph,frame,plan)
    assert any(c['type']=='direct'for c in pair_group(result)['conflicts'])
    for edge,member in zip(graph['edges'][:2],['ue_0','ue_1']):edge['write_scope']=dict(members=[member])
    assert not pair_group(enumerate_combinations(graph,frame,plan))['conflicts']


def test_offload_scope_overlap_uses_explicit_members_and_retains_shared_controls_when_dashed_hidden():
    graph=graph_view(build_graph());frame=dict(slot=1)
    ranges=[dict(kpi=key,lower=0,upper=None,lower_inclusive=True,upper_inclusive=False)for key in ('ue_power_w','mec_expected_gflops')]
    result=enumerate_combinations(graph,frame,ranges)
    group=next(g for g in result['groups']if set(g['app_ids'])=={'xapp:ue_energy_rf','xapp:mec_offload_budget_qgbdt'})
    direct=next(r for r in group['conflicts']if r['type']=='direct')
    assert direct['overlap_members']==['ue_4'] and direct['scope_status']=='known_overlap'
    assert direct['edge_ids']==['control:ue_energy_rf:ue_offload_fraction','control:mec_offload_budget_qgbdt:ue_offload_fraction']
    assert any(r['type']=='indirect'for r in group['conflicts'])
    assert all(not r['conditional_evidence']for r in group['conflicts']if r['type']=='indirect')


def test_indirect_coupling_retains_graph_paths_but_does_not_infer_harmful_direction():
    graph,frame,plan=fixture()
    result=enumerate_combinations(graph,frame,plan)
    risk=next(r for r in pair_group(result)['conflicts']if r['type']=='indirect')
    assert risk['apps']==['xapp:A','xapp:B'] and risk['status']=='structural_coupling'
    assert risk['parameters']==['p','q'] and risk['kpis']==['latency']
    assert not risk['dashed'] and risk['conditional_evidence']
    assert all(not path['action_selected'] for path in risk['paths'])
    assert set(risk['edge_ids'])<=set(edge['id']for edge in graph['edges'])
    assert 'needed_direction'not in risk and 'predicted_kpi'not in risk
    for edge in graph['edges']:
        for row in edge.get('sensitivities',[]):row.update(estimate=-1,confidence_interval=[-1.5,-.5])
    changed=enumerate_combinations(graph,frame,plan)
    assert changed['groups']==result['groups']


@pytest.mark.parametrize('change',['ineligible_population','missing_estimate'])
def test_dashed_filter_only_hides_indirect_relationships_and_never_selects_a_contrast(change):
    graph,frame,plan=fixture()
    edge=graph['edges'][3]
    if change=='ineligible_population':edge['sensitivities'][0]['population_control_eligible']=False
    if change=='missing_estimate':edge['sensitivities']=[]
    graph['edges'].append(control('B','p'))
    solid=enumerate_combinations(graph,frame,plan)
    dashed=enumerate_combinations(graph,frame,plan,show_dashed=True)
    assert solid['group_count']==2 and dashed['group_count']==3
    assert len([c for c in pair_group(solid)['conflicts']if c['type']=='direct'])==1
    assert not [c for c in pair_group(solid)['conflicts']if c['type']=='indirect']
    assert any(c['type']=='indirect'for c in pair_group(dashed)['conflicts'])
    assert pair_group(solid)['raw_conflicts']==pair_group(dashed)['raw_conflicts']


def test_any_resolved_context_makes_overview_relation_visible_without_choosing_that_action():
    graph,frame,plan=fixture()
    graph['edges'][3]['sensitivities'][0]['confidence_interval']=[-.5,1.5]
    graph['edges'][3]['sensitivities'].append(dict(contrast_id='other_context',confidence_interval=[2,3],population_control_eligible=True))
    result=enumerate_combinations(graph,frame,plan)
    risk=next(r for r in pair_group(result)['conflicts']if r['type']=='indirect')
    source=risk['paths'][0]
    assert source['resolved_contrasts']==['other_context']
    assert len(source['tested_contrasts'])==2 and source['action_selected']is False


@pytest.mark.parametrize('change',['empty','questions','snapshot'])
def test_unresolved_or_contradictory_intent_cannot_generate_groups(change):
    graph,frame,plan=fixture()
    if change=='empty':plan['ranges'][0].update(lower=20,upper=10,empty=False)
    if change=='questions':plan['questions']=['Which latency?']
    if change=='snapshot':plan['snapshot']['slot']=2
    result=enumerate_combinations(graph,frame,plan)
    assert result['groups']==[] and result['group_count']==0


def test_locked_controls_are_excluded_without_using_observed_rcp_values_or_profiles():
    graph,frame,plan=fixture()
    graph['edges'][0]['deployment']='native_locked'
    result=enumerate_combinations(graph,frame,plan)
    assert result['missing_kpis']==['energy']
    assert result['excluded_apps']==[dict(app_id='xapp:A',reason='no_unlocked_control_edges')]
    assert not result['groups']
    with pytest.raises(ValueError):enumerate_combinations(graph,frame,[dict(kpi='energy',lower='current',upper=10)])


def test_indirect_controller_without_requested_objective_can_cover_the_intent():
    graph,frame,plan=fixture()
    graph['nodes']=[app('side_effect_only',{'other':'minimize'})]
    graph['edges']=[control('side_effect_only','r'),influence('r','energy'),influence('r','latency')]
    result=enumerate_combinations(graph,frame,plan)
    assert result['groups'][0]['app_ids']==['xapp:side_effect_only']
    candidate=result['candidates'][0]
    assert candidate['owns_kpis']==[] and candidate['related_kpis']==['energy','latency']
    assert len(candidate['relation_paths'])==2
    assert not result['attainment_verified'] and result['selection_algorithm'] is None
    assert not result['ranking_performed'] and not result['automatic_selection']
    graph['edges'][0]['deployment']='native_locked'
    assert not enumerate_combinations(graph,frame,plan)['groups']


def test_measured_uncertain_side_effect_is_coverage_but_zero_is_not_unrelated_coverage():
    graph,frame,plan=fixture()
    graph['nodes']=[app('side_effect_only',{})]
    graph['edges']=[control('side_effect_only','r'),influence('r','energy'),influence('r','latency',resolved=False)]
    assert enumerate_combinations(graph,frame,plan)['group_count']==1
    assert enumerate_combinations(graph,frame,plan,show_dashed=True)['group_count']==1
    graph['edges'][2]['sensitivities'][0].update(estimate=0,confidence_interval=[0,0],zero_in_both=True)
    assert not enumerate_combinations(graph,frame,plan,show_dashed=True)['groups']


def test_zero_current_response_does_not_remove_an_apps_declared_target_path():
    graph,frame,plan=fixture();graph['nodes']=[app('mec',{'energy':'minimize','latency':'minimize'})]
    graph['edges']=[control('mec','p'),influence('p','energy'),influence('p','latency',resolved=False)]
    graph['edges'][-1]['sensitivities'][0].update(estimate=0,confidence_interval=[0,0])
    result=enumerate_combinations(graph,frame,plan)
    assert result['group_count']==1
    path=result['candidates'][0]['relation_paths'][-1]
    assert path['declared_target']and path['observed_zero']
    assert not result['attainment_verified']


def test_all_covering_subsets_are_kept_without_selection_or_minimal_group_filter():
    graph,frame,plan=fixture()
    graph['nodes']=[app(str(i),{}) for i in range(6)]
    graph['edges']=[control(str(i),'p') for i in range(6)]+[influence('p','energy'),influence('p','latency')]
    result=enumerate_combinations(graph,frame,plan)
    assert result['group_count']==63
    assert [len(g['app_ids']) for g in result['groups']].count(1)==6
    assert all(all(g['coverage'].values()) for g in result['groups'])
    assert not result['selected_by_default'] and not result['deployment_enabled']


def test_declared_goal_without_a_reverse_path_does_not_establish_coverage():
    graph,frame,plan=fixture()
    graph['edges']=[e for e in graph['edges'] if e['target']!='kpi:energy']
    result=enumerate_combinations(graph,frame,plan)
    assert result['missing_kpis']==['energy'] and result['groups']==[]
    assert result['kpi_backtrace']['energy']==[]


def test_three_kpi_reverse_trace_enumerates_exact_covering_subsets_with_witnesses():
    related={'A':['k1'],'B':['k2'],'C':['k3'],'D':['k1','k2']}
    graph=dict(nodes=[app(a,{})for a in related],edges=[])
    for a,keys in related.items():
        graph['edges'].append(control(a,'p'+a))
        graph['edges'].extend(influence('p'+a,k) for k in keys)
    ranges=[dict(kpi=k,lower=0,upper=10,lower_inclusive=True,upper_inclusive=True)for k in ('k1','k2','k3')]
    result=enumerate_combinations(graph,dict(slot=1),ranges)
    assert {frozenset(g['app_ids']) for g in result['groups']}=={
        frozenset('xapp:'+a for a in group)for group in ('CD','ABC','ACD','BCD','ABCD')}
    edge_ids={e['id']for e in graph['edges']}
    for group in result['groups']:
        assert group['intent_coverage']['covered_count']==group['intent_coverage']['required_count']==3
        assert set(group['coverage_paths'])=={'k1','k2','k3'}
        for key,rows in group['coverage_paths'].items():
            assert rows and {row['app_id']for row in rows}==set(group['coverage'][key])
            for row in rows:
                assert row['app_id']in group['app_ids'] and row['paths']
                assert all(p['kpi']==key and p['control_edge_id']in edge_ids and p['influence_edge_id']in edge_ids for p in row['paths'])
    assert all(not g['intent_coverage']['attainment_verified'] for g in result['groups'])
