"""Significance changes KG admission, never estimates or deployed KPI labels."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from smo.statistical_filter import filter_graph,reason,screening_estimate
from smo.graph_combinations import enumerate_combinations


@pytest.mark.parametrize('value,ci,expected',[(1,[.2,2],None),(-1,[-2,-.2],None),
    (1,[-1,2],'interval_includes_zero'),(1,[0,2],'interval_includes_zero'),
    (0,[0,0],'observed_zero'),(1,None,'confidence_interval_unavailable'),
    (4,[1,2],'estimate_outside_interval')])
def test_existing_pointwise_interval_rule(value,ci,expected):
    assert reason(dict(estimate=value,confidence_interval=ci,confidence_level=.95))==expected


def test_unknown_confidence_and_wrong_population_are_not_evidence_of_significance():
    row=dict(estimate=1,confidence_interval=[.2,2])
    assert reason(row)=='confidence_level_unverified'
    row['confidence_level']=.95
    assert reason(row,population=True)=='population_evidence_unavailable'


def test_relaxation_uses_seed_student_interval_and_preserves_source_evidence():
    # n=3 -> df=2, not a normal interval and not the number of slots/grid points.
    row=dict(estimate=3.,standard_error=1.,sample_count=3,independent_unit='seed_block',
        confidence_level=.95,confidence_interval=[3-4.302652729696142,3+4.302652729696142],
        observed_run_count=15000)
    original=deepcopy(row);relaxed=screening_estimate(row,.90)
    assert reason(row)=='interval_includes_zero'
    assert reason(relaxed,minimum_confidence_level=.90)is None
    assert relaxed['confidence_interval']==pytest.approx([.080014419644484,5.919985580355516])
    assert relaxed['confidence_recalculation']['degrees_of_freedom']==2
    assert relaxed['confidence_recalculation']['source_confidence_interval']==row['confidence_interval']
    assert relaxed['estimate']==row['estimate'] and row==original


@pytest.mark.parametrize('changes',[{'standard_error':None},{'independent_unit':'slot'},
    {'sample_count':15000},{'confidence_interval':[-100,100]},{'sample_count':True}])
def test_unverified_recalculation_cannot_admit_noisy_evidence(changes):
    row=dict(estimate=3.,standard_error=1.,sample_count=3,independent_unit='seed_pair',
        confidence_level=.95,confidence_interval=[3-4.302652729696142,3+4.302652729696142])
    row.update(changes);screened=screening_estimate(row,.90)
    assert screened==row and reason(screened,minimum_confidence_level=.90)=='interval_includes_zero'


def test_four_seed_pair_interval_uses_three_degrees_of_freedom():
    row=dict(estimate=-2.5,standard_error=1.,sample_count=4,independent_unit='seed_pair',
        confidence_level=.95,confidence_interval=[-2.5-3.182446305284263,-2.5+3.182446305284263])
    screened=screening_estimate(row,.90)
    assert screened['confidence_interval']==pytest.approx([-4.853363434801826,-.146636565198174])
    assert screened['confidence_recalculation']['degrees_of_freedom']==3


def test_filter_rejects_unsupported_confidence_instead_of_guessing():
    with pytest.raises(ValueError,match='0.90 or 0.95'):filter_graph({'edges':[]},confidence_level=.5)


def test_mixed_contexts_keep_only_significant_rows_and_control_edges_unchanged():
    from test_smo_graph_combinations import fixture
    graph,_,_=fixture()
    for e in graph['edges']:
        for r in e.get('sensitivities',[]):r['confidence_level']=.95
    e=graph['edges'][2];uncertain=deepcopy(e['sensitivities'][0]);uncertain.update(contrast_id='uncertain',confidence_interval=[-2,2])
    e['sensitivities'].append(uncertain);e['sensitivity']=uncertain
    original=deepcopy(graph);filtered=filter_graph(graph)
    kept=next(x for x in filtered['edges']if x['id']==e['id'])
    assert [r['contrast_id']for r in kept['sensitivities']]==['unselected_tested_context']
    assert kept['sensitivity']==original['edges'][2]['sensitivities'][0]
    assert [x for x in filtered['edges']if x['kind']=='controls']==original['edges'][:2]
    assert filtered['statistical_filter']['excluded_estimate_count']==1 and graph==original


def test_declared_xapp_goal_and_show_dashed_cannot_restore_filtered_paths():
    from test_smo_graph_combinations import fixture
    graph,frame,plan=fixture()
    for edge in graph['edges']:
        for row in edge.get('sensitivities',[]):
            row['confidence_level']=.95
            if edge['target']=='kpi:latency':row['confidence_interval']=[-1,2]
    before=enumerate_combinations(graph,frame,plan)
    assert before['groups']
    filtered=filter_graph(graph)
    for dashed in (False,True):
        result=enumerate_combinations(filtered,frame,plan,show_dashed=dashed)
        assert not result['groups'] and result['missing_kpis']==['latency']
        assert all(p['kpi']!='latency'for c in result['candidates']for p in c['relation_paths'])


def test_service_explicit_policy_applies_to_exported_graph_without_deleting_controls(tmp_path):
    from smo.__main__ import GraphService,export_html
    raw=GraphService().graph
    service=GraphService(influence_selection='significant_only')
    assert service.graph['statistical_filter']['enabled']
    assert not any(e['kind']=='influences'for e in service.graph['edges']) # contracts alone are not statistical evidence
    assert [e for e in service.graph['edges']if e['kind']=='controls']==[e for e in raw['edges']if e['kind']=='controls']
    artifact=export_html(service.bundle(),tmp_path/'graph.html').read_text()
    assert 'pointwise_ci_nonzero_v1'in artifact and 'pk-filter-panel'in artifact


def test_risk_ignores_excluded_relations_without_claiming_zero_physical_effect():
    from test_smo_goal_obstruction import fixture,review
    g,f,c,p,_=fixture()
    for e in g['edges']:
        for r in e.get('sensitivities',[]):r.update(confidence_level=.95)
    drx=next(e for e in g['edges']if e.get('source')=='rcp:ue_drx_configs'and e['kind']=='influences')
    drx['sensitivities'][0].update(estimate=1e6,confidence_interval=[-2e6,2e6])
    filtered=filter_graph(g);first=review(filtered,f,c,p,method='goal_obstruction_v2')
    drx['sensitivities'][0]['estimate']=-1e6
    second=review(filter_graph(g),f,c,p,method='goal_obstruction_v2')
    assert first['groups'][0]['score']==second['groups'][0]['score']
    assert first['influence_scope']=='significant_relations_only'
    assert not first['excluded_relations_are_zero_physical_effects']
    for outcome in first['groups'][0]['actual_outcomes']:
        goal=outcome['goals'][0]
        assert all(c['parameter']!='ue_drx_configs'for c in goal['chains'])
        assert 'ue_drx_configs'in goal['excluded_parameters']


def test_filtered_categorical_context_cannot_borrow_another_sleep_profile():
    edge=dict(kind='influences',source='rcp:cell_sleep_mode',statistical_selection={'method':'test'},sensitivities=[
        dict(kind='finite_contrast',parameter='cell_sleep_mode',contrast_id='micro',members=['c'],
             estimate=2,confidence_interval=[1,3],**{'from':'active','to':'micro_sleep'})])
    js="""const fs=require('fs'),vm=require('vm'),c={structuredClone};vm.createContext(c);
vm.runInContext(fs.readFileSync('smo/web/app.js','utf8'),c);
const e=JSON.parse(fs.readFileSync(0,'utf8'));
process.stdout.write(JSON.stringify(c.SmoConflictRisk.project(e,{c:'active'},{c:'light_sleep'},{states:{}},true)));"""
    result=json.loads(subprocess.check_output(['node','-e',js],input=json.dumps(edge),text=True,cwd=Path(__file__).resolve().parents[1]))
    assert result['drive']is None and not result['calibrations']
