"""An objective path must survive uncertain or zero calibration, without fiction."""
from copy import deepcopy

from smo.graph_audit import audit_graph


def graph(estimate=2,interval=(-1,5)):
    return dict(metric_scope='ue_population.v1',nodes=[
        dict(id='xapp:mec',kind='xapp',target_kpis={'delay':'minimize'}),
        dict(id='rcp:priority',kind='rcp'),dict(id='kpi:delay',kind='kpi',primary=True)],edges=[
        dict(id='control',source='xapp:mec',target='rcp:priority',kind='controls'),
        dict(id='influence',source='rcp:priority',target='kpi:delay',kind='influences',
             objectives={'mec':'minimize'},sensitivities=[dict(estimate=estimate,
             confidence_interval=list(interval),population_control_eligible=True)])])


def test_uncertain_measured_gain_keeps_a_complete_objective_path():
    g=graph();before=deepcopy(g);result=audit_graph(g)
    assert result['topology_complete']and not result['isolated_nodes']
    assert result['measured_influence_links']==1
    path=result['target_paths'][0]['paths'][0]
    assert not path['dashed']and not path['gain_direction_resolved']
    assert not result['attainment_verified']and g==before


def test_missing_causal_path_is_reported_without_inventing_a_link():
    g=graph();g['edges'].pop();before=deepcopy(g);result=audit_graph(g)
    assert not result['topology_complete']
    assert result['missing_target_paths']==[dict(app_id='xapp:mec',kpi='delay')]
    assert result['isolated_nodes']==['kpi:delay']and g==before


def test_observed_zero_is_visible_but_is_not_labeled_nonzero():
    result=audit_graph(graph(0,(0,0)))
    assert result['topology_complete']and result['observed_zero_links']==1
    assert result['target_paths'][0]['paths'][0]['observed_zero']
    assert result['zero_response_is_not_nonzero_influence']


def test_uncalibrated_target_path_has_no_invented_measurement():
    g=graph();g['edges'][-1]['sensitivities']=[];result=audit_graph(g)
    assert result['topology_complete']and result['declared_only_links']==1
    assert result['measured_influence_links']==0
