"""Numerical path coverage is not direction certainty or achieved performance."""
from copy import deepcopy
import math

import pytest

from smo.gain_audit import audit_gains
from smo.influence_units import align_graph


def graph(estimate=-1.,ci=(-3.,1.)):
    return align_graph(dict(metric_scope='ue_population.v1',nodes=[
        dict(id='xapp:power',kind='xapp',target_kpis={'power':'minimize'}),
        dict(id='rcp:ue_offload_fraction',kind='rcp'),
        dict(id='kpi:power',kind='kpi',primary=True,unit='W')],edges=[
        dict(id='control',source='xapp:power',target='rcp:ue_offload_fraction',kind='controls'),
        dict(id='gain',source='rcp:ue_offload_fraction',target='kpi:power',kind='influences',
            objectives={'power':'minimize'},sensitivities=[dict(parameter='ue_offload_fraction',kpi='power',
                contrast_id='offload',kind='range_linear_fit',unit='W / fraction',x_unit='fraction',
                estimate=estimate,confidence_interval=list(ci),confidence_level=.9,
                fit_domain=[0,1],members=['u'],population_control_eligible=True)])]))


def test_uncertain_finite_gain_is_a_complete_numeric_path_without_claiming_effectiveness():
    g=graph();before=deepcopy(g);a=audit_gains(g)
    assert a['all_link_gains_complete'] and a['visible_link_gains_complete'] and a['topology_complete']
    assert a['link_count']==a['valid_link_count']==a['visible_influence_links']==1
    assert a['uncertain_estimate_count']==1 and a['valid_estimate_count']==1
    assert a['edges'][0]['rows'][0]['evidence']=='uncertain_direction'
    assert not a['attainment_verified'] and not a['deployment_results_used'] and not a['action_applicability_checked']
    assert g==before and not a['graph_modified']


def test_zero_measurement_remains_zero_with_numeric_path_and_no_nonzero_claim():
    a=audit_gains(graph(0.,(0.,0.)))
    assert a['all_link_gains_complete'] and a['topology_complete']
    assert a['observed_zero_estimate_count']==1 and a['uncertain_estimate_count']==0
    assert a['observed_zero_is_not_nonzero_influence']


def test_declared_target_without_measurement_does_not_count_as_a_numeric_gain():
    g=graph();g['edges'][-1]['sensitivities']=[]
    a=audit_gains(g)
    assert not a['all_link_gains_complete'] and not a['visible_link_gains_complete']
    assert a['invalid_edge_ids']==a['visible_invalid_edge_ids']==['gain']
    assert a['edges'][0]['reasons']==['gain_unavailable']
    assert not a['topology_complete']
    assert a['missing_target_paths']==[dict(app_id='xapp:power',kpi='power')]
    assert a['isolated_nodes']==['kpi:power']


@pytest.mark.parametrize('estimate',[None,True,math.inf,math.nan])
def test_non_numeric_coefficients_are_invalid_even_if_the_old_unit_proof_exists(estimate):
    g=graph();g['edges'][-1]['sensitivities'][0]['estimate']=estimate
    a=audit_gains(g)
    assert a['invalid_edge_ids']==['gain'] and a['valid_estimate_count']==0
    assert 'non_finite_scalar_gain'in a['edges'][0]['reasons']


@pytest.mark.parametrize('change',[
    lambda r:r.pop('projection_units'),
    lambda r:r.update(unit='ms / fraction'),
    lambda r:r.update(estimate=-2.),
    lambda r:r['projection_units']['fields'].update(estimate=-1000.),
])
def test_missing_incompatible_and_stale_unit_contracts_cannot_count_as_valid_gains(change):
    g=graph();change(g['edges'][-1]['sensitivities'][0])
    a=audit_gains(g)
    assert not a['all_link_gains_complete'] and a['invalid_edge_ids']==['gain']
    assert not a['topology_complete']


def test_recomputed_canonical_coordinates_accept_real_kpi_unit_conversion():
    g=graph();g['nodes'][-1]['unit']='mW';g=align_graph(g)
    assert audit_gains(g)['all_link_gains_complete']
    # Merely relabeling the KPI after alignment leaves a stale contract.
    g['nodes'][-1]['unit']='W'
    assert audit_gains(g)['invalid_edge_ids']==['gain']


def test_auxiliary_gain_failure_is_reported_without_misreporting_the_default_visible_scope():
    g=graph();g['nodes'].append(dict(id='kpi:aux',kind='kpi',unit='W',primary=False))
    bad=deepcopy(g['edges'][-1]);bad.update(id='aux',target='kpi:aux',sensitivities=[]);g['edges'].append(bad)
    a=audit_gains(g)
    assert not a['all_link_gains_complete'] and a['visible_link_gains_complete']
    assert a['invalid_edge_ids']==['aux'] and a['visible_invalid_edge_ids']==[]
    assert audit_gains(g,all_kpis=True)['visible_invalid_edge_ids']==['aux']


def test_population_gain_cannot_be_backfilled_from_an_ineligible_entity_measurement():
    g=graph();g['edges'][-1]['sensitivities'][0]['population_control_eligible']=False
    a=audit_gains(g)
    assert a['invalid_edge_ids']==['gain']
    assert 'population_evidence_unavailable'in a['edges'][0]['reasons']


def test_malformed_or_misdirected_links_are_reported_without_creating_target_paths():
    g=graph();g['edges'][-1]['sensitivities']=[None]
    a=audit_gains(g)
    assert a['invalid_edge_ids']==['gain'] and not a['topology_complete']
    g=graph();g['edges'][-1]['source']='xapp:power'
    a=audit_gains(g)
    assert 'invalid_parameter_kpi_endpoints'in a['edges'][0]['reasons']
    assert a['invalid_edge_ids']==['gain']


def test_a_topological_path_with_wrong_parameter_or_kpi_provenance_is_not_accepted():
    for field,value in [('parameter','ue_scheduler_weights'),('kpi','other')]:
        g=graph();g['edges'][-1]['sensitivities'][0][field]=value
        a=audit_gains(g)
        assert a['invalid_edge_ids']==['gain'] and not a['topology_complete']


def test_missing_uncertainty_is_distinguished_from_a_missing_numeric_gain():
    g=graph();g['edges'][-1]['sensitivities'][0].pop('confidence_interval');g=align_graph(g)
    a=audit_gains(g)
    assert a['all_link_gains_complete'] and a['unknown_uncertainty_count']==1
    assert a['uncertain_estimate_count']==0
