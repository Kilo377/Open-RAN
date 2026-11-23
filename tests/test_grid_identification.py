"""Known analytic models validate the range estimator; fixtures are not run data."""
from copy import deepcopy
import json

import pytest

from smo.grid_identify import fit_grid,line_fit
from smo.identify import paired_summary
from smo.contracts import build_graph,validate_influences


def setup_model(fn):
    levels=[0.,.2,.4,.5,.6,.8,1.];seeds=[1,2,3]
    g=dict(id='offload_range_fixture',parameter='ue_offload_fraction',field='offload_probability',reference=.5,
        levels=levels,arms=[str(x)for x in levels],mode='round_robin',component='mec_workload',x_unit='fraction',scope='Analytic test fixture only')
    lookup={(s,str(x)):dict(metrics=dict(ue_energy_j=fn(s,x)))for s in seeds for x in levels}
    return g,seeds,lookup


def test_multi_point_fit_recovers_slope_despite_large_seed_intercepts():
    g,seeds,lookup=setup_model(lambda s,x:s*100+2*x)
    r=fit_grid(g,seeds,lookup,'ue_energy_j','a'*64)
    assert r['estimate']==pytest.approx(2.)
    assert r['confidence_interval']==pytest.approx([2.,2.])
    assert r['sample_count']==3 and r['fit_point_count']==7 and r['observed_run_count']==21
    assert r['intercept_at_reference']==pytest.approx(201.)
    assert r['fit_diagnostics']['within_seed_r2']==pytest.approx(1.)
    assert r['fit_diagnostics']['leave_level_out_rmse']<1e-10


def test_uncertainty_counts_seed_slopes_not_individual_grid_points():
    g,seeds,lookup=setup_model(lambda s,x:100*s+s*x)
    r=fit_grid(g,seeds,lookup,'ue_energy_j','a'*64)
    assert r['estimate']==pytest.approx(2.)
    assert r['confidence_interval']==pytest.approx(paired_summary([1.,2.,3.])['confidence_interval'])
    assert r['independent_unit']=='seed_block'


def test_missing_endpoint_stays_missing_and_is_not_fitted_as_zero():
    g,seeds,lookup=setup_model(lambda s,x:None if x==0 else 100*s+2*x)
    r=fit_grid(g,seeds,lookup,'ue_energy_j','a'*64)
    assert r['estimate']==pytest.approx(2.) and r['fit_point_count']==6
    assert r['sampled_domain']==[0.,1.]and r['fit_domain']==[.2,1.]
    assert r['excluded_levels']==[0.]
    assert r['grid_points'][0]['mean']is None and r['grid_points'][0]['used_in_fit']is False
    for x in [.2,.4,.5]:
        for s in seeds:lookup[s,str(x)]['metrics']['ue_energy_j']=None
    assert fit_grid(g,seeds,lookup,'ue_energy_j','a'*64)is None


def test_precise_nonzero_average_slope_does_not_hide_nonlinearity():
    g,seeds,lookup=setup_model(lambda s,x:100*s+100*(x-.5)**2+2*x)
    r=fit_grid(g,seeds,lookup,'ue_energy_j','a'*64)
    assert r['estimate']==pytest.approx(2.)
    assert r['confidence_interval'][0]>0
    assert r['fit_diagnostics']['assessment']=='limited_first_order_approximation'
    assert r['fit_diagnostics']['usable_for_linear_control']is False


def test_constant_observed_response_is_explicit():
    g,seeds,lookup=setup_model(lambda s,x:100.*s)
    r=fit_grid(g,seeds,lookup,'ue_energy_j','a'*64)
    assert r['estimate']==0 and r['confidence_interval']==[0.,0.]
    assert r['fit_diagnostics']['assessment']=='constant_response_observed'


def test_flat_average_slope_is_not_a_constant_response():
    g,seeds,lookup=setup_model(lambda s,x:100*s+100*(x-.5)**2)
    r=fit_grid(g,seeds,lookup,'ue_energy_j','a'*64)
    assert r['estimate']==pytest.approx(0,abs=1e-10)
    assert r['fit_diagnostics']['assessment']=='limited_first_order_approximation'
    assert r['fit_diagnostics']['rmse']>0


def test_adequate_linearity_does_not_resolve_an_uncertain_direction():
    g,seeds,lookup=setup_model(lambda s,x:100*s+(.5*s)*x)
    r=fit_grid(g,seeds,lookup,'ue_energy_j','a'*64)
    d=r['fit_diagnostics']
    assert d['assessment']=='first_order_approximation_supported'
    assert d['average_direction_resolved']is False
    assert d['usable_for_linear_control']is False


def test_line_fit_rejects_unpaired_values():
    with pytest.raises(ValueError):line_fit([0.,.5,1.],[1.,2.])


def test_range_import_requires_domains_diagnostics_and_explicit_timer_component(tmp_path):
    g,seeds,lookup=setup_model(lambda s,x:100*s+2*x)
    row=fit_grid(g,seeds,lookup,'ue_energy_j','a'*64);path=tmp_path/'fit.json'
    def save(r):path.write_text(json.dumps(dict(schema='oran.intervention-edges.v1',estimates=[r])))
    save(row);updated=validate_influences(build_graph(),path)
    assert next(e for e in updated['edges']if e.get('sensitivity'))['causal_status']=='range_linear_approximation'
    broken=deepcopy(row);broken.pop('fit_diagnostics');save(broken)
    with pytest.raises(ValueError):validate_influences(build_graph(),path)
    row['parameter']='ue_drx_configs';save(row)
    with pytest.raises(ValueError):validate_influences(build_graph(),path)
    row['scalar_component']='inactivity_ms';save(row);validate_influences(build_graph(),path)
    row['parameter']='scheduler_mode';save(row)
    with pytest.raises(ValueError):validate_influences(build_graph(),path)
