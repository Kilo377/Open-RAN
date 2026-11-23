"""Known analytic fixtures for the estimator, not physical-network evidence."""
from copy import deepcopy
import json

import pytest

from smo.contracts import build_graph,validate_influences,KPIS
from smo.identify import fit_contrasts,paired_summary,response_delta
from experiments.run_rcp_perturbation import CONTRASTS


def test_blocked_symmetric_regression_recovers_slope_despite_seed_offsets_and_curvature():
    contrast=deepcopy(next(c for c in CONTRASTS if c['id']=='weight_mec_ue4'))
    plan=dict(seeds=[101,202],contrasts=[contrast],scoring_start_ms=200,scoring_end_ms=700,switch_ms=100)
    cases=[]
    for seed,offset in ((101,10.),(202,500.)):
        for arm,x in ((contrast['low'],-.25),(contrast['reference'],0.),(contrast['high'],.25)):
            metrics={k:offset+3*x+4*x*x for k in KPIS}
            metrics['per_ue_goodput_mbps']={'ue_0':offset-2*x,'ue_4':offset+6*x}
            cases.append(dict(seed=seed,arm=arm,metrics=metrics))
    result=fit_contrasts(plan,cases,'a'*64)
    row=next(r for r in result if r['kpi']=='ue_energy_j')
    assert row['estimate']==pytest.approx(3.)
    assert row['center_linearity_residual']==pytest.approx(.25)
    assert row['sample_count']==2 and row['observed_run_count']==6
    assert row['confidence_interval']==pytest.approx([3.,3.])
    vector=next(r for r in result if r['kpi']=='per_ue_goodput_mbps')
    assert vector['estimate']is None
    assert vector['entity_estimates']==pytest.approx({'ue_0':-2.,'ue_4':6.})


def test_uncertainty_uses_seed_pairs_and_keeps_zero_response_explicit():
    result=paired_summary([1.,3.])
    assert result['estimate']==2. and result['standard_error']==pytest.approx(1.)
    assert result['confidence_interval']==pytest.approx([-10.7062047364,14.7062047364])
    assert result['sign_agreement']==2
    assert paired_summary([0.,0.])['zero_in_both']is True
    assert response_delta(10.,10.+1e-13)==0.
    assert response_delta(10.,10.+1e-5)==pytest.approx(1e-5)


def estimate(**changes):
    base=dict(parameter='ue_scheduler_weights',kpi='ue_energy_j',contrast_id='mec',estimate=-.2,
        unit='J / weight',kind='local_derivative',sample_count=2,horizon_ms=600,scope='UE4',
        context='latency_aware',method='paired central difference',data_sha256='a'*64,
        design='controlled_perturbation',confidence_interval=[-.3,-.1])
    return{**base,**changes}


def test_multiple_scopes_remain_selectable_instead_of_being_averaged(tmp_path):
    a=estimate();b=estimate(contrast_id='urllc',estimate=.5,scope='UE6',context='qos_deadline',confidence_interval=[.2,.8])
    p=tmp_path/'effects.json';p.write_text(json.dumps(dict(schema='oran.intervention-edges.v1',estimates=[a,b])))
    g=validate_influences(build_graph(),p)
    edge=next(e for e in g['edges']if e.get('sensitivity'))
    assert edge['sensitivity']['estimate']==-.2
    assert [s['estimate']for s in edge['sensitivities']]==[-.2,.5]
    p.write_text(json.dumps(dict(schema='oran.intervention-edges.v1',estimates=[a,a])))
    with pytest.raises(ValueError,match='Duplicate'):validate_influences(build_graph(),p)


def test_entity_effects_do_not_collapse_opposite_ue_responses(tmp_path):
    row=estimate(kpi='per_ue_goodput_mbps',estimate=None,confidence_interval=None,
        entity_estimates={'ue_0':-1.,'ue_4':1.})
    p=tmp_path/'effects.json';p.write_text(json.dumps(dict(schema='oran.intervention-edges.v1',estimates=[row])))
    g=validate_influences(build_graph(),p)
    edge=next(e for e in g['edges']if e.get('sensitivity'))
    assert edge['sign']=='mixed'and edge['sensitivity']['entity_estimates']=={'ue_0':-1.,'ue_4':1.}
