"""Population denominators, missing-member handling and identification scope."""
import json
from copy import deepcopy

import pytest

from smo.runtime import FrameBuilder
from smo.__main__ import GraphService
from smo.population import SCOPE
from smo.identify import fit_contrasts
from smo.contracts import KPIS
from experiments.run_rcp_perturbation import CONTRASTS


def fixture():
    ids=['a','b','c','d']
    config=dict(configs=dict(ue=dict(ues=[dict(ue_id=u,slice_id='urllc' if u in ('b','c') else 'embb') for u in ids]),
        phy=dict(rt_sys=dict(ue_services=dict(mec=dict(tasks_by_ue={
            'a':dict(interval_ms=100,compute_gflop=.2),
            'd':dict(interval_ms=50,compute_gflop=.1)}))))))
    def users(end):
        return [dict(ue_id=u,cell_id='cell',queue_bytes=0,packet=dict(
            delivered_bytes=i*100 if end else 0,completed_packets=[0,1,3,6][i] if end else 0,
            completion_latency_ms_sum=[0,1,9,60][i] if end else 0,
            deadline_failed_packets=[0,1,3,0][i] if end else 0,
            overflow_packets=0,admitted_packets=[10,10,30,20][i] if end else 0,
            overflow_bytes=0,hol_delay_ms=0)) for i,u in enumerate(ids)]
    def services(end):
        rows={u:dict(ue_energy_total_j=(i+1)*10 if end else 0,ue_active_fraction=.5,
            ue_service_policy=dict(offload_fraction=.25 if u=='a' else .75),ue_task_metrics=None) for i,u in enumerate(ids)}
        for u,n,completed,latency,on_time,offloaded in [('a',2,1,10,1,2),('d',6,3,90,1,3)]:
            rows[u]['ue_task_metrics']=dict(generated=n if end else 0,completed=completed if end else 0,
                completion_latency_ms_sum=latency if end else 0,on_time_completed=on_time if end else 0,
                offloaded_generated=offloaded if end else 0,offloaded_completed=completed if end else 0,
                offloaded_completion_latency_ms_sum=latency if end else 0)
        return rows
    return config,users,services


def test_population_sums_all_energy_and_mec_and_weights_events_not_ue_means():
    config,users,services=fixture();b=FrameBuilder(config,population=True)
    b.frame(1,users(False),services(False),{},{});f=b.frame(201,users(True),services(True),{},{})
    value=lambda k:f['states']['kpi:'+k]['value']
    assert value('ue_energy_j')==100
    assert value('ue_power_w')==pytest.approx(1000)  # 100 J / 0.1 s, summed across all UEs
    assert value('mec_expected_gflops')==pytest.approx(2)
    assert value('mec_realized_gflops')==pytest.approx(7)
    assert value('per_ue_goodput_mbps')==pytest.approx(.012)  # includes the zero-rate member
    assert value('task_latency_ms')==25  # (10 + 90) / (1 + 3), not (10 + 30) / 2
    assert value('task_success_ratio')==.25
    assert value('urllc_packet_latency_ms')==2.5
    assert value('urllc_failure_events_per_s')==40
    assert value('urllc_failure_fraction')==.1
    assert set(f['states']['rcp:ue_offload_fraction']['by_entity'])=={'a','d'}


def test_incomplete_population_is_unavailable_instead_of_silently_partial():
    config,users,services=fixture();s=services(True);s['c'].pop('ue_energy_total_j')
    frame=FrameBuilder(config,population=True).frame(1,users(True),s,{},{})
    assert frame['states']['kpi:ue_energy_j']['value']is None
    assert frame['states']['kpi:ue_power_w']['value']is None
    config['configs']['ue']['ues'].append(dict(ue_id='missing',slice_id='urllc'))
    b=FrameBuilder(config,population=True);b.frame(1,users(False),services(False),{},{})
    frame=b.frame(201,users(True),services(True),{},{})
    assert frame['states']['kpi:urllc_failure_events_per_s']['value']is None
    assert frame['states']['kpi:urllc_failure_fraction']['value']is None
    assert frame['states']['kpi:network_goodput_mbps']['value']is None
    assert frame['states']['kpi:ue_energy_j']['value']is None
    assert frame['states']['kpi:ue_power_w']['value']is None


def test_population_graph_rejects_legacy_kpi_scope_and_single_member_claims(tmp_path):
    graph=GraphService().bundle()['graph']
    assert graph['metric_scope']==SCOPE
    assert next(n for n in graph['nodes']if n['id']=='kpi:ue_energy_j')['scope']=='All UEs'
    p=tmp_path/'effects.json'
    p.write_text(json.dumps(dict(schema='oran.intervention-edges.v1',estimates=[])))
    with pytest.raises(ValueError,match='scope differs'):GraphService(influences=p)
    p.write_text(json.dumps(dict(metric_scope=SCOPE,estimates=[dict(population_control_eligible=False)])))
    with pytest.raises(ValueError,match='single-member'):GraphService(influences=p)


def test_population_estimator_retains_local_probes_without_promoting_their_scope():
    contrasts=[deepcopy(c)for c in CONTRASTS if c['id']in ('offload_ue4','scheduler_latency')]
    plan=dict(seeds=[101,202],contrasts=contrasts,scoring_start_ms=200,scoring_end_ms=700,switch_ms=100)
    cases=[]
    arms={c[k]for c in contrasts for k in ('low','high','reference')}
    for seed in plan['seeds']:
        for i,arm in enumerate(sorted(arms)):
            cases.append(dict(seed=seed,arm=arm,metrics=dict.fromkeys(KPIS,float(i+1))))
    rows=fit_contrasts(plan,cases,'a'*64,population=True)
    assert all(r['metric_scope']==SCOPE for r in rows)
    assert all(r['population_control_eligible']==(r['contrast_id']=='scheduler_latency')for r in rows)
    assert all(isinstance(r['estimate'],float)for r in rows)
