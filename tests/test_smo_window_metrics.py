"""Exact-window audits against analytic counters/events, not network results."""
import gzip
import json
import zlib
from copy import deepcopy
import pytest
from smo.window_metrics import WindowIndex,JsonTail
from smo.intent_quality import assess,normalization,score,quality


def config():
    return dict(manifest=dict(slot_duration_ms=1,ues=[dict(ue_id='a'),dict(ue_id='b')],cells=[dict(cell_id='c')]),
        configs=dict(phy=dict(rt_sys=dict(ue_services=dict(mec=dict(tasks_by_ue={
            'a':dict(interval_ms=10,compute_gflop=.01,deadline_ms=2),
            'b':dict(interval_ms=10,compute_gflop=.01,deadline_ms=5)}))))))


def test_per_entity_counter_reset_cannot_be_masked_by_other_ue_growth():
    w=WindowIndex(config())
    for s in range(1,14):
        w.observe(s,[dict(ue_id='a'),dict(ue_id='b')],{
            'a':dict(ue_energy_total_j=s if s<8 else s-7),
            'b':dict(ue_energy_total_j=10*s)}, {})
    r=w.window(['ue_power_w'],2,12,through=12)['states']['kpi:ue_power_w']
    assert r['value']is None and r['reason']=='missing_or_reset_counter'
    assert w.window(['ue_power_w'],8,12,through=12)['states']['kpi:ue_power_w']['value']==pytest.approx(11000)


def test_interior_missing_slot_counter_and_configuration_are_not_hidden():
    w=WindowIndex(config())
    for s in [1,2,3,5,6]:
        w.observe(s,[],{}, {})
    assert w.window(['ue_power_w'],2,6,through=6)['states']['kpi:ue_power_w']['reason']=='exact_window_data_unavailable'
    assert w.window(['ue_power_w'],2,6,through=5)['states']['kpi:ue_power_w']['reason']=='invalid_or_future_window'


def task_events(pending=False):
    def birth(uid,ident,t,d):return dict(kind='task_generated',ue_id=uid,task_id=ident,created_s=t,time_s=t,deadline_s=d,offloaded=True)
    a=birth('a',1,.002,.002);b=birth('b',2,.003,.005)
    old=birth('a',0,0,.002);edge=birth('a',3,.007,.002)
    events={1:[old],3:[a],4:[b],8:[edge]}
    def finish(birth,when):return dict(birth,kind='task_completed',time_s=when,latency_ms=(when-birth['created_s'])*1000)
    events.setdefault(5,[]).append(finish(old,.005)) # pre-window birth excluded even if completed inside
    events.setdefault(6,[]).append(finish(a,.005)) # 3ms > its own 2ms deadline
    events.setdefault(10,[]).append(finish(edge,.009)) # birth exactly at guard cutoff excluded
    events.setdefault(14 if pending else 7,[]).append(finish(b,.014 if pending else .007))
    return events


def task_index(*,pending=False,drop=False):
    w=WindowIndex(config());events=task_events(pending)
    counts={u:dict(generated=0,completed=0,failed=0,offloaded_generated=0,offloaded_completed=0,offloaded_failed=0)for u in ('a','b')}
    for slot in range(1,16):
        rows=events.get(slot,[])
        for e in rows:
            c=counts[e['ue_id']]
            for field in ('generated','offloaded_generated')if e['kind']=='task_generated'else('completed','offloaded_completed'):c[field]+=1
        services={u:dict(ue_energy_total_j=slot*.001,ue_task_metrics=deepcopy(counts[u]),ue_service_policy=dict(offload_fraction=1.))for u in counts}
        w.observe(slot,[dict(ue_id=u)for u in counts],services,{},[]if drop and slot==3 else rows)
    return w


def test_mec_equal_birth_cohort_no_pre_window_or_guard_edge_samples():
    w=task_index()
    r=w.window(['mec_offloaded_latency_ms','task_success_ratio'],2,12,through=12)['states']
    m=r['kpi:mec_offloaded_latency_ms']
    assert m['value']==pytest.approx(3.5) and m['generated']==2 and m['sample_count']==2
    assert m['all_generated']==3 and m['cohort_exposure_ms']==pytest.approx(5.)
    assert m['pending']==0 and m['failed']==0
    assert r['kpi:task_success_ratio']['value']==.5 # each task's own deadline, not max=5ms


def test_future_terminal_cannot_leak_back_to_earlier_window():
    w=task_index(pending=True)
    r=w.window(['mec_offloaded_latency_ms'],2,12,through=15)['states']['kpi:mec_offloaded_latency_ms']
    assert r['value']is None and r['pending']==1 and r['reason']=='unresolved_mature_tasks'


def test_event_missing_route_identity_change_and_empty_samples_are_unavailable():
    w=task_index(drop=True)
    assert w.window(['mec_offloaded_latency_ms'],2,12,through=12)['states']['kpi:mec_offloaded_latency_ms']['reason']=='incomplete_task_event_log'
    w=task_index()
    # Window shorter than max deadline => zero mature birth exposure, never zero latency.
    m=w.window(['mec_offloaded_latency_ms'],8,12,through=12)['states']['kpi:mec_offloaded_latency_ms']
    assert m['value']is None and m['reason']=='no_mature_task_samples' and m['cohort_exposure_ms']==0
    e=dict(w.tasks[('a',1)]['terminal']);e['offloaded']=False
    w.observe_task(e,15)
    assert w.task_errors


def test_configuration_time_mean_does_not_hide_transient_limit_violation():
    w=WindowIndex(config())
    for s in range(1,14):
        fraction=1. if s==6 else .5
        w.observe(s,[],{u:dict(ue_service_policy=dict(offload_fraction=fraction))for u in ('a','b')},{})
    metric=w.window(['mec_expected_gflops'],2,12,through=12)['states']['kpi:mec_expected_gflops']
    assert metric['value']==pytest.approx(1.1) and metric['maximum']==2.
    row=dict(kpi='mec_expected_gflops',lower=0,upper=1.8,lower_inclusive=True,upper_inclusive=True)
    assert assess(row,dict(states={'kpi:mec_expected_gflops':metric}),dict(scale=1))['status']=='above_range'


def test_gzip_tail_partial_flush_and_concatenated_members_without_duplicates(tmp_path):
    p=tmp_path/'trace.jsonl.gz';seen=[];tail=JsonTail(tmp_path,'trace',seen.append)
    encoder=zlib.compressobj(wbits=31)
    a=encoder.compress(b'{"slot":1}\n{"slot":')+encoder.flush(zlib.Z_SYNC_FLUSH)
    p.write_bytes(a[:10]);tail.sync();assert not seen
    with p.open('ab')as f:f.write(a[10:])
    tail.sync();assert seen==[{'slot':1}]
    with p.open('ab')as f:f.write(encoder.compress(b'2}\n')+encoder.flush(zlib.Z_FINISH)+gzip.compress(b'{"slot":3}\n'))
    tail.sync();tail.sync();assert seen==[{'slot':1},{'slot':2},{'slot':3}]


def constraint(**overrides):
    return dict(dict(kpi='ue_power_w',lower=0.,upper=5.8,lower_inclusive=True,upper_inclusive=True),**overrides)


def test_zero_scale_has_explicit_bound_resolution_or_unavailable_fallback():
    r=constraint(upper=0)
    assert normalization(r,dict(value=0,resolution=4))==dict(scale=4,basis='measurement_resolution')
    assert normalization(r,dict(value=0))['scale']is None
    assert normalization(constraint(),dict(value=0))['scale']==5.8
    assert normalization(constraint(),dict(value=6))['scale']==6


def test_worst_gap_is_dimensionless_no_compensation_no_oversatisfaction_reward():
    rows=[constraint(),constraint(kpi='mec_expected_gflops',upper=1.8)]
    scales={'ue_power_w':dict(scale=6),'mec_expected_gflops':dict(scale=2)}
    def state(a,b):return dict(states={'kpi:ue_power_w':dict(value=a),'kpi:mec_expected_gflops':dict(value=b)})
    before=score(rows,state(6,2),scales);after=score(rows,state(1,2.2),scales)
    assert after['satisfied']==1 and not after['all_satisfied']
    q=quality(before,after);assert q['state']=='further' and q['improvement_percentage_points']==pytest.approx(-10.)
    a=score(rows,state(5,1.7),scales);b=score(rows,state(1,0),scales)
    assert quality(a,b)['state']=='maintained_feasibility' and quality(a,b)['improvement_percentage_points']==0
    assert not quality(a,b)['ranking_performed'] and not quality(a,b)['oversatisfaction_reward']
