"""MEC stage timing follows actual slot clocks without changing service work."""
from copy import deepcopy
from types import SimpleNamespace as NS
import random

import pytest

from phy.rt_sys.ue_services import TASK_STAGES, TASK_SERVICE_STAGES, TASK_STAGE_SCHEMA
from phy.rt_sys.ue_services import UeServices
from tests.test_ue_services import fixture, step


def config(*, offload=1., count=1, **overrides):
    mec=dict(enabled=True, local_gflops=2., server_gflops=2.,
             uplink_bps_per_cell=1000000, backhaul_one_way_ms=2.5,
             tasks_by_ue={f'ue_{i}':dict(interval_ms=1000,input_bytes=125,
                 result_bytes=200,compute_gflop=.001,deadline_ms=100,
                 offload_fraction=offload) for i in range(count)})
    mec.update(overrides)
    return dict(task_cohort_ms=100,stage_metrics=True,mec=mec)


def timing(event):
    t=event['task_stage_timing']
    assert t['schema']==TASK_STAGE_SCHEMA
    assert set(t['stage_ms'])==set(TASK_STAGES)
    assert sum(t['stage_ms'].values())==pytest.approx(t['elapsed_ms'],abs=1e-7)
    assert set(t['service_ms'])==set(TASK_SERVICE_STAGES)
    assert sum(t['upload_blocked_by_reason_ms'].values())==pytest.approx(t['upload_blocked_ms'])
    for stage in TASK_SERVICE_STAGES:
        pieces=t['service_ms'][stage]+t['wait_or_quantization_ms'][stage]
        if stage=='upload':pieces+=t['upload_blocked_ms']
        assert pieces==pytest.approx(t['stage_ms'][stage],abs=1e-7)
    return t


def test_slot_boundary_residency_and_fractional_work_have_different_truthful_clocks():
    _,m=fixture(config())
    events=[]
    for slot in range(1,10):events+=step(m,slot,deliver=True).get('ue_service_events',[])
    terminal=[e for e in events if e['kind']=='task_completed']
    assert len(terminal)==1
    e=terminal[0];t=timing(e)
    # 1 ms UL, round 2.5 ms backhaul up to each next slot boundary,
    # 0.5 ms of CPU work held for the physical 1 ms slot, then 1 DL slot.
    assert e['latency_ms']==pytest.approx(9.)
    assert t['elapsed_ms']==pytest.approx(e['latency_ms'])
    assert t['stage_ms']==pytest.approx(dict(admission=0,upload=1,
        backhaul_to_edge=3,edge_compute=1,backhaul_to_result=3,downlink=1,local_compute=0))
    assert t['service_ms']==pytest.approx(dict(upload=1,edge_compute=.5,local_compute=0))
    assert t['wait_or_quantization_ms']['edge_compute']==pytest.approx(.5)
    assert t['terminal_stage']=='downlink'
    summary=m.summary()['task_stage_metrics']
    assert summary['completed']['count']==1 and summary['failed']['count']==0
    assert summary['completed']['stage_ms_sum']==pytest.approx(t['stage_ms'])
    assert summary['pending']['count']==0 and summary['pending']['tasks']==[]
    assert m.reports['ue_0']['ue_task_stage_metrics']['completed']['count']==1


def test_shared_fifo_upload_part_slots_count_actual_work_once():
    cfg=config(count=2,uplink_bps_per_cell=5000000,backhaul_one_way_ms=0)
    for p in cfg['mec']['tasks_by_ue'].values():p['input_bytes']=1000
    _,m=fixture(cfg,count=2)
    for slot in range(1,8):step(m,slot)
    pending=m.summary()['task_stage_metrics']['pending']['tasks']
    assert len(pending)==2
    times={row['ue_id']:row['task_stage_timing'] for row in pending}
    assert times['ue_0']['stage_ms']['upload']==pytest.approx(2.)
    assert times['ue_1']['stage_ms']['upload']==pytest.approx(4.)
    assert [times[u]['service_ms']['upload'] for u in ['ue_0','ue_1']]==pytest.approx([1.6,1.6])
    assert times['ue_1']['wait_or_quantization_ms']['upload']==pytest.approx(2.4)
    assert m.server_stats['cell_0']['uploaded_bits']==16000
    for row in pending:
        assert sum(row['task_stage_timing']['stage_ms'].values())==pytest.approx(7.)


@pytest.mark.parametrize('reason',['no_path','not_connected','radio_unavailable'])
def test_blocked_uplink_ages_and_preserves_real_reason_without_treating_it_as_service(reason):
    state,m=fixture(config())
    if reason=='no_path':state.paper=NS(raw={'ue_0':{'cell_0':{'no_path':True}}})
    if reason=='not_connected':state.connections['ue_0'].connection_state='handover_interruption'
    for slot in range(1,11):
        m.begin_slot(slot);m.before_scheduling()
        if reason=='radio_unavailable':state.ues[0].ue_radio_available=False
        m.after_phy(dict(ues=[]))
    report=m.summary()['task_stage_metrics']
    assert report['completed']['count']==report['failed']['count']==0
    assert report['pending']['by_stage']['upload']==1
    t=report['pending']['tasks'][0]['task_stage_timing']
    assert t['elapsed_ms']==pytest.approx(10.)
    assert t['stage_ms']['upload']==pytest.approx(10.)
    assert t['upload_blocked_ms']==pytest.approx(10.)
    assert t['upload_blocked_by_reason_ms'][reason]==pytest.approx(10.)
    assert t['service_ms']['upload']==0
    assert m.server_stats['cell_0']['uploaded_bits']==0
    assert m.reports['ue_0']['ue_task_stage_metrics']['pending']['elapsed_ms_sum']==pytest.approx(10.)


def test_handover_moves_upload_but_does_not_restart_its_stage_clock_or_double_serve():
    state,m=fixture(config(uplink_bps_per_cell=8000))
    state.managed_cells['cell_1']=NS()
    m.upload['cell_1']=__import__('collections').deque()
    m.edge['cell_1']=__import__('collections').deque()
    m.server_stats['cell_1']=__import__('collections').Counter()
    step(m,1)
    state.connections['ue_0'].serving_cell_id='cell_1'
    step(m,2)
    t=m.summary()['task_stage_metrics']['pending']['tasks'][0]['task_stage_timing']
    assert t['stage_ms']['upload']==pytest.approx(2.)
    assert t['service_ms']['upload']==pytest.approx(2.)
    assert m.server_stats['cell_0']['uploaded_bits']==m.server_stats['cell_1']['uploaded_bits']==8


def test_downlink_partial_ack_release_and_final_ack_keep_one_continuous_result_residency():
    state,m=fixture(config())
    for slot in range(1,11):step(m,slot)
    ledger=state.ues[0].packet_ledger
    ledger.reserve('first',[('lc_0',100)])
    ledger.acknowledge('first',10)
    assert m.stats['ue_0']['completed']==0
    ledger.reserve('retried',[('lc_0',100)])
    ledger.release('retried')
    assert m.summary()['task_stage_metrics']['pending']['by_stage']['downlink']==1
    for slot in range(11,21):step(m,slot)
    ledger.reserve('last',[('lc_0',100)])
    m.events=[]
    ledger.acknowledge('last',20)
    e=m.events[0];t=timing(e)
    assert e['latency_ms']==pytest.approx(20.)
    assert t['stage_ms']['downlink']==pytest.approx(12.)
    assert 'downlink' not in t['service_ms']  # No invented PHY duration.
    ledger.acknowledge('last',21)
    assert m.summary()['task_stage_metrics']['completed']['count']==1
    assert m.summary()['task_stage_metrics']['pending']['count']==0


def test_local_computation_is_separate_from_edge_and_soft_deadline_does_not_cancel_work():
    cfg=config(offload=0.,local_gflops=1.)
    cfg['mec']['tasks_by_ue']['ue_0'].update(compute_gflop=.0055,deadline_ms=2)
    _,m=fixture(cfg)
    events=[]
    for slot in range(1,7):events+=step(m,slot).get('ue_service_events',[])
    assert m.stats['ue_0']['deadline_misses']==1
    e=next(e for e in events if e['kind']=='task_completed');t=timing(e)
    assert e['latency_ms']==pytest.approx(6.) and e['offloaded'] is False
    assert t['stage_ms']['local_compute']==pytest.approx(6.)
    assert t['service_ms']['local_compute']==pytest.approx(5.5)
    assert t['wait_or_quantization_ms']['local_compute']==pytest.approx(.5)
    assert sum(t['stage_ms'][s] for s in TASK_STAGES if s!='local_compute')==0
    assert m.summary()['task_stage_metrics']['completed']['by_route']['local']['count']==1
    assert 'do not cancel' in m.summary()['task_stage_metrics']['deadline_semantics']


def test_preroute_and_upload_admission_rejects_keep_zero_age_and_distinct_stage():
    cfg=config(count=2,max_queued_tasks=1,uplink_bps_per_cell=8000)
    for p in cfg['mec']['tasks_by_ue'].values():p['interval_ms']=1
    _,m=fixture(cfg,count=2)
    reference=random.Random();reference.setstate(m.rng['ue_0'].getstate());reference.random()
    events=[]
    for slot in range(1,3):events+=step(m,slot).get('ue_service_events',[])
    failed=[e for e in events if e['kind']=='task_queue_overflow']
    assert failed
    assert {timing(e)['terminal_stage'] for e in failed}=={'admission','upload'}
    assert all(timing(e)['elapsed_ms']==0 for e in failed)
    assert m.rng['ue_0'].getstate()==reference.getstate()
    global_report=m.summary()['task_stage_metrics']
    assert global_report['failed']['count']==len(failed)
    assert global_report['failed']['by_route']['unassigned']['count']>0
    assert global_report['failed']['by_route']['offloaded']['count']>0
    births=[e for e in events if e['kind']=='task_generated']
    assert all('task_stage_timing' not in e for e in births)


def test_result_admission_overflow_records_time_to_failure_not_a_success_latency():
    state,m=fixture(config())
    state.ues[0].logical_channels[0].max_queue_bytes=0
    events=[]
    for slot in range(1,11):events+=step(m,slot).get('ue_service_events',[])
    failed=next(e for e in events if e['kind']=='task_result_overflow')
    t=timing(failed)
    assert failed['latency_ms'] is None  # Legacy failure semantics unchanged.
    assert t['elapsed_ms']==pytest.approx(8.)
    assert t['terminal_stage']=='downlink' and t['stage_ms']['downlink']==0
    report=m.summary()['task_stage_metrics']
    assert report['failed']['count']==1 and report['completed']['count']==0
    assert report['failed']['elapsed_mean_ms']==pytest.approx(8.)
    assert report['completed']['elapsed_mean_ms'] is None


def test_pending_snapshots_are_pure_and_terminal_storage_does_not_retain_task_ids():
    _,m=fixture(config())
    for slot in range(1,7):step(m,slot)
    before=deepcopy((m.tasks,m.task_timings,m.future,m.upload,m.edge,m.stats,m.energy))
    a=m.summary();b=m.summary()
    assert a==b and (m.tasks,m.task_timings,m.future,m.upload,m.edge,m.stats,m.energy)==before
    for slot in range(7,15):step(m,slot,deliver=True)
    assert not m.tasks and not m.task_timings
    assert m.summary()['task_stage_metrics']['pending']['tasks']==[]
    assert m.summary()['task_stage_metrics']['completed']['count']==1


@pytest.mark.parametrize('explicit',[False,True])
def test_default_disabled_omits_diagnostics_and_never_enters_stage_accounting_loops(explicit):
    cfg=config()
    if explicit:cfg['stage_metrics']=False
    else:cfg.pop('stage_metrics')
    _,m=fixture(cfg)
    def unexpected(*args,**kwargs):
        raise AssertionError('Disabled stage accounting must not execute')
    for name in ('_stage_reports','_global_stage_report','_stage_transition',
                 '_stage_snapshot','_finish_stage_timing','_upload_block_reason'):
        setattr(m,name,unexpected)
    events=[]
    for slot in range(1,10):events+=step(m,slot,deliver=True).get('ue_service_events',[])
    assert any(e['kind']=='task_completed' for e in events)
    assert all('task_stage_timing' not in e for e in events)
    assert all('ue_task_stage_metrics' not in row for row in m.reports.values())
    assert 'task_stage_metrics' not in m.summary()
    assert m.task_timings==m.terminal_stage_metrics=={}


@pytest.mark.parametrize('value',[1,0,None,'true',[],{}])
def test_stage_metrics_requires_a_real_bool_before_mutating_existing_state(value):
    state,m=fixture(config())
    old_listener=state.ues[0].packet_ledger.service_delivery_listener
    old_available=state.ues[0].ue_radio_available
    with pytest.raises(ValueError,match='invalid_stage_metrics_enabled'):
        UeServices(state,dict(stage_metrics=value),1.)
    assert state.ues[0].packet_ledger.service_delivery_listener is old_listener
    assert state.ues[0].ue_radio_available==old_available
