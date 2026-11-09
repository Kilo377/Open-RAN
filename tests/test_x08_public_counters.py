"""Real packet events become delayed cumulative public counters, not QoS truth."""
from copy import deepcopy

from ric.near_rt.rule_based_v1.telemetry import packet_cumulative_counters, PACKET_COUNTER_FIELDS
from ric.xapp_contract import project, XAppContract
from tests.test_ue_services import fixture as services_fixture
from tests.test_xapp_contract import fixture as observation_fixture


def packet_events():
    state, _ = services_fixture()
    ue=state.ues[0]
    channel=ue.logical_channels[0]
    channel.max_queue_bytes=400
    channel.packet_delay_budget_ms=10
    ledger=ue.packet_ledger
    assert ledger.add(channel,600,1,packet_size=200)==400
    ledger.refresh(12)
    ledger.deliver(channel,200,13)
    return ledger,channel


def public_view(counters,now=5):
    observation=observation_fixture()
    observation.slot_id=now
    observation.delivered_at_slot=5
    observation.paper_report=dict(produced_at_slot=3,window_slots=4,reference_window_slots=4,connection_model_enabled=False,
        values={'ue':{'u':deepcopy(counters)},'cell':{}})
    contract=XAppContract(tuple(PACKET_COUNTER_FIELDS),frozenset({'scheduler_mode'}),max_age_slots=10)
    return project(observation,contract,now_slot=now)


def test_real_overflow_overdue_and_late_completion_are_not_a_unique_failure_probability():
    ledger,channel=packet_events()
    values=packet_cumulative_counters(ledger)
    assert values==dict(admitted_packets_total=2,deadline_missed_packets_total=2,
        overflow_packets_total=1,completed_packets_total=1,late_completed_packets_total=1,
        on_time_completed_packets_total=0,deadline_failed_packets_total=2,
        completion_latency_ms_total=12.)
    # The same overdue packet can subsequently complete late.
    assert values['deadline_missed_packets_total']+values['late_completed_packets_total']+values['overflow_packets_total']>3
    ledger.deliver(channel,200,14)
    later=packet_cumulative_counters(ledger)
    assert later['deadline_missed_packets_total']==2
    assert later['late_completed_packets_total']==later['completed_packets_total']==2
    assert later['deadline_failed_packets_total']==2
    assert later['completion_latency_ms_total']==25.


def test_real_counters_traverse_delayed_projection_with_cumulative_metadata():
    ledger,_=packet_events()
    values=packet_cumulative_counters(ledger)
    view,errors=public_view(values)
    assert not errors
    ue=view.ue_observations[0]
    for name,value in values.items():
        assert getattr(ue,name)==value
        metadata=ue.kpi_metadata[name]
        assert metadata['valid'] and metadata['unit']==('ms' if name=='completion_latency_ms_total' else 'packet')
        assert metadata['window']['kind']=='cumulative'
        assert metadata['produced_at_slot']==3 and metadata['delivered_at_slot']==5
    assert not hasattr(ue,'packet_ledger')


def test_zero_missing_and_stale_public_counters_remain_distinct():
    state,_=services_fixture()
    values=packet_cumulative_counters(state.ues[0].packet_ledger)
    view,errors=public_view(values)
    assert not errors and view.ue_observations[0].admitted_packets_total==0
    del values['overflow_packets_total']
    _,errors=public_view(values)
    assert any('overflow_packets_total' in e and ('missing' in e or 'no_samples' in e) for e in errors)
    values=packet_cumulative_counters(state.ues[0].packet_ledger)
    _,errors=public_view(values,now=30)
    assert any('stale' in e for e in errors)


def test_late_completion_before_refresh_is_one_first_failure_for_delivery_and_ack():
    for via_ack in (False, True):
        state,_=services_fixture()
        ue=state.ues[0]
        channel=ue.logical_channels[0]
        channel.packet_delay_budget_ms=10
        ledger=ue.packet_ledger
        ledger.add(channel,200,1,packet_size=200)
        if via_ack:
            ledger.reserve('late-tb',[(channel.logical_channel_id,200)])
            ledger.acknowledge('late-tb',12)
            ledger.acknowledge('late-tb',13)  # Duplicate ACK cannot add an outcome.
        else:
            ledger.deliver(channel,200,12)
        ledger.refresh(13)
        assert ledger.stats['deadline_failed_packets']==1
        assert ledger.stats['late_completed_packets']==1
        assert ledger.stats['deadline_missed_packets']==0
        assert ledger.stats['completion_latency_ms_sum']==11
