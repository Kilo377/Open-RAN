from types import SimpleNamespace as NS
from copy import deepcopy
from models.action import ConfigAction
from ric.xapp_contract import project, contract_for, validate_action, XAppContract, KPI_REGISTRY


def fixture():
    u=NS(ue_id='u',serving_cell_id='c',queue_bytes=0,hol_delay_ms=0,reference_sinr_db=5.,
        rsrp_dbm=-80.,data_sinr_db=None,throughput_mbps=0.,bler=.5,bler_sample_count=0,
        sinr_kind='rt_full_band_reference',measurement_produced_at_slot=3,
        measurement_delivered_at_slot=5,queue_produced_at_slot=3,sinr_window_slots=4,
        sinr_effective_sample_count=3,throughput_window_slots=4,bler_window_tbs=16,
        demand_true_backlog_bytes=8000,cqi=13,post_eq_sinr_db=99.,candidate_cell_observations=[])
    c=NS(cell_id='c',total_queue_bytes=8000,resource_grid_utilization=.5,rf_tx_power_w=.2,rf_power_produced_at_slot=4)
    return NS(slot_id=5,cell_id='c',ue_observations=[u],cell_observations=[c],scheduled_ue_ids=[],
        measurement_queue_delay_slots=2,rt_channel_snapshot=object(),scheduler_true_backlog_bytes_total=8000)


def test_truth_isolation_zero_traffic_vs_no_bler_samples_and_metadata():
    observation=fixture();view,errors=project(observation,contract_for(['power_cap_dbm']))
    assert errors==[]
    ue=view.ue_observations[0]
    assert ue.queue_bytes==0 and ue.bler is None
    assert not hasattr(ue,'demand_true_backlog_bytes') and not hasattr(ue,'cqi')
    assert not hasattr(view,'rt_channel_snapshot') and not hasattr(view,'scheduler_true_backlog_bytes_total')
    assert view.total_queue_bytes==view.cell_observations[0].total_queue_bytes==0
    assert ue.kpi_metadata['bler']['reason']=='no_samples'
    assert ue.kpi_metadata['reference_sinr_db']['source'].startswith('RT_reference')
    ue.queue_bytes=100;assert observation.ue_observations[0].queue_bytes==0


def test_missing_stale_future_and_unknown_required_kpis():
    o=fixture();o.ue_observations[0].reference_sinr_db=None
    _,errors=project(o,contract_for(['power_cap_dbm']));assert any('reference_sinr_db:missing' in e for e in errors)
    o=fixture();o.slot_id=100
    _,errors=project(o,contract_for(['scheduler_mode']));assert any('stale' in e for e in errors)
    o=fixture();o.ue_observations[0].measurement_delivered_at_slot=9
    _,errors=project(o,contract_for(['power_cap_dbm']));assert any('future' in e for e in errors)


def test_actions_enforce_ownership_backend_and_cell_scope():
    o=fixture();c=contract_for(['power_cap_dbm','muting_ratio'])
    assert c.writable_actions=={'power_cap_dbm'}
    assert not validate_action(ConfigAction(power_cap_dbm=25,target_cell_id='c'),c,o)
    assert 'invalid_target_cell' in validate_action(ConfigAction(power_cap_dbm=25,target_cell_id='bad'),c,o)
    assert 'tx_power_dbm:undeclared_action' in validate_action(ConfigAction(tx_power_dbm=10),c,o)
    assert 'muting_ratio:unsupported_rt_action' in validate_action(ConfigAction(muting_ratio=.2),c,o)
    assert not contract_for(['muting_ratio']).available


def test_catalog_reports_missing_independent_csi_and_rt_only_actions():
    from ric.xapp_contract import capability_record
    from ric.non_rt.intent_orchestration_helpers import available_xapp_catalog_for_llm
    m=capability_record('mobility_predictor_xapp',['mobility_bias_db','time_to_trigger_ms'])
    assert not m['available'] and m['missing_kpi_sources']==['independent_reported_cqi']
    entries=available_xapp_catalog_for_llm(['traffic_steering_xapp'],backend='sionna_rt_sys')
    assert entries and 'cio_bias_db' not in entries[0]['tunable_fields']
    assert 'cio_bias_db' in entries[0]['unavailable_actions']


def test_unsupported_builtin_parameters_are_not_requested_on_rt_projection():
    from ric.near_rt.power_saving_xapp import PowerSavingXApp
    from models.policy import PolicySnapshot
    observation=fixture();view,errors=project(observation,contract_for(['power_cap_dbm']))
    assert not errors
    # This xApp reads cell resource counters/observed queues only; no CSI truth.
    action=PowerSavingXApp().run(view,PolicySnapshot()).config_action
    assert action.sleep_threshold is action.wake_up_threshold is action.dwell_time_slots is None
    assert not validate_action(action,contract_for(PowerSavingXApp().controlled_fields),observation)


def test_restricted_contract_rejects_implicit_primary_cell_and_unknown_scope():
    from dataclasses import replace
    o=fixture();o.cell_id='primary'
    c=replace(contract_for(['power_cap_dbm']),cell_ids=('c',))
    assert 'default_primary_cell_exceeds_contract' in validate_action(ConfigAction(power_cap_dbm=20),c,o)
    assert not validate_action(ConfigAction(power_cap_dbm=20,target_cell_id='c'),c,o)
    assert 'unsupported_scope' in validate_action(ConfigAction(power_cap_dbm=20,apply_scope='primary_cell'),c,o)
