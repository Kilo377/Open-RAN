"""Actual AP reports keep their delayed timestamps, counters and nullable ACK."""
from copy import deepcopy
from types import SimpleNamespace

from ric.xapp_contract import project, XAppContract
from ric.near_rt.pools.chongyu.ap_energy_rf import AP_COUNTERS, AP_PROOF, AP_NULLABLE_PROOF
from tests.test_ap_services import model, step, policy
from tests.test_xapp_contract import fixture


def view(report, produced, now):
    observation=fixture();observation.slot_id=now
    observation.paper_report=dict(produced_at_slot=produced,window_slots=4,reference_window_slots=4,
        connection_model_enabled=False,values={'ue':{},'cell':{'c':deepcopy(report)}})
    required=AP_COUNTERS+tuple(k for k in AP_PROOF if k not in AP_NULLABLE_PROOF)
    return project(observation,XAppContract(required,frozenset({'cell_sleep_mode'}),max_age_slots=10),now_slot=now)


def test_actual_counters_keep_units_and_delayed_cumulative_metadata_without_initial_owner():
    state,ap=model()
    for slot in range(1,4):step(state,ap,slot)
    observation,errors=view(ap.reports['cell_0'],3,5)
    assert not errors
    cell=observation.cell_observations[0]
    assert cell.ap_policy_owner is None and cell.ap_policy_applied_slot is None
    assert cell.ap_energy_total_j==ap.reports['cell_0']['ap_energy_total_j']
    for name in AP_COUNTERS:
        meta=cell.kpi_metadata[name]
        assert meta['valid'] and meta['produced_at_slot']==3 and meta['delivered_at_slot']==5
        assert meta['window']=={'kind':'cumulative','size':None}
    assert cell.kpi_metadata['ap_energy_total_j']['unit']=='J'
    assert cell.kpi_metadata['ap_policy']['window']=={'kind':'slots','size':1}
    assert not hasattr(observation,'ap_services')


def test_consumer_ack_is_not_inferred_from_initial_active_config_and_stale_data_rejects():
    state,ap=model()
    for slot in range(1,4):step(state,ap,slot)
    before=deepcopy(ap.reports['cell_0'])
    policy(state,'active',4,owner='ap_energy_rf');step(state,ap,4)
    observation,errors=view(ap.reports['cell_0'],4,6)
    assert not errors
    assert before['ap_policy_owner'] is None
    cell=observation.cell_observations[0]
    assert cell.ap_policy_owner=='ap_energy_rf' and cell.ap_policy_applied_slot==4
    assert cell.kpi_metadata['ap_policy_applied_slot']['produced_at_slot']==4
    _,errors=view(ap.reports['cell_0'],4,30)
    assert any('ap_energy_total_j:stale' in e for e in errors)


def test_missing_ap_reference_masks_held_numeric_values_and_candidates_until_wake():
    observation=fixture();observation.delivered_at_slot=5
    ue=observation.ue_observations[0]
    ue.candidate_cell_observations=[SimpleNamespace(cell_id='c',rsrp_dbm=-80.,
        candidate_beam_id=0,is_serving=True,is_target=False,
        measurement_produced_at_slot=3,measurement_delivered_at_slot=5,measurement_window_slots=4)]
    report=dict(reference_sinr_db=5.,rsrp_dbm=-80.,wideband_cqi=8,
        reference_measurement_available=False,candidate_reference_available={'c':False})
    observation.paper_report=dict(produced_at_slot=3,window_slots=4,reference_window_slots=4,
        connection_model_enabled=False,values={'ue':{'u':report},'cell':{}})
    contract=XAppContract(('reference_sinr_db',),frozenset({'scheduler_mode'}),max_age_slots=10)
    projected,errors=project(observation,contract)
    public=projected.ue_observations[0]
    assert errors==['u:reference_sinr_db:ap_reference_unavailable']
    assert public.reference_measurement_available is False
    assert public.reference_sinr_db is public.rsrp_dbm is public.wideband_cqi is None
    assert public.candidate_rsrp_dbm is None
    assert public.kpi_metadata['reference_sinr_db']['sample_count']==0
    candidate=public.candidate_cell_observations[0]
    assert candidate.rsrp_dbm is None and not candidate.kpi_metadata['valid']
    assert candidate.kpi_metadata['reason']=='ap_reference_unavailable'
    report.update(reference_measurement_available=True,candidate_reference_available={'c':True})
    projected,errors=project(observation,contract)
    assert not errors
    assert projected.ue_observations[0].reference_sinr_db==5.
    assert projected.ue_observations[0].candidate_rsrp_dbm==( ('c',-80.), )
