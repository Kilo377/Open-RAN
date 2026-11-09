from copy import deepcopy
from models.action import ConfigAction
from phy.rt_sys.ue_services import UeServices
from test_paper_interfaces import make_sim


def enable(sim, config):
    sim.state.ue_services = UeServices(sim.state, config, sim.config.slot_duration_ms, 1)
    return sim.state.ue_services


def test_sleep_is_consumed_by_real_scheduler(tmp_path):
    sim = make_sim(tmp_path, cells=1)
    services = enable(sim, {'drx': dict(enabled=True, cycle_ms=10, on_duration_ms=1, inactivity_ms=0, wakeup_ms=0)})
    sleeping = 0
    for _ in range(40):
        sim.step()
        for row in sim.phy_adapter.trace['ues']:
            if not services.users[row['ue_id']].ue_radio_available:
                sleeping += 1
                assert not row['scheduled']
    assert sleeping > 0
    assert 'ue_services' in sim.phy_adapter.trace
    assert all(v['ue_energy_total_j'] > 0 for v in services.reports.values())


def test_service_action_applies_and_invalid_batch_changes_nothing(tmp_path):
    sim = make_sim(tmp_path, cells=1)
    services = enable(sim, {})
    original = deepcopy(services.drx)
    power = sim.state.cell.tx_power_dbm
    sim._apply_rt_config_action(ConfigAction(tx_power_dbm=20, ue_drx_configs={'ue_0': {'cycle_ms': -1}}), 0)
    assert sim.state.control.actuation_outcomes[-1]['status'] == 'rejected'
    assert services.drx == original and sim.state.cell.tx_power_dbm == power
    sim._apply_rt_config_action(ConfigAction(ue_drx_configs={'ue_0': {'enabled': True}}), 0)
    assert sim.state.control.actuation_outcomes[-1]['status'] == 'applied'
    assert services.drx['ue_0']['enabled']
    for _ in range(5):
        sim.step()
    from ric.xapp_contract import XAppContract, project
    observation = sim.step()
    observation.delivered_at_slot = observation.slot_id
    projected, errors = project(observation, XAppContract(('ue_energy_total_j',), frozenset({'ue_drx_configs'})), now_slot=observation.slot_id)
    assert not errors
    assert projected.ue_observations[0].ue_energy_total_j >= 0


def test_mec_result_uses_existing_nr_payload_and_task_counters(tmp_path):
    sim = make_sim(tmp_path, cells=1)
    services = enable(sim, {'mec': dict(enabled=True, tasks_by_ue={'ue_0': dict(interval_ms=500, offload_fraction=1.)})})
    for _ in range(100):
        sim.step()
    metrics = services.reports['ue_0']['ue_task_metrics']
    assert metrics['generated'] == 1
    assert metrics['completed'] == 1
    assert metrics['failed'] == metrics['pending'] == 0
    assert metrics['mean_completion_latency_ms'] > 4
    assert sim.state.ues[0].packet_ledger.stats['offered_bytes'] >= 200
    assert services.server_stats['cell_0']['uploaded_bits'] == 8000
