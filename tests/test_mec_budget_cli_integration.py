"""Mitigation CLI configuration reaches the actual Simulator's manager."""
from experiments.scenario_scan_utils import build_runtime_scenario_config


def test_ric_random_parameters_reach_simulator_manager(tmp_path, monkeypatch):
    import experiments.scenario_scan_utils as scenarios
    from tests.test_paper_interfaces import make_sim
    original = scenarios.build_runtime_scenario_config

    def configured(**kwargs):
        kwargs['ric_config'] = dict(kwargs['ric_config'], direct_conflict_mitigation='uniform_random',
                                    mitigation_seed=23, mitigation_window_ms=100.)
        return original(**kwargs)

    monkeypatch.setattr(scenarios, 'build_runtime_scenario_config', configured)
    sim = make_sim(tmp_path, cells=1)
    assert sim.config.direct_conflict_mitigation == 'uniform_random'
    arbiter = sim.controller_manager.direct_conflict_arbiter
    assert arbiter.seed == 23
    assert arbiter.window_slots == round(100 / sim.config.slot_duration_ms)


def test_existing_runtime_defaults_keep_priority_and_no_random_draws():
    cfg = build_runtime_scenario_config(scenario_name='default', scenario_type='rt_scene',
        sim_config={}, cell_config={}, ric_config={})
    assert cfg.direct_conflict_mitigation == 'priority'
    assert cfg.mitigation_window_ms == 100


def test_live_comparison_forwards_budget_and_mitigation_settings():
    from experiments.live_xapp_test import ARGUMENTS
    assert {'budget_xapp_mode', 'budget_xapp_model', 'budget_xapp_window_ms',
            'budget_xapp_gflops', 'direct_conflict_mitigation',
            'mitigation_seed', 'mitigation_window_ms'} <= set(ARGUMENTS)
