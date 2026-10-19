import json
from types import SimpleNamespace as NS

import pytest

from experiments.live_xapp_test import LiveXAppSession, configure, submit, atomic_json
from ric.near_rt.live_scheduler_test_xapp import LiveSchedulerTestXApp


def fixture_sim(tmp_path, enabled=False):
    from test_rt_sys import FakeTracer, FixtureAbstraction
    from phy.rt_sys.scenario import build_case
    from phy.rt_sys.channel import RtChannelProvider
    from phy.rt_sys.adapter import RtSysPhyAdapter
    from core.factory import build_controller_registry, build_scheduler
    from core.simulator import Simulator
    from experiments.scenario_scan_utils import build_runtime_scenario_config
    manifest, configs, _ = build_case(num_cells=2, num_ues=4, total_slots=100, initial_queue_bytes=20000)
    args = NS(live_xapp_test=True, phase='integration', control_mode='baseline', slots=100,
        xapp_test_start='on' if enabled else 'off', xapp_test_cell='*', _live_slot_ms=.5)
    configure(configs, manifest, args)
    configs['ric']['xapp_interval_slots'] = 2  # Short unit-test clock, not the live 20 ms period.
    configs['sim']['config_actuation_delay_slots'] = 5
    configs['phy']['rt_sys']['feedback_mode'] = 'sys_abstraction'

    class LowCapacity(FixtureAbstraction):
        def evaluate(self, *a, **kw):
            return {**super().evaluate(*a, **kw), 'capacity_bytes': 32}

    provider = RtChannelProvider(manifest, tmp_path / 'cache', backend=FakeTracer(manifest))
    adapter = RtSysPhyAdapter({**configs['phy']['rt_sys'], 'device': 'numpy'},
        abstraction=LowCapacity(), channel_provider=provider)
    runtime = build_runtime_scenario_config(scenario_name='live_test', scenario_type='rt_scene',
        sim_config=configs['sim'], cell_config=configs['cell'], ric_config=configs['ric'])
    sim = Simulator(runtime, build_controller_registry(configs['ric']), build_scheduler(configs['sim']),
        adapter, configs['cell'], configs['ue'], configs['beamforming'], str(tmp_path))
    session = LiveXAppSession(sim, args, tmp_path, tmp_path / 'live.json')

    def step(n):
        for _ in range(n):
            session.before_step()
            sim.step()
            session.after_step(adapter)
    return sim, session, step


def test_live_scope_delayed_execution_withdrawal_and_reenable(tmp_path):
    sim, session, step = fixture_sim(tmp_path)
    step(4)
    session.change(True, 'cell_1', sim.state.slot_id)
    step(8)
    assert sim.state.managed_cells['cell_0'].scheduler.scheduler_mode == 'round_robin'
    assert sim.state.managed_cells['cell_1'].scheduler.scheduler_mode == 'latency_aware'
    session.change(False, '*', sim.state.slot_id)
    with pytest.raises(ValueError, match='previous disable'):
        session.change(True, '*', sim.state.slot_id)
    step(9)
    assert all(r.scheduler.scheduler_mode == 'round_robin' for r in sim.state.managed_cells.values())
    assert all(r.scheduler.scheduler_policy_hint == 'balanced' for r in sim.state.managed_cells.values())
    session.change(True, 'cell_0', sim.state.slot_id)
    step(9)
    assert sim.state.managed_cells['cell_0'].scheduler.scheduler_mode == 'latency_aware'
    assert sim.state.managed_cells['cell_1'].scheduler.scheduler_mode == 'round_robin'
    assert session.snapshot()['segments'][-1]['slots'] > 0


def test_disable_invalidates_pending_selection_generation(tmp_path):
    sim, session, step = fixture_sim(tmp_path)
    step(4)
    session.change(True, 'cell_1', sim.state.slot_id)
    step(2)
    pending = [a for _, _, a in sim._pending_config_actuations if getattr(a, '_library_provenances', [])]
    assert pending
    provenance = pending[-1]._library_provenances[0]
    session.change(False, '*', sim.state.slot_id)
    assert not sim.controller_manager.instance_selection.valid(provenance)
    receipts = []
    for _ in range(10):
        step(1)
        receipts.extend(sim.state.control.actuation_outcomes)
    assert any(r.get('reason') == 'stale_xapp_selection' for r in receipts)
    assert all(r.scheduler.scheduler_mode == 'round_robin' for r in sim.state.managed_cells.values())


def test_commands_are_run_scoped_validated_and_acknowledged(tmp_path):
    sim, session, step = fixture_sim(tmp_path)
    path = tmp_path / 'live.json'
    data = dict(stage='simulating', xapp_test=session.snapshot())
    ident = submit(path, data, 'enable', 'cell_0')
    session.before_step()
    assert session.app.active_cells == {'cell_0'}
    assert session.receipts[-1]['request_id'] == ident
    assert session.receipts[-1]['status'] == 'accepted'
    atomic_json(session.command_dir / 'stale.json', dict(run_id='other_run', request_id='old',
        operation='disable', app=session.app.name, cell='*'))
    session.before_step()
    assert session.receipts[-1]['status'] == 'rejected'
    assert session.app.active_cells == {'cell_0'}
    with pytest.raises(ValueError, match='finished'):
        submit(path, {**data, 'stage': 'completed'}, 'enable')
    assert not list(session.command_dir.glob('*.json'))


def test_test_mode_is_opt_in_and_isolated_from_orchestration():
    configs = {'ric': {}}
    assert not configure(configs, {}, NS(live_xapp_test=False))
    with pytest.raises(ValueError, match='integration baseline'):
        configure(configs, {}, NS(live_xapp_test=True, phase='integration', control_mode='orchestrated'))


def test_live_rule_uses_measured_cell_inputs_and_declared_actions():
    from test_rule_based_v1 import view
    from models.policy import PolicySnapshot
    app = LiveSchedulerTestXApp(['a', 'b'])
    app.select({'a', 'b'}, {'a': 1, 'b': 1})
    o = view()
    o.ue_observations[1].queue_bytes, o.ue_observations[1].hol_delay_ms = 0, 0
    actions = {a.target_cell_id: a for a in app.run(o, PolicySnapshot()).config_actions}
    assert actions['a'].scheduler_mode == 'latency_aware'
    assert actions['b'].scheduler_mode == 'round_robin'
    assert actions['a']._library_provenance['selection_version'] == 1
    assert app.controlled_fields == ['scheduler_mode', 'scheduler_policy_hint']


def test_absent_reports_hold_instead_of_fabricating_zero_queue():
    from models.policy import PolicySnapshot
    app = LiveSchedulerTestXApp(['a'])
    app.select({'a'}, {'a': 1})
    decision = app.run(NS(slot_id=10, ue_observations=[]), PolicySnapshot())
    assert not any(a.scheduler_mode for a in decision.config_actions)
    assert app.last_decisions[0]['reason'] == 'no_ue_reports_hold'


def test_segment_metrics_exclude_transition_and_use_packet_counter_deltas(tmp_path):
    sim, session, step = fixture_sim(tmp_path)
    step(12)
    segment = session.snapshot()['segments'][0]
    assert segment['skipped_slots'] == 7
    assert segment['slots'] == 5
    assert segment['duration_ms'] == 2.5
    assert segment['throughput_mbps'] == pytest.approx(segment['delivered_bytes'] * 8 / 2500)
    session.change(True, 'cell_0', sim.state.slot_id)
    step(8)
    assert session.snapshot()['segments'][-1]['slots'] == 1


@pytest.mark.parametrize('mismatch', [False, True])
def test_paired_runner_checks_exogenous_identity_and_reports_absolute_deltas(tmp_path, monkeypatch, mismatch):
    from experiments.compare_live_xapp import compare
    import experiments.run_rt_sys as runner
    import experiments.compare_live_xapp as comparison
    monkeypatch.setattr(comparison, 'ROOT', tmp_path)
    arguments = []

    def fake_run(args):
        arguments.append(args)
        args.output.mkdir(parents=True)
        on = args.xapp_test_start == 'on'
        atomic_json(args.output / 'run_status.json', dict(stage='completed', source_sha256='same_source'))
        atomic_json(args.output / 'initial_conditions.json', dict(environment_sha256='different' if mismatch and on else 'same_environment'))
        atomic_json(args.live_state, dict(cell_metrics=[dict(cell_id='cell_0', scheduler=dict(scheduler_mode='latency_aware' if on else 'round_robin'))]))
        metrics = dict(slots=20, duration_ms=10., throughput_mbps=3. if on else 2.,
            mean_queue_bytes=10., mean_max_hol_ms=1., mean_completion_delay_ms=None, nack_fraction=0.)
        atomic_json(args.output / 'final_summary.json', dict(packet_metrics={},
            xapp_test=dict(segments=[metrics], last_decisions=[])))

    monkeypatch.setattr(runner, 'run_case', fake_run)
    spec, result = tmp_path / 'spec.json', tmp_path / 'result.json'
    atomic_json(spec, dict(arguments=dict(scene='munich', seed=73, cells=3, ues=12), slots=80))
    if mismatch:
        with pytest.raises(ValueError, match='initial conditions changed'):
            compare(spec, tmp_path / 'outputs' / 'pair', result)
    else:
        report = compare(spec, tmp_path / 'outputs' / 'pair', result)
        assert report['matched_initial_conditions']
        assert report['delta_on_minus_off']['throughput_mbps'] == 1.
        assert report['delta_on_minus_off']['mean_completion_delay_ms'] is None
    assert [a.xapp_test_start for a in arguments] == ['off', 'on']
    assert all(a.seed == 73 and a.slots == 80 and a.live_xapp_test for a in arguments)


def test_compare_client_deduplicates_running_worker_and_cleans_failed_start(tmp_path, monkeypatch):
    import os
    import experiments.live_xapp_test as module
    data = dict(xapp_test=dict(simulation_python='python', comparison_spec=str(tmp_path / 'spec.json')))
    state = tmp_path / 'live.json'
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **k: NS(pid=os.getpid()))
    assert module.start_comparison(state, data) == os.getpid()
    with pytest.raises(ValueError, match='already running'):
        module.start_comparison(state, data)
    state.with_suffix('.comparison.lock').unlink()

    def failed(*a, **k):
        raise OSError('startup failed')
    monkeypatch.setattr(module.subprocess, 'Popen', failed)
    with pytest.raises(OSError, match='startup failed'):
        module.start_comparison(state, data)
    assert not state.with_suffix('.comparison.lock').exists()
    assert json.loads(state.with_suffix('.comparison.json').read_text())['stage'] == 'failed'
