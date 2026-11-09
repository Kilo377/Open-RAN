import json
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from core.factory import build_controller_registry
from experiments.live_xapp_test import configure, submit, start_comparison
from ric.near_rt.pools import pool_catalog, build_pool_xapp
from test_live_xapp_test import fixture_sim


def empty_chongyu_pool(monkeypatch):
    """An empty catalog is a supported fixture, not the shipped pool contents."""
    import ric.near_rt.pools as pools
    original = pools.pool_catalog
    monkeypatch.setattr(pools, 'pool_catalog', lambda name: {} if name == 'chongyu' else original(name))


def test_default_pool_and_cli_override_preserve_identical_radio_and_traffic():
    from phy.rt_sys.scenario import build_case
    a = build_case(num_cells=2, num_ues=4, total_slots=20)
    b = build_case(num_cells=2, num_ues=4, total_slots=20, xapp_pool='yisong')
    assert a[1]['ric']['xapp_pool'] == 'chongyu'
    assert set(x['implementation_id'] for x in a[1]['ric']['xapps']) == {'ue_energy_rf', 'latency_scheduler_rf', 'mec_offload_budget_qgbdt', 'throughput_scheduler_mlp', 'latency_urllc_rf', 'ap_energy_rf'}
    assert b[1]['ric']['xapp_pool'] == 'yisong' and b[1]['ric']['xapps']
    assert a[0] == b[0] and a[1]['ue'] == b[1]['ue'] and a[1]['sim'] == b[1]['sim']
    with pytest.raises(ValueError, match='Unknown xApp pool'):
        build_case(xapp_pool='other')


def test_existing_catalog_is_yisong_and_chongyu_does_not_import_it():
    assert len(pool_catalog('yisong')) == 47
    assert set(pool_catalog('chongyu')) == {'ue_energy_rf', 'latency_scheduler_rf', 'mec_offload_budget_qgbdt', 'throughput_scheduler_mlp', 'latency_urllc_rf', 'ap_energy_rf'}
    assert all(v['pool'] == 'chongyu' for v in pool_catalog('chongyu').values())
    assert not set(pool_catalog('chongyu')) & set(pool_catalog('yisong'))
    assert all(v['pool'] == 'yisong' for v in pool_catalog('yisong').values())
    with pytest.raises(ValueError, match='belongs to yisong'):
        build_controller_registry(dict(xapp_pool='chongyu', xapps=[dict(type='live_scheduler_test',cell_ids=['a'])]))
    with pytest.raises(ValueError, match='different pool'):
        build_pool_xapp(dict(pool='yisong', implementation_id='any'), 'chongyu')
    with pytest.raises(ValueError, match='Unregistered implementation'):
        build_pool_xapp(dict(implementation_id='ue_fairness:equal_share'), 'chongyu')


def test_real_chongyu_factory_identity_and_no_yisong_fallback(monkeypatch):
    import ric.near_rt.pools as pools
    from ric.near_rt.pools.chongyu.ue_energy_rf import UeEnergyRF
    imports = []
    original = pools.importlib.import_module
    def guarded(name, *args, **kwargs):
        imports.append(name)
        if '.pools.yisong' in name or '.xapp_library.' in name:
            raise AssertionError('Chongyu factory attempted a Yisong fallback')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(pools.importlib, 'import_module', guarded)
    spec = pools.pool_catalog('chongyu')['ue_energy_rf']
    assert spec['factory'] == 'ric.near_rt.pools.chongyu.ue_energy_rf:UeEnergyRF'
    app = pools.build_pool_xapp(dict(pool='chongyu', implementation_id='ue_energy_rf',
                                    instance_id='ue_energy_rf', cell_ids=['cell_0']), 'chongyu')
    assert isinstance(app, UeEnergyRF)
    assert app.implementation_id == app.name == 'ue_energy_rf'
    assert app.category_id == spec['category_id'] == 'ue_energy_saving'
    assert app.descriptor()['pool'] == 'chongyu'
    assert imports == ['ric.near_rt.pools.chongyu.ue_energy_rf']
    with pytest.raises(ValueError, match='Unregistered implementation'):
        pools.build_pool_xapp(dict(implementation_id='missing', cell_ids=['cell_0']), 'chongyu')


def test_empty_pool_runs_baseline_and_rejects_activation_and_comparison(tmp_path, monkeypatch):
    empty_chongyu_pool(monkeypatch)
    sim, session, step = fixture_sim(tmp_path, pool='chongyu')
    step(12)
    state = session.snapshot()
    assert state['pool'] == 'chongyu' and state['catalog'] == [] and state['app'] is None
    assert not sim.controller_registry.xapps and not session.apps
    assert state['segments'][0]['slots'] > 0
    assert state['selected_cells'] == state['last_decisions'] == []
    assert all(c.scheduler.scheduler_mode == 'round_robin' for c in sim.state.managed_cells.values())
    spec = json.loads(session.spec_path.read_text())
    assert spec['arguments']['xapp_pool'] == 'chongyu'
    submit(tmp_path / 'live.json', dict(stage='simulating', xapp_test=state), 'activate', app='live_scheduler_test')
    step(1)
    assert session.receipts[-1]['status'] == 'rejected'
    with pytest.raises(ValueError, match='not available in this pool'):
        start_comparison(tmp_path / 'live.json', dict(xapp_test=state))


def test_empty_pool_cannot_start_on_or_accept_a_foreign_implementation(monkeypatch):
    empty_chongyu_pool(monkeypatch)
    manifest = dict(cells=[dict(cell_id='a')], slot_duration_ms=.5)
    args = NS(live_xapp_test=True, phase='integration', control_mode='baseline', xapp_pool='chongyu',
              xapp_test_start='on')
    with pytest.raises(ValueError, match='empty'):
        configure({'ric': {}}, manifest, args)
    args.xapp_test_start = 'off'
    args.xapp_test_app = 'ue_fairness:equal_share'
    with pytest.raises(ValueError, match='in chongyu pool'):
        configure({'ric': {}}, manifest, args)


def test_custom_pool_plugin_uses_existing_actuation_and_withdrawal(tmp_path, monkeypatch):
    from models.action import ConfigAction, ControlDecision
    from ric.near_rt.pools.base import PoolXApp
    import ric.near_rt.pools as pools

    class TestScheduler(PoolXApp):
        implementation_id = 'chongyu_test_scheduler'
        category_id = 'custom_scheduler'
        required_kpis = ('queue_bytes',)
        writable_actions = ('scheduler_mode', 'scheduler_policy_hint')
        description = 'Test-only plugin, never shipped in the development catalog'

        def run(self, observation, policy_snapshot):
            actions = []
            for cell in sorted(self.active_cells):
                a = ConfigAction(source_name=self.name, target_cell_id=cell,
                    scheduler_mode='latency_aware', scheduler_policy_hint='latency_bias')
                a._library_provenance = self.identity(cell)
                actions.append(a)
            return ControlDecision(config_actions=actions)

    module_name = 'ric.near_rt.pools.chongyu.test_plugin'
    module = ModuleType(module_name)
    module.TestScheduler = TestScheduler
    monkeypatch.setitem(sys.modules, module_name, module)
    original = pools.pool_catalog
    spec = dict(factory=module_name + ':TestScheduler', category_id='custom_scheduler',
        description='Test-only scheduler', commands=[], implementation_id='chongyu_test_scheduler',
        config=dict(type='pool_xapp', pool='chongyu', instance_id='chongyu_test_scheduler',
                    implementation_id='chongyu_test_scheduler'))
    monkeypatch.setattr(pools, 'pool_catalog', lambda name: {'chongyu_test_scheduler': spec}
                        if name == 'chongyu' else original(name))
    from phy.rt_sys.scenario import build_case
    _, config, _ = build_case(xapp_pool='chongyu')
    assert [a['implementation_id'] for a in config['ric']['xapps']] == ['chongyu_test_scheduler']
    sim, session, step = fixture_sim(tmp_path, pool='chongyu', app_id='chongyu_test_scheduler')
    step(4)
    assert set(session.apps) == {'chongyu_test_scheduler'}
    assert session.snapshot()['catalog'][0]['available']
    session.change(True, 'cell_1', sim.state.slot_id)
    step(8)
    assert sim.state.managed_cells['cell_1'].scheduler.scheduler_mode == 'latency_aware'
    assert sim.state.managed_cells['cell_0'].scheduler.scheduler_mode == 'round_robin'
    session.change(False, '*', sim.state.slot_id)
    step(10)
    assert all(c.scheduler.scheduler_mode == 'round_robin' for c in sim.state.managed_cells.values())
    assert session.app.identity('cell_1')['pool'] == 'chongyu'
    # A later Yisong run has its own catalog and does not inherit the custom plugin.
    other = tmp_path / 'yisong'
    other.mkdir()
    _, old_session, _ = fixture_sim(other)
    assert 'chongyu_test_scheduler' not in old_session.apps
    assert len(old_session.snapshot()['catalog']) == 47


def test_custom_rapp_catalog_cannot_fall_back_to_yisong():
    from ric.non_rt.xapp_selection_rapp import XAppSelectionRApp
    rapp = XAppSelectionRApp.__new__(XAppSelectionRApp)
    rapp.rt_xapp_pool = 'chongyu'
    rapp.rt_xapp_capabilities = []
    rapp.rt_xapp_categories = {}
    assert rapp._available_xapp_catalog() == []
    assert rapp._available_xapp_category_catalog() == []
    rapp.rt_xapp_capabilities = [dict(xapp_name='custom', category_id='custom_category',
        available=True, writable_actions=['scheduler_mode'], required_kpis=['queue_bytes'])]
    assert rapp._available_xapp_catalog()[0]['xapp_category'] == 'custom_category'
    assert rapp._available_xapp_category_catalog()[0]['example_variants'] == ['custom']


def test_cli_pool_override_and_default_outputs_are_separate(monkeypatch, capsys):
    from experiments.run_rt_sys import parse_args, run_case
    paths = []
    for pool in ['chongyu', 'yisong']:
        monkeypatch.setattr(sys, 'argv', ['run_rt_sys', '--dry-run', '--platform', 'dgx',
            '--live-xapp-test', '--xapp-pool', pool])
        run_case(parse_args())
        report = json.loads(capsys.readouterr().out)
        assert report['xapp_pool'] == pool
        assert ('pool_' + pool) in report['output']
        assert report['registered_xapps']
        if pool == 'chongyu':
            assert 'ue_energy_rf' in str(report['registered_xapps'])
            assert 'ue_fairness:equal_share' not in str(report['registered_xapps'])
        paths.append(report['output'])
    assert paths[0] != paths[1]
