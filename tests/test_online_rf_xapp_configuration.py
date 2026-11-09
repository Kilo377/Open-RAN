from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from experiments.online_rf_xapps import configure


def setup(tmp_path, **overrides):
    configs = {'ric': {'rapps': ['original'], 'xapps': []}, 'phy': {'rt_sys': {
        'ue_services': {'enabled': True, 'mec': {'enabled': True,
            'tasks_by_ue': {'ue_4': dict(interval_ms=50, compute_gflop=.05, deadline_ms=100)}}}}}}
    manifest = {'cells': [{'cell_id': 'cell_0'}, {'cell_id': 'cell_1'}],
                'slot_duration_ms': .5}
    args = NS(control_mode='baseline', xapp_pool='chongyu', seed=42,
              energy_xapp_mode='off', latency_xapp_mode='off', **overrides)
    return configs, manifest, args, tmp_path / 'outputs' / 'case', tmp_path


def test_native_defaults_stay_off_with_matched_task_observability(tmp_path):
    configs, manifest, args, output, root = setup(tmp_path)
    assert configure(configs, manifest, args, output, root) == 'baseline'
    assert configs['ric']['xapps'] == []
    assert configs['ric']['rapps'] == ['original']
    assert configs['phy']['rt_sys']['ue_services']['task_cohort_ms'] == 100


@pytest.mark.parametrize('active', [('energy',), ('latency',), ('budget',), ('throughput',),
                                  ('energy', 'latency'), ('energy', 'budget'),
                                  ('energy', 'throughput'), ('latency', 'throughput'),
                                  ('energy', 'latency', 'budget'), ('energy', 'latency', 'budget', 'throughput')])
def test_shared_complete_bindings_allow_disjoint_categories_together(tmp_path, active):
    configs, manifest, args, output, root = setup(tmp_path)
    for name in active:
        setattr(args, name + '_xapp_mode', 'online')
    configure(configs, manifest, args, output, root)
    apps = configs['ric']['xapps']
    assert len(apps) == len(active)
    assert all(app['initial_bindings'] == apps[0]['initial_bindings'] for app in apps)
    assert len(apps[0]['initial_bindings']['cell_0']) == len(active)
    assert configs['ric']['rapps'] == []
    assert configs['ric']['xapp_interval_slots'] == 40
    for name, app in zip(active, apps):
        assert app['parameters']['mode'] == 'online'
        suffix = '_qgbdt_model.json' if name == 'budget' else '_mlp_model.json' if name == 'throughput' else '_rf_model.json'
        assert Path(app['parameters']['model_path']) == output / (name + suffix)


def test_budget_uses_actual_configured_task_profile_and_opt_in_arbitration(tmp_path):
    configs, manifest, args, output, root = setup(tmp_path)
    args.budget_xapp_mode = 'rules'
    args.direct_conflict_mitigation = 'uniform_random'
    args.mitigation_seed = 19
    args.mitigation_window_ms = 100
    profile = configs['phy']['rt_sys']['ue_services']['mec']['tasks_by_ue']['ue_4']
    profile['offload_fraction'] = .5
    configure(configs, manifest, args, output, root)
    app = configs['ric']['xapps'][0]
    assert app['parameters']['task_profile'] == {k: profile[k] for k in ('interval_ms', 'compute_gflop', 'deadline_ms')}
    assert app['parameters']['log_path'].endswith('budget_qgbdt_events.jsonl')
    assert 'explore_windows' not in app['parameters']
    assert configs['ric']['mitigation_seed'] == 19
    assert configs['ric']['direct_conflict_mitigation'] == 'uniform_random'


def test_explicit_budget_activation_requires_real_ue4_task_profile(tmp_path):
    configs, manifest, args, output, root = setup(tmp_path)
    args.budget_xapp_mode = 'online'
    configs['phy']['rt_sys']['ue_services']['mec']['tasks_by_ue'] = {}
    with pytest.raises(ValueError, match='actual positive UE4'):
        configure(configs, manifest, args, output, root)


def test_energy_and_budget_categories_share_selection_but_contest_ue4_rcp(tmp_path):
    from core.factory import build_controller_registry
    configs, manifest, args, output, root = setup(tmp_path)
    args.energy_xapp_mode = args.budget_xapp_mode = 'online'
    configure(configs, manifest, args, output, root)
    for app in configs['ric']['xapps']:
        app['parameters'].pop('model_path')
        app['parameters'].pop('log_path')
    registry = build_controller_registry(configs['ric'])
    energy, budget = registry.xapps
    assert set(energy.controlled_fields) & set(budget.controlled_fields) == {'ue_offload_fraction'}
    assert energy.initial_bindings == budget.initial_bindings
    assert set(energy.initial_bindings['cell_0']) == {'ue_energy_saving', 'mec_offload_budget'}


def test_throughput_does_not_require_ue_energy_or_mec_models(tmp_path):
    from core.factory import build_controller_registry
    configs, manifest, args, output, root = setup(tmp_path)
    configs['phy']['rt_sys']['ue_services'] = {'enabled': False}
    configs['phy']['rt_sys']['paper_interfaces'] = {'enabled': True, 'rule_observations': True}
    args.throughput_xapp_mode = 'fixed'
    args.throughput_xapp_fixed_action = 1
    configure(configs, manifest, args, output, root)
    item = configs['ric']['xapps'][0]
    assert item['parameters']['mode'] == 'fixed' and item['parameters']['fixed_action'] == 1
    assert item['parameters']['policy'] == 'throughput'
    assert item['parameters']['log_path'].endswith('throughput_mlp_events.jsonl')
    assert 'explore_windows' not in item['parameters'] and 'task_profile' not in item['parameters']
    item['parameters'].pop('model_path'); item['parameters'].pop('log_path')
    app = build_controller_registry(configs['ric']).xapps[0]
    assert app.implementation_id == 'throughput_scheduler_mlp'
    assert set(app.controlled_fields) == {'scheduler_mode', 'ue_scheduler_weights'}
    assert configs['phy']['rt_sys']['ue_services'] == {'enabled': False}


@pytest.mark.parametrize('policy', ['guarded', 'throughput'])
def test_throughput_policy_is_forwarded_only_to_x08(tmp_path, policy):
    configs, manifest, args, output, root = setup(tmp_path)
    args.energy_xapp_mode = args.throughput_xapp_mode = 'online'
    args.throughput_xapp_policy = policy
    configure(configs, manifest, args, output, root)
    energy, throughput = configs['ric']['xapps']
    assert 'policy' not in energy['parameters']
    assert throughput['parameters']['policy'] == policy
    from experiments.live_xapp_test import ARGUMENTS
    assert 'throughput_xapp_policy' in ARGUMENTS


def test_throughput_policy_cli_default_and_invalid_choice(monkeypatch):
    from experiments.run_rt_sys import parse_args
    monkeypatch.setattr('sys.argv', ['run_rt_sys.py'])
    assert parse_args().throughput_xapp_policy == 'throughput'
    monkeypatch.setattr('sys.argv', ['run_rt_sys.py', '--throughput-xapp-policy', 'invalid-policy'])
    with pytest.raises(SystemExit):
        parse_args()


@pytest.mark.parametrize('policy', [None, 'guarded', 'throughput'])
def test_live_policy_updates_only_the_actual_throughput_app(tmp_path, monkeypatch, policy):
    from experiments.live_xapp_test import configure as configure_live
    import ric.near_rt.pools as pools
    configs, manifest, args, _, _ = setup(tmp_path)
    original = [dict(implementation_id='ue_energy_rf', instance_id='ue_energy_rf', parameters={'mode':'frozen'}),
                dict(implementation_id='throughput_scheduler_mlp', instance_id='throughput_scheduler_mlp', parameters={'mode':'frozen'})]
    monkeypatch.setattr(pools, 'live_configs', lambda *unused: deepcopy(original))
    args.live_xapp_test = True
    if policy is not None:
        args.throughput_xapp_policy = policy
    assert configure_live(configs, manifest, args)
    energy, throughput = configs['ric']['xapps']
    assert energy == original[0]
    assert throughput['parameters'] == {'mode':'frozen', 'policy':policy or 'throughput'}


def test_throughput_and_latency_share_rcps_in_distinct_categories(tmp_path):
    from core.factory import build_controller_registry
    configs, manifest, args, output, root = setup(tmp_path)
    args.throughput_xapp_mode = args.latency_xapp_mode = 'online'
    configure(configs, manifest, args, output, root)
    for item in configs['ric']['xapps']:
        item['parameters'].pop('model_path'); item['parameters'].pop('log_path')
    latency, throughput = build_controller_registry(configs['ric']).xapps
    assert latency.category_id != throughput.category_id
    assert set(latency.controlled_fields) & set(throughput.controlled_fields) == {'scheduler_mode', 'ue_scheduler_weights'}
    assert latency.initial_bindings == throughput.initial_bindings


@pytest.mark.parametrize('invalid', ['foreign_pool', 'live', 'orchestrated', 'services_off'])
def test_study_cannot_silently_mix_unrelated_controllers_or_missing_models(tmp_path, invalid):
    configs, manifest, args, output, root = setup(tmp_path)
    args.latency_xapp_mode = 'online'
    if invalid == 'foreign_pool': args.xapp_pool = 'yisong'
    if invalid == 'orchestrated': args.control_mode = 'orchestrated'
    if invalid == 'services_off': configs['phy']['rt_sys']['ue_services']['enabled'] = False
    with pytest.raises(ValueError):
        configure(configs, manifest, args, output, root, live_test_enabled=invalid == 'live')


def test_models_remain_scoped_to_this_project_outputs(tmp_path):
    configs, manifest, args, output, root = setup(tmp_path)
    args.latency_xapp_mode = 'frozen'
    args.latency_xapp_model = tmp_path / 'foreign.json'
    with pytest.raises(ValueError, match='under outputs'):
        configure(configs, manifest, args, output, root)


def test_registered_controllers_have_disjoint_rcps_and_shared_initial_selection(tmp_path):
    from core.factory import build_controller_registry
    configs, manifest, args, output, root = setup(tmp_path)
    # No fitting or inference is involved in initializing a native online run.
    args.energy_xapp_mode = args.latency_xapp_mode = 'online'
    configure(configs, manifest, args, output, root)
    for app in configs['ric']['xapps']:
        # Artifacts are not needed for this registry/selection integration test.
        app['parameters'].pop('model_path')
        app['parameters'].pop('log_path')
    registry = build_controller_registry(configs['ric'])
    energy, latency = registry.xapps
    assert not set(energy.controlled_fields) & set(latency.controlled_fields)
    assert energy.initial_bindings == latency.initial_bindings
    from ric.near_rt.xapp_library.selection import InstanceSelection
    manager = NS(e2_interfaces=[NS(xapp=app) for app in registry.xapps])
    selection = InstanceSelection(manager)
    assert selection.default_bindings() == energy.initial_bindings
    assert set(selection.default_bindings()['cell_0']) == {'ue_energy_saving', 'latency_scheduler'}


def test_urllc_and_throughput_bind_independent_categories_without_mec(tmp_path):
    from core.factory import build_controller_registry
    configs,manifest,args,output,root=setup(tmp_path)
    configs['phy']['rt_sys']['ue_services']={'enabled':False}
    args.urllc_xapp_mode=args.throughput_xapp_mode='online'
    configure(configs,manifest,args,output,root)
    assert len(configs['ric']['xapps'])==2
    assert set(configs['ric']['xapps'][0]['initial_bindings']['cell_0'])=={'latency_scheduler','throughput_scheduler'}
    for item in configs['ric']['xapps']:
        item['parameters'].pop('model_path');item['parameters'].pop('log_path')
    throughput,urllc=build_controller_registry(configs['ric']).xapps
    assert urllc.implementation_id=='latency_urllc_rf'
    assert set(throughput.controlled_fields)==set(urllc.controlled_fields)=={'scheduler_mode','ue_scheduler_weights'}
    assert 'explore_windows' not in urllc.parameters


def test_mec_and_urllc_latency_variants_cannot_silently_share_one_binding(tmp_path):
    configs,manifest,args,output,root=setup(tmp_path)
    args.urllc_xapp_mode=args.latency_xapp_mode='online'
    with pytest.raises(ValueError,match='either MEC or URLLC'):
        configure(configs,manifest,args,output,root)


def test_ap_energy_requires_actual_opt_in_model_and_coexists_with_scheduler(tmp_path):
    from core.factory import build_controller_registry
    configs,manifest,args,output,root=setup(tmp_path)
    args.ap_xapp_mode='fixed';args.ap_xapp_fixed_action=1
    args.throughput_xapp_mode='fixed'
    with pytest.raises(ValueError,match='opt-in AP'):
        configure(configs,manifest,args,output,root)
    configs['phy']['rt_sys']['ap_services']={'enabled':True}
    configure(configs,manifest,args,output,root)
    for item in configs['ric']['xapps']:
        item['parameters'].pop('model_path');item['parameters'].pop('log_path')
    throughput,ap=build_controller_registry(configs['ric']).xapps
    assert ap.implementation_id=='ap_energy_rf' and ap.parameters['fixed_action']==1
    assert ap.controlled_fields==['cell_sleep_mode'] or set(ap.controlled_fields)=={'cell_sleep_mode'}
    assert not set(ap.controlled_fields)&set(throughput.controlled_fields)
    assert set(ap.initial_bindings['cell_0'])=={'throughput_scheduler','ap_energy_saving'}
    assert 'explore_windows' not in ap.parameters


def test_ap_cli_is_off_by_default_and_profile_is_explicit(monkeypatch):
    from experiments.run_rt_sys import parse_args
    monkeypatch.setattr('sys.argv',['run_rt_sys.py'])
    args=parse_args();assert args.ap_xapp_mode=='off' and args.ap_services_profile is None
    monkeypatch.setattr('sys.argv',['run_rt_sys.py','--ap-services-profile','configs/rt_sys/ap_services_reference.json',
        '--ap-xapp-mode','fixed','--ap-xapp-fixed-action','1'])
    args=parse_args();assert args.ap_xapp_fixed_action==1 and args.ap_services_profile.name=='ap_services_reference.json'
