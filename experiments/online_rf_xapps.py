"""Explicit run-local activation of Chongyu measured-outcome xApps."""
from pathlib import Path
import math
import json


def configure(configs, manifest, args, output, root, *, live_test_enabled=False):
    mitigation = getattr(args, 'direct_conflict_mitigation', None)
    if mitigation is not None:
        if mitigation not in ('priority', 'uniform_random'):
            raise ValueError('Unknown direct-conflict mitigation')
        window = getattr(args, 'mitigation_window_ms', 100)
        seed = getattr(args, 'mitigation_seed', None)
        seed = args.seed if seed is None else seed
        if type(seed) is not int or seed < 0 or not isinstance(window, (int, float)) or isinstance(window, bool) or not math.isfinite(window) or window <= 0:
            raise ValueError('Mitigation requires a nonnegative integer seed and positive finite window')
        configs['ric'].update(direct_conflict_mitigation=mitigation,
                              mitigation_seed=seed, mitigation_window_ms=window)
    services = configs['phy']['rt_sys'].get('ue_services', {})
    if services.get('enabled') and services.get('mec', {}).get('enabled'):
        services['task_cohort_ms'] = 100
    modes = {name: getattr(args, name + '_xapp_mode', 'off')
             for name in ('energy', 'latency', 'budget', 'throughput', 'urllc', 'ap')}
    active = {name: mode for name, mode in modes.items() if mode != 'off'}
    if 'latency' in active and 'urllc' in active:
        raise ValueError('Choose either MEC or URLLC latency scheduler for this study')
    if not active:
        return getattr(args, 'control_mode', 'baseline')
    if (live_test_enabled or getattr(args, 'control_mode', 'baseline') != 'baseline'
            or args.xapp_pool != 'chongyu'):
        raise ValueError('Learned xApp studies require the Chongyu native baseline and no unrelated controllers')
    if set(active) & {'energy', 'latency', 'budget'} and (not services.get('enabled') or not services.get('mec', {}).get('enabled')):
        raise ValueError('Energy/latency/budget xApps require enabled UE energy/DRX/MEC models')
    if 'ap' in active and not configs['phy']['rt_sys'].get('ap_services',{}).get('enabled'):
        raise ValueError('AP energy xApp requires the opt-in AP power/availability model')
    cells = [cell['cell_id'] for cell in manifest['cells']]
    ids = {'energy': ('ue_energy_rf', 'ue_energy_saving'),
           'latency': ('latency_scheduler_rf', 'latency_scheduler'),
           'budget': ('mec_offload_budget_qgbdt', 'mec_offload_budget'),
           'throughput': ('throughput_scheduler_mlp', 'throughput_scheduler'),
           'urllc': ('latency_urllc_rf', 'latency_scheduler'),
           'ap': ('ap_energy_rf', 'ap_energy_saving')}
    # Every instance supplies the same complete selection map. Different
    # categories can coexist; overlapping fields still undergo arbitration.
    bindings = {cid: {ids[name][1]: ids[name][0] for name in active} for cid in cells}
    apps = []
    for name, mode in active.items():
        model_path = getattr(args, name + '_xapp_model', None)
        suffix = '_qgbdt_model.json' if name == 'budget' else '_mlp_model.json' if name == 'throughput' else '_rf_model.json'
        default_model = name + suffix
        model_path = Path(model_path or output / default_model).resolve()
        if not model_path.is_relative_to(root / 'outputs'):
            raise ValueError(name.title() + ' model must stay under outputs')
        parameters = dict(mode=mode, model_path=str(model_path),
            log_path=str(output / (name + '_rf_events.jsonl')),
            window_ms=getattr(args, name + '_xapp_window_ms', 100),
            explore_windows=getattr(args, name + '_xapp_explore_windows', 8 if name == 'latency' else 9), seed=args.seed)
        if name == 'energy':
            parameters['deploy_drx'] = getattr(args, 'energy_xapp_deploy_drx', True)
            action_set = getattr(args, 'energy_xapp_action_set_version', 'legacy9')
            if action_set not in ('legacy9', 'extended24', 'balanced42'):raise ValueError('Unknown energy action set')
            if action_set != 'legacy9':parameters['action_set_version'] = action_set
        if name=='latency' and getattr(args,'latency_xapp_policy',None):
            if mode!='frozen':raise ValueError('An adaptive policy requires explicit frozen mode; training is controlled by its versioned artifact')
            parameters.pop('explore_windows',None)
            parameters['adaptive_policy_path']=str(Path(args.latency_xapp_policy).resolve())
            parameters['complete_scheduler_bundle']=True
        if name == 'latency' and mode == 'fixed':
            parameters['fixed_action'] = getattr(args, 'latency_xapp_fixed_action', 3)
        if name == 'budget':
            parameters.pop('explore_windows')
            parameters['log_path'] = str(output / 'budget_qgbdt_events.jsonl')
            parameters['budget_gflops'] = getattr(args, 'budget_xapp_gflops', .75)
            profile = services.get('mec', {}).get('tasks_by_ue', {}).get('ue_4', {})
            keys = ('interval_ms', 'compute_gflop', 'deadline_ms')
            if any(type(profile.get(k)) not in (int, float) or not math.isfinite(profile[k]) or profile[k] <= 0 for k in keys):
                raise ValueError('Budget xApp requires the actual positive UE4 task profile')
            parameters['task_profile'] = {k: profile[k] for k in keys}
        if name == 'throughput':
            parameters.pop('explore_windows')
            parameters['log_path'] = str(output / 'throughput_mlp_events.jsonl')
            parameters['fixed_action'] = getattr(args, 'throughput_xapp_fixed_action', 3)
            parameters['policy'] = getattr(args, 'throughput_xapp_policy', 'throughput')
        if name == 'urllc':
            parameters.pop('explore_windows')
            parameters['fixed_action'] = getattr(args, 'urllc_xapp_fixed_action', 3)
            if getattr(args, 'urllc_xapp_policy_version', 'legacy') == 'qos_v2':
                parameters['policy_version'] = 'qos_v2'
                parameters['balanced_start_windows'] = getattr(args, 'urllc_xapp_balanced_start_windows', 0)
        if name == 'ap':
            parameters.pop('explore_windows')
            parameters['fixed_action']=getattr(args,'ap_xapp_fixed_action',1)
        ident, _ = ids[name]
        apps.append(dict(type='pool_xapp', pool='chongyu', implementation_id=ident,
            instance_id=ident, cell_ids=cells, slot_duration_ms=manifest['slot_duration_ms'],
            parameters=parameters, initial_bindings=bindings))
        if name == 'energy' and action_set != 'legacy9':apps[-1]['implementation_variant']=action_set
        if name=='latency' and parameters.get('adaptive_policy_path'):
            apps[-1]['implementation_variant']='adaptive_v2'
    configs['ric'].update(xapp_pool='chongyu', rapps=[], run_rapps_on_start=False, freeze_learning=True,
        xapp_interval_slots=max(1, round(20 / manifest['slot_duration_ms'])), xapps=apps)
    if 'latency' in active and getattr(args,'latency_xapp_policy',None):
        configs['ric']['freeze_learning']=not json.loads(Path(args.latency_xapp_policy).read_text()).get('training',False)
    return '_'.join(name + '_xapp_' + mode for name, mode in active.items())
