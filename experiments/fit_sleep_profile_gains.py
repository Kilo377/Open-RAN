"""Fit independently paired sleep-profile contrasts without reading validation.

The intervention is one complete cell profile, not three additive per-cell
effects. Only P–K relations already admitted in the frozen graph can receive
another conditional estimate. Held-out combination data are never an input.
"""
from copy import deepcopy
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from functools import lru_cache

from experiments.context_identification import SCENARIOS
from experiments.profile_calibration_audit import audit_pair
from experiments.sleep_profile_calibration import verify_plan
from smo.contracts import validate_influences
from smo.context_identify import importance
from smo.gain_audit import audit_gains
from smo.identify import paired_summary
from smo.influence_units import align_graph, factor
from smo.window_metrics import RecordedWindows


# RecordedWindows has fixed canonical output coordinates; a graph's display
# unit must never silently relabel those numeric observations.
METRIC_UNITS = dict(ue_power_w='W', ue_energy_j='J', ap_power_w='W', network_goodput_mbps='Mbit/s',
    per_ue_goodput_mbps='Mbit/s', urllc_failure_events_per_s='events/s',
    ue_active_fraction='ratio', ap_available_fraction='ratio', queue_overflow_bytes='byte',
    mec_realized_gflops='GFLOP/s', urllc_failure_fraction='ratio',
    mec_offloaded_latency_ms='ms', task_latency_ms='ms', task_success_ratio='ratio',
    urllc_packet_latency_ms='ms', packet_latency_ms='ms', mec_expected_gflops='GFLOP/s',
    queue_bytes='byte', hol_delay_ms='ms')


def _arrival_recipe(configs):
    from experiments.urllc_traffic_profile import FIELDS
    users = configs.get('ue', {}).get('ues', [])
    ids = [u.get('ue_id') for u in users]
    if not ids or any(not isinstance(u, str) for u in ids) or len(ids) != len(set(ids)):
        raise ValueError('Calibration scene lacks unique UE arrival configurations')
    return {u['ue_id']: dict(arrivals={k:deepcopy(u[k]) for k in FIELDS if k in u},
        logical_channels=[{k:deepcopy(lc[k]) for k in FIELDS if k in lc}
                          for lc in u.get('logical_channels', [])]) for u in users}


@lru_cache(maxsize=None)
def _expected_scene(scene):
    """Rebuild configuration only, never run a simulator or inspect outcomes."""
    from phy.rt_sys.scenario import build_case
    from experiments.urllc_scenarios import apply_named_scenario
    if scene not in SCENARIOS:
        raise ValueError('Unregistered calibration scene')
    context = SCENARIOS[scene]
    config_path = Path(__file__).resolve().parents[1]/'configs/rt_sys/agx_munich_6ue_depth2_fast.json'
    manifest, configs, _ = build_case('munich', 3, 6, 1, 1, device='cpu',
        config_path=config_path, traffic_load=context['traffic_load'], xapp_pool='chongyu')
    apply_named_scenario(manifest, configs, context['urllc_scenario'])
    return dict(urllc_scenario=manifest['urllc_scenario'],
        traffic_profile_sha256=manifest.get('traffic_profile', {}).get('sha256'),
        arrivals=_arrival_recipe(configs))


def _verify_scene(config, scene):
    expected = _expected_scene(scene)
    manifest = config.get('manifest', {})
    actual = dict(urllc_scenario=manifest.get('urllc_scenario'),
        traffic_profile_sha256=manifest.get('traffic_profile', {}).get('sha256'),
        arrivals=_arrival_recipe(config.get('configs', {})))
    if actual != expected:
        raise ValueError('Calibration actual arrival recipe differs from registered scene')


def _in_graph_unit(reading, key, unit):
    try:
        source = METRIC_UNITS[key]
        multiplier = factor(source, unit)
    except (KeyError, ValueError) as error:
        raise ValueError('Calibration metric has no compatible canonical unit: '+key) from error
    result = deepcopy(reading)
    result.update(unit=unit, source_canonical_unit=source,
        source_value=reading.get('value'), metric_unit_conversion_factor=multiplier)
    for field in ('value', 'resolution', 'minimum', 'maximum'):
        if finite(reading.get(field)):
            result[field] = reading[field]*multiplier
    # These describe the original ratio presentation, not converted values.
    for field in ('canonical_unit', 'display_unit', 'display_multiplier'):
        result.pop(field, None)
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def topology(graph):
    return sorted((e['id'], e['source'], e['target'], e['kind']) for e in graph['edges'])


def metrics(run, keys, start, end):
    run = Path(run)
    index = RecordedWindows(run, read(run/'config_snapshot.json'))
    index.sync()
    return index.window(keys, start, end, through=end)['states']


def _verify_seal(case, plan):
    run = Path(case['run_path'])
    config = read(run/'config_snapshot.json')
    probe = config.get('sleep_profile_probe', {})
    if (config.get('control_mode') != 'sleep_profile_probe'
            or config.get('manifest', {}).get('rng_seed') != case['seed']
            or case['seed'] not in plan['training_seeds']
            or case['seed'] in plan['validation_seeds_excluded']
            or probe.get('calibration_plan_sha256') != plan['plan_sha256']
            or probe.get('profile_id') != case['arm']
            or config.get('source_sha256') != case['source_sha256']):
        raise ValueError('Calibration configuration is not a registered training probe')
    _verify_scene(config, case['scene'])
    worker = run.parents[3]
    seal = read(worker/'checkpoints'/(case['id']+'.json'))
    identity = seal.get('case', {})
    expected_profile = (plan['reference_profile'] if case['arm'] == 'REF'
                        else plan['profiles'][case['scene']][case['arm']])
    if (seal.get('completed') is not True or seal.get('source_sha256') != case['source_sha256']
            or any(identity.get(key) != case.get(key) for key in ('id', 'scene', 'seed', 'arm'))
            or identity.get('profile') != expected_profile
            or any(key in case and identity.get(key) != case[key] for key in ('name', 'origin_key', 'profile'))
            or seal.get('physical_plan_sha256') != plan['plan_sha256']
            or seal.get('heldout_after_kpis_read') is not False):
        raise ValueError('Unsealed or mismatched independent calibration run')
    job = run.parents[1]
    pins = seal.get('files_sha256', {})
    required = {'data/run/config_snapshot.json', 'data/run/run_status.json',
                'data/parameter-admission-proof.json'}
    if not required <= pins.keys() or not any(p.startswith('data/run/rt_sys_trace.jsonl') for p in pins):
        raise ValueError('Calibration seal lacks required evidence files')
    for name, pin in pins.items():
        path = job/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != pin:
            raise ValueError('Calibration evidence changed after sealing: '+name)
    return seal


def collect_pairs(manifest, plan, graphs):
    """Audit all registered runs before extracting matched post-settle metrics."""
    if (manifest.get('schema') != 'oran.profile-calibration-cases.v1'
            or manifest.get('calibration_plan_sha256') != plan.get('plan_sha256')
            or manifest.get('heldout_after_kpis_read') is not False):
        raise ValueError('Independent calibration manifest contract differs')
    expected_window = dict(start_slot=plan['scoring_start_slot'], end_slot=plan['scoring_end_slot'],
                           slot_duration_ms=plan['slot_duration_ms'])
    if manifest.get('scoring_window') != expected_window:
        raise ValueError('Paired scoring window differs from preregistration')
    cases = manifest['cases']
    identities = [(c['scene'], c['seed'], c['arm']) for c in cases]
    expected = {(scene, seed, arm) for scene, profiles in plan['profiles'].items()
                for seed in plan['training_seeds'] for arm in ['REF', *profiles]}
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError('Complete, unique registered calibration cases required')
    # Validate the same immutable protocol as the workers before opening any
    # run. A self-consistent replacement manifest cannot redefine validation
    # seeds as training seeds simply by supplying a different plan digest.
    verify_plan(plan, plan['plan_sha256'])
    if (len({c['id'] for c in cases}) != len(cases)
            or len({str(Path(c['run_path']).resolve()) for c in cases}) != len(cases)):
        raise ValueError('Independent calibration runs must have unique IDs and paths')
    if len({c['source_sha256'] for c in cases}) != 1:
        raise ValueError('Independent calibration source versions differ')
    lookup = {key: case for key, case in zip(identities, cases)}
    seals = {case['id']: _verify_seal(case, plan) for case in cases}
    pairs = []
    for scene, profiles in plan['profiles'].items():
        keys = [n['id'][4:] for n in graphs[scene]['nodes'] if n['kind'] == 'kpi']
        for seed in plan['training_seeds']:
            reference = lookup[scene, seed, 'REF']
            reference_metrics = None
            for profile_id in profiles:
                profile = lookup[scene, seed, profile_id]
                proof = audit_pair(reference['run_path'], profile['run_path'], plan, scene, seed)
                if reference_metrics is None:
                    reference_metrics = metrics(reference['run_path'], keys, plan['scoring_start_slot'], plan['scoring_end_slot'])
                actual = metrics(profile['run_path'], keys, plan['scoring_start_slot'], plan['scoring_end_slot'])
                pairs.append(dict(scene=scene, seed=seed, profile_id=profile_id, reference=reference_metrics,
                    profile=actual, audit=proof, reference_case_id=reference['id'], profile_case_id=profile['id'],
                    checkpoint_sha256={case['id']: digest(seals[case['id']]) for case in (reference, profile)}))
    return pairs


def estimate_rows(pairs, plan, scene, graph, data_sha256):
    """One paired observation per independent seed; missing means stay missing."""
    nodes = {n['id'][4:]: n for n in graph['nodes'] if n['kind'] == 'kpi'}
    rows, missing = [], []
    for profile_id, profile in plan['profiles'][scene].items():
        selected = [p for p in pairs if p['scene'] == scene and p['profile_id'] == profile_id]
        if sorted(p['seed'] for p in selected) != sorted(plan['training_seeds']):
            raise ValueError('Every preregistered training seed must be retained')
        for key, node in nodes.items():
            observations = []
            for pair in selected:
                low, high = [_in_graph_unit(pair[side]['kpi:'+key], key, node['unit'])
                             for side in ('reference', 'profile')]
                valid = finite(low.get('value')) and finite(high.get('value'))
                observations.append(dict(seed=pair['seed'], low=deepcopy(low), high=deepcopy(high), valid=valid,
                    paired_difference=high['value']-low['value'] if valid else None,
                    pair_audit_sha256=digest(pair['audit'])))
            valid = [o for o in observations if o['valid']]
            if len(valid) < 2:
                missing.append(dict(scene=scene, profile_id=profile_id, kpi=key,
                    reason='fewer_than_two_valid_independent_pairs', seed_pairs=observations))
                continue
            row = dict(parameter='cell_sleep_mode', kpi=key, contrast_id=profile_id+':'+key,
                kind='finite_contrast', parameter_component='joint_cell_sleep_profile',
                members=sorted(profile), population_control_eligible=True,
                unit=node['unit']+' / contrast', sample_count=len(valid), observed_run_count=2*len(valid),
                source_metric_unit=METRIC_UNITS[key],
                metric_unit_conversion_factor=factor(METRIC_UNITS[key], node['unit']),
                independent_unit='seed_pair', confidence_level=.95,
                scope='joint fixed sleep profile across all three cells', context=scene,
                method='independent-seed paired post-settle whole-profile contrast',
                design='controlled_perturbation', data_sha256=data_sha256,
                horizon_ms=(plan['scoring_end_slot']-plan['switch_slot'])*plan['slot_duration_ms'],
                scoring_start_ms=plan['scoring_start_slot']*plan['slot_duration_ms'],
                scoring_end_ms=plan['scoring_end_slot']*plan['slot_duration_ms'],
                scoring_window_ms=(plan['scoring_end_slot']-plan['scoring_start_slot'])*plan['slot_duration_ms'],
                metric_window=dict(kind=valid[0]['low'].get('aggregation'),
                    start_slot=plan['scoring_start_slot'], end_slot=plan['scoring_end_slot'],
                    matched_estimator='smo.window_metrics.RecordedWindows'),
                reference_kpi_mean=statistics.mean(o['low']['value'] for o in valid),
                seed_pairs=observations, missing_pair_count=len(observations)-len(valid),
                from_profile_sha256=digest(plan['reference_profile']), to_profile_sha256=digest(profile),
                calibration_plan_sha256=plan['plan_sha256'],
                validation_combination_outcomes_used=False,
                caveat='A whole-profile finite contrast conditional on this scene and horizon; not additive per-cell gains.',
                **paired_summary([o['paired_difference'] for o in valid]))
            row['from'], row['to'] = deepcopy(plan['reference_profile']), deepcopy(profile)
            ci = row['confidence_interval']
            significant = ci[0] > 0 or ci[1] < 0
            zero = row['estimate'] == 0 and ci == [0, 0]
            row['statistical_context'] = dict(direction_significant=significant, confidence_level=.95,
                screening_reason=None if significant else 'observed_zero' if zero else 'interval_includes_zero',
                retained_as='significant_reference' if significant else 'observed_zero_reference' if zero else 'uncertain_reference',
                relation_admission_inherited=True)
            row['importance'] = importance(row)
            if key == 'urllc_failure_fraction':
                probability_span = factor('ratio', node['unit'])
                row['importance'] = dict(signed_reference_importance=row['estimate']/probability_span, parameter_span=1.,
                    kpi_scale=probability_span, normalization_basis='canonical_probability_range')
            rows.append(row)
    return rows, missing


def overlay(graph, rows, estimates_path, *, provenance):
    """Append conditional rows only; never admit another graph relation."""
    admitted = {(e['source'], e['target']) for e in graph['edges'] if e['kind'] == 'influences'}
    accepted, excluded = [], []
    for row in rows:
        pair = ('rcp:'+row['parameter'], 'kpi:'+row['kpi'])
        (accepted if pair in admitted else excluded).append(deepcopy(row))
    write(estimates_path, dict(schema='oran.intervention-edges.v1', estimates=accepted, provenance=provenance))
    base = deepcopy(graph)
    for edge in base['edges']:
        if edge['kind'] == 'influences':
            edge.setdefault('source_files', [])
    result = align_graph(validate_influences(base, estimates_path))
    for edge in result['edges']:
        if edge['kind'] != 'influences' or not edge.get('statistical_selection'):
            continue
        added = [r for r in accepted if 'rcp:'+r['parameter']==edge['source'] and 'kpi:'+r['kpi']==edge['target']]
        selection = edge['statistical_selection']
        for row in added:
            selection.setdefault('retained_contrasts', []).append(row['contrast_id'])
            field = dict(significant_reference='significant_contrasts',
                uncertain_reference='uncertain_contrasts', observed_zero_reference='observed_zero_contrasts')[
                    row['statistical_context']['retained_as']]
            selection.setdefault(field, []).append(row['contrast_id'])
        selection['independent_profile_contexts_appended'] = [r['contrast_id'] for r in added]
    if result.get('statistical_filter'):
        summary = result['statistical_filter']
        for field in ('original_estimate_count', 'active_estimate_count'):
            if field in summary:
                summary[field] += len(accepted)
        for state, field in [('significant_reference', 'significant_estimate_count'),
                             ('uncertain_reference', 'retained_uncertain_estimate_count'),
                             ('observed_zero_reference', 'retained_observed_zero_estimate_count')]:
            if field in summary:
                summary[field] += sum(r['statistical_context']['retained_as'] == state for r in accepted)
        summary['independent_profile_contexts_appended'] = len(accepted)
    if topology(result) != topology(graph):
        raise AssertionError('Independent calibration changed frozen graph topology')
    result['profile_calibration'] = dict(**deepcopy(provenance), new_conditional_estimates=len(accepted),
        excluded_nonadmitted_pairs=[r['kpi'] for r in excluded], topology_unchanged=True)
    audit = audit_gains(result, all_kpis=True)
    if not audit['all_link_gains_complete']:
        raise ValueError('Overlay has invalid gain contracts: '+str(audit['invalid_edge_ids']))
    return result, excluded, audit


def analyze(manifest_path, plan_path, base_graphs_path, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new immutable profile-gain analysis directory')
    manifest, plan, graph_paths = map(read, (manifest_path, plan_path, base_graphs_path))
    graphs = {scene: read(path) for scene, path in graph_paths.items()}
    if set(graphs) != set(plan['profiles']):
        raise ValueError('One frozen graph per registered scene required')
    root = Path(__file__).resolve().parents[1]
    dependencies = ('experiments/fit_sleep_profile_gains.py', 'experiments/profile_calibration_audit.py',
        'experiments/sleep_profile_calibration.py', 'experiments/context_identification.py',
        'experiments/urllc_scenarios.py', 'experiments/urllc_traffic_profile.py',
        'phy/rt_sys/scenario.py', 'smo/window_metrics.py', 'smo/identify.py',
        'smo/influence_units.py', 'smo/context_identify.py', 'smo/contracts.py',
        'smo/gain_audit.py', 'configs/rt_sys/agx_munich_6ue_depth2_fast.json')
    code = {name:(root/name).read_bytes() for name in dependencies}
    code_pins = {name:hashlib.sha256(raw).hexdigest() for name, raw in code.items()}
    pairs = collect_pairs(manifest, plan, graphs)
    provenance = dict(schema='oran.independent-profile-gain-provenance.v1',
        calibration_plan_sha256=plan['plan_sha256'], cases_manifest_sha256=digest(manifest),
        paired_data_sha256=digest(pairs), fitter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        analysis_code_sha256=code_pins,
        base_graph_sha256={scene: digest(g) for scene, g in graphs.items()},
        training_seeds=plan['training_seeds'], validation_seeds_excluded=plan['validation_seeds_excluded'],
        heldout_validation_data_read=False, score_weights_fitted=False)
    results, graph_map = {}, {}
    for scene, graph in graphs.items():
        rows, missing = estimate_rows(pairs, plan, scene, graph, provenance['paired_data_sha256'])
        updated, excluded, audit = overlay(graph, rows, output/scene/'influences.json', provenance=provenance)
        graph_path = output/scene/'graph.json'
        write(graph_path, updated)
        write(output/scene/'all-estimates.json', dict(estimates=rows, unmeasurable=missing,
            excluded_nonadmitted_pairs=[r['kpi'] for r in excluded]))
        write(output/scene/'gain-audit.json', audit)
        graph_map[scene] = str(graph_path)
        results[scene] = dict(estimated=len(rows), appended=len(rows)-len(excluded), unmeasurable=len(missing),
            graph_sha256=digest(updated), topology_unchanged=topology(updated)==topology(graph))
    write(output/'paired-data.json', pairs)
    write(output/'graph-map.json', graph_map)
    for name, raw in code.items():
        if (root/name).read_bytes() != raw:
            raise ValueError('Analysis dependency changed while fitting: '+name)
        target = output/'pipeline-sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    write(output/'analysis-seal.json', dict(provenance=provenance, scenes=results,
        files_sha256={str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(output.rglob('*')) if p.is_file()}))
    return dict(output=str(output), scenes=results, graph_map=str(output/'graph-map.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--base-graphs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.manifest, args.plan, args.base_graphs, args.output), indent=2))
