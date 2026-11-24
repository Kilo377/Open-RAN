"""Audited fixed-RCP population contrasts, independent of deployed combinations."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

SCHEMA = 'oran.action-gain-supplement.v1'
MEMBERS = ['ue_2', 'ue_6', 'ue_8']
UNITS = {'mec_offloaded_latency_ms': 'ms', 'ue_power_w': 'W'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError('Action gain supplement: ' + message)


def quantile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    left = math.floor(position)
    right = math.ceil(position)
    return values[left] + (position - left) * (values[right] - values[left])


def fit(observations):
    """Equal seed weights; repeated start times stay inside their seed block."""
    require(bool(observations), 'no fixed-parameter observations')
    keys = set()
    grouped = {k: defaultdict(list) for k in UNITS}
    for row in observations:
        require(row.get('input_role') == 'fixed_parameter_probe'
                and row.get('arm') == 'FIXED' and row.get('split') == 'training'
                and row.get('learned_policy_controls_used') is False
                and row.get('combination_evaluation_data_used') is False,
                'deployment or learned-policy outcomes are not fitting inputs')
        key = (row['worker'], row['seed'], row['origin_slot'])
        require(key not in keys, 'duplicate paired context')
        keys.add(key)
        require(row.get('members') == MEMBERS and row.get('from_weight') == 1.
                and row.get('to_weight') == .5 and row.get('mec_weight') == 2.
                and row.get('scheduler_mode') == 'latency_aware', 'different control contrast')
        require(row.get('prefix_equal') is True and row.get('task_identities_equal') is True
                and row.get('normal_controls_verified') is True
                and row.get('only_competitor_weights_changed') is True
                and row.get('network_configuration_verified') is True,
                'unverified paired fixed-RCP experiment')
        require(type(row['seed']) is int, 'invalid seed')
        for kpi in UNITS:
            before, after = row['before'][kpi], row['after'][kpi]
            require(all(type(x) in (int, float) and math.isfinite(x) for x in (before, after)),
                    'missing or censored measured KPI')
            grouped[kpi][row['seed']].append((after - before) / (row['to_weight'] - row['from_weight']))
    seeds = sorted(grouped['mec_offloaded_latency_ms'])
    require(len(seeds) >= 3, 'at least three independent seed blocks are required')
    rows = []
    for kpi, unit in UNITS.items():
        means = [math.fsum(grouped[kpi][s]) / len(grouped[kpi][s]) for s in seeds]
        estimate = math.fsum(means) / len(means)
        if len(seeds) <= 6:
            resamples = [math.fsum(items) / len(items)
                         for items in itertools.product(means, repeat=len(means))]
            interval_method = 'exact_seed_block_bootstrap'
        else:
            rng = random.Random(20261124)
            resamples = [math.fsum(rng.choices(means, k=len(means))) / len(means) for _ in range(10000)]
            interval_method = 'deterministic_seed_block_bootstrap'
        rows.append(dict(parameter='ue_scheduler_weights', kpi=kpi, kind='local_derivative',
            label='MEC-competing traffic priority · Latency-aware',
            contrast_id='supplement:competitor_priority:' + kpi, parameter_component='mec_competitor_priority',
            members=MEMBERS, estimate=estimate,
            confidence_interval=[quantile(resamples, .05), quantile(resamples, .95)], confidence_level=.9,
            unit=unit + ' / weight', x_unit='weight', fit_domain=[.5, 1.],
            conditioning=dict(scheduler_mode='latency_aware'), reference_scheduler_mode='latency_aware',
            reference_mec_weight=2., projection_role='residual_population_reference',
            population_control_eligible=True,
            aggregation='equal_seed_mean_of_paired_group_contrasts', independent_seeds=seeds,
            paired_contexts=len(observations), interval_method=interval_method,
            per_ue_causal_derivatives_identified=False, combination_evaluation_data_used=False))
    return rows


def apply(graph, entry, recipe):
    """Add an action context to existing P-K edges; never create topology."""
    item = entry['action_gain_supplement']
    path = Path(item['path'])
    require(sha(path) == item['sha256'], 'certificate checksum differs')
    certificate = json.loads(path.read_text())
    require(certificate.get('schema') == SCHEMA
            and certificate.get('environment_id') == recipe['id']
            and certificate.get('combination_evaluation_data_used') is False,
            'wrong environment or provenance')
    require(certificate.get('base_matrix_sha256') == sha(entry['influences'])
            and certificate.get('runtime_recipe_sha256') == sha(entry['runtime_recipe']),
            'supplement belongs to another graph or network runtime')
    data = certificate['measurements']
    relative = Path(data['path'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'unconfined observations path')
    data_path = path.parent / relative
    require(sha(data_path) == data['sha256'], 'fixed-probe observations changed')
    dataset = json.loads(data_path.read_text())
    require(dataset.get('combination_evaluation_data_used') is False, 'invalid measurement source')
    rows = fit(dataset['observations'])
    require(rows == certificate['estimates'], 'coefficients differ from the fixed-RCP measurements')
    result = deepcopy(graph)
    for row in rows:
        # Match the graph's dimensionless display convention. These scales
        # come from the fixed probes, never from a deployed combination.
        by_seed = defaultdict(list)
        for observation in dataset['observations']:
            by_seed[observation['seed']].append(observation['before'][row['kpi']])
        reference = math.fsum(math.fsum(values) / len(values)
                              for values in by_seed.values()) / len(by_seed)
        require(math.isfinite(reference) and abs(reference) > 0,
                'fixed-probe KPI reference cannot normalize the display')
        span = row['fit_domain'][1] - row['fit_domain'][0]
        row = dict(row, contrast_label=row['label'], reference_kpi_mean=reference,
                   importance=dict(signed_reference_importance=row['estimate'] * span / abs(reference),
                                   parameter_span=span, kpi_scale=abs(reference),
                                   normalization_basis='absolute_paired_reference_kpi'),
                   metric_scope='ue_population.v1', calibration_environment_id=recipe['id'])
        edge = next((e for e in result['edges'] if e['kind'] == 'influences'
                     and e['source'] == 'rcp:' + row['parameter'] and e['target'] == 'kpi:' + row['kpi']), None)
        require(edge is not None, 'a supplement cannot create a new selected relationship')
        require(not any(r.get('contrast_id') == row['contrast_id'] for r in edge.get('sensitivities', [])),
                'duplicate action calibration')
        edge['sensitivities'] = [*edge.get('sensitivities', []), row]
        selection = edge.get('statistical_selection')
        if selection is not None:
            require(selection.get('action_context_policy') == 'retain_valid_measured',
                    'current graph does not retain measured conditional contexts')
            selection['retained_contrasts'].append(row['contrast_id'])
            low, high = row['confidence_interval']
            classification = 'significant_contrasts' if low > 0 or high < 0 else 'uncertain_contrasts'
            selection.setdefault(classification, []).append(row['contrast_id'])
    from smo.influence_units import align_graph
    result = align_graph(result)
    info = dict(schema=SCHEMA, certificate_sha256=item['sha256'], independent_seeds=rows[0]['independent_seeds'],
                paired_contexts=rows[0]['paired_contexts'], added_action_contexts=len(rows),
                topology_changed=False, combination_evaluation_data_used=False,
                population_projection='measured_common_population_direction; residual members only')
    result.setdefault('action_gain_supplements', []).append(info)
    return result, info
