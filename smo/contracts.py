"""Build an evidence-labelled graph from the released Chongyu contracts.

An xApp's target/affected KPI list is not parameter-level causal identification.
Its cross-product with writable RCPs creates candidate edges only. Reading an
input KPI does not create an influence edge. No simulator or learner is imported.
"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = 'artifacts/xapps/chongyu/20261117/release.json'
POOL = 'ric/near_rt/pools/chongyu'

PARAMETERS = {
    'ue_offload_fraction': ('Offload fraction', 'ratio', 'UE', 'continuous'),
    'ue_drx_configs': ('DRX configuration', 'configuration', 'UE', 'structured'),
    'scheduler_mode': ('Scheduler mode', 'category', 'Cell', 'categorical'),
    'ue_scheduler_weights': ('Scheduling weights', 'weight', 'UE', 'continuous'),
    'cell_sleep_mode': ('BS sleep policy', 'category', 'Cell', 'categorical'),
}

# Each canonical KPI has an explicit scope and aggregation, independent of the
# text aliases in older metadata. The UI always preserves per-entity readings.
KPIS = {
    'ue_power_w': ('UE power consumption', 'W', 'UE0 + UE4', 'sum of modeled radio + CPU energy counter deltas / elapsed seconds over the same observation window; total mean UE power'),
    'ue_energy_j': ('UE energy consumed', 'J', 'UE0 + UE4', 'sum; cumulative modeled radio + CPU energy'),
    'ap_power_w': ('BS power consumption', 'W', 'All cells', 'total modeled base-station energy counter delta / elapsed time; mean total power over the measurement window, not cumulative energy'),
    'network_goodput_mbps': ('Network Downlink throughput', 'Mbit/s', 'All UEs', 'successfully delivered user-data byte counter delta / elapsed time; goodput, excluding retransmitted or undelivered bytes'),
    'mec_offloaded_latency_ms': ('MEC task delay', 'ms', 'UE0 + UE4', 'completion-count-weighted mean; completed offloaded tasks'),
    'urllc_failure_events_per_s': ('URLLC failure event rate', 'events/s', 'UE2 + UE6', 'deadline failures plus disjoint overflows; counter delta / time; includes carryover; distinct from packet failure fraction'),
    'urllc_failure_fraction': ('URLLC packet error rate', 'ratio', 'UE2 + UE6', 'deadline failures plus disjoint overflows / offered packets over the stated measurement window; stored as a fraction and displayed as percent; includes carryover, not mature-cohort reliability probability'),
    'urllc_packet_latency_ms': ('URLLC packet delay', 'ms', 'UE2 + UE6', 'completion-count-weighted interval mean; completed packets only'),
    'mec_expected_gflops': ('Expected MEC compute load', 'GFLOP/s', 'UE4', 'configured task rate × task compute × executed offload fraction; offered compute demand, not server utilization or energy'),
    'task_latency_ms': ('Task E2E latency', 'ms', 'UE0 + UE4', 'completion-count-weighted mean; completed tasks'),
    'task_success_ratio': ('On-time task completion', 'ratio', 'UE0 + UE4', 'on-time completions / generated tasks; cumulative, pending included'),
    'per_ue_goodput_mbps': ('Per-UE Downlink throughput', 'Mbit/s', 'All UEs', 'one interval delivered-byte rate per UE'),
    'packet_latency_ms': ('Mean Downlink packet delay', 'ms', 'All UEs', 'completion-count-weighted interval mean; completed packets only'),
    'queue_bytes': ('Queued downlink data', 'byte', 'All UEs', 'sum of current pending bytes'),
    'hol_delay_ms': ('Oldest queued-packet delay', 'ms', 'All UEs', 'maximum reported per-UE HOL delay'),
    'queue_overflow_bytes': ('Queue-overflow data loss', 'byte', 'All UEs', 'sum of cumulative dropped bytes'),
    'ue_active_fraction': ('UE active-time ratio', 'ratio', 'UE0 + UE4', 'mean of cumulative per-UE active fractions'),
    'ap_available_fraction': ('BS service availability', 'ratio', 'All cells', 'mean of cumulative per-cell availability'),
    'mec_realized_gflops': ('Observed MEC compute load', 'GFLOP/s', 'UE4', 'generated offloaded task counter delta × configured task compute / time; offered compute demand, not server utilization or energy'),
}

# Display guidance for the released objectives, not a live trend or a guarantee
# that minimizing a metric serves every possible operator intent.
KPI_GOALS = {
    'ue_power_w': 'minimize', 'ap_power_w': 'minimize',
    'network_goodput_mbps': 'maximize', 'per_ue_goodput_mbps': 'maximize',
    'mec_offloaded_latency_ms': 'minimize', 'task_latency_ms': 'minimize',
    'urllc_failure_events_per_s': 'minimize', 'urllc_failure_fraction': 'minimize',
    'urllc_packet_latency_ms': 'minimize', 'packet_latency_ms': 'minimize',
    'task_success_ratio': 'maximize', 'queue_bytes': 'minimize',
    'hol_delay_ms': 'minimize', 'queue_overflow_bytes': 'minimize',
    'mec_expected_gflops': 'upper_bound', 'mec_realized_gflops': 'upper_bound',
}

ALIASES = {
    'ue_power_w': 'ue_power_w', 'ue_total_power_w': 'ue_power_w', 'UE power': 'ue_power_w',
    'ue_energy_total_j': 'ue_energy_j', 'UE_radio_energy': 'ue_energy_j', 'UE energy': 'ue_energy_j',
    'network_ap_power_w': 'ap_power_w',
    'network_dl_goodput_mbps': 'network_goodput_mbps', 'network_goodput_mbps': 'network_goodput_mbps',
    'throughput_mbps': 'per_ue_goodput_mbps', 'eMBB goodput': 'per_ue_goodput_mbps',
    'per_UE_goodput': 'per_ue_goodput_mbps', 'per_ue_goodput_mbps': 'per_ue_goodput_mbps',
    'per_ue_delivery': 'per_ue_goodput_mbps',
    'ue_task_metrics.mean_completion_latency_ms': 'task_latency_ms',
    'MEC_task_end_to_end_latency': 'task_latency_ms',
    'ue_task_metrics.offloaded_mean_completion_latency_ms': 'mec_offloaded_latency_ms',
    'ue_task_metrics.success_ratio': 'task_success_ratio',
    'UE2+UE6 unique deadline/overflow failure event rate': 'urllc_failure_events_per_s',
    'URLLC deadline failures': 'urllc_failure_events_per_s',
    'urllc_unique_deadline_failure_events_per_s': 'urllc_failure_events_per_s',
    'urllc_packet_deadline_failure_fraction': 'urllc_failure_fraction',
    'packet_deadline_failure_fraction': 'urllc_failure_fraction',
    'deadline_failure_fraction': 'urllc_failure_fraction',
    'urllc_packet_error_rate': 'urllc_failure_fraction',
    'URLLC packet error rate': 'urllc_failure_fraction',
    'completed-packet mean latency': 'urllc_packet_latency_ms',
    'packet_completion_latency_ms': 'packet_latency_ms', 'per-UE latency': 'packet_latency_ms',
    'packet_delay': 'packet_latency_ms', 'queue_bytes': 'queue_bytes', 'hol_delay_ms': 'hol_delay_ms',
    'queue_overflow': 'queue_overflow_bytes', 'ue_active_fraction': 'ue_active_fraction',
    'ap_available_fraction': 'ap_available_fraction',
    'expected_offloaded_compute_demand_gflops': 'mec_expected_gflops',
    'realized_offloaded_compute_demand_gflops': 'mec_realized_gflops',
}

APP_SPECS = {
    'x01': ('x01.json', 'Reduce UE power', {'ue_power_w': 'minimize'}),
    'x02': ('x02.json', 'Reduce BS energy', {'ap_power_w': 'minimize'}),
    'x08': ('x08.json', 'Increase Downlink throughput', {'network_goodput_mbps': 'maximize'}),
    'x12_mec': ('x12.json', 'Reduce MEC task delay', {'mec_offloaded_latency_ms': 'minimize'}),
    'x12_urllc': ('x12_urllc_qos.json', 'Reduce URLLC failures', {
        'urllc_failure_fraction': 'minimize', 'urllc_packet_latency_ms': 'minimize'}),
    'x13': ('mec_budget.json', 'Limit MEC compute load', {'mec_expected_gflops': 'upper_bound'}),
}


def load_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError('Non-finite JSON: '+s)))


def class_fields(path, name):
    """Read literal public contract attributes without importing GPU/ML code."""
    tree = ast.parse(Path(path).read_text())
    constants = {t.id: row.value for row in tree.body if isinstance(row, ast.Assign)
                 for t in row.targets if isinstance(t, ast.Name)}
    def literal(node):
        if isinstance(node, ast.Name):
            return literal(constants[node.id])
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return literal(node.left) + literal(node.right)
        return ast.literal_eval(node)
    cls = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == name)
    result = {}
    for row in cls.body:
        if isinstance(row, ast.Assign):
            for target in row.targets:
                if isinstance(target, ast.Name) and target.id == 'writable_actions':
                    result[target.id] = literal(row.value)
    return result


def build_graph(root=ROOT):
    root = Path(root)
    release = load_json(root/RELEASE)
    catalog = load_json(root/POOL/'catalog.json')
    nodes, edges, relations, sources = [], [], {}, set()
    target_ids = set()
    for key, (filename, label, targets) in APP_SPECS.items():
        released = release['models'][key]
        ident = released['implementation_id']
        metadata_path = f'{POOL}/{filename}'
        metadata = load_json(root/metadata_path)
        if metadata['implementation'] != ident:
            raise ValueError('Metadata implementation mismatch: '+key)
        module, cls = catalog[ident]['factory'].split(':')
        module_path = module.replace('.', '/')+'.py'
        public = class_fields(root/module_path, cls)
        # The QoS subclass is selected by the released controller factory, while
        # the common factory class declares its public writable RCPs.
        params = [p for p in PARAMETERS if p in metadata.get('rcp', {})]
        if set(params) != set(public['writable_actions']):
            raise ValueError('Metadata/public writable RCP drift: '+ident)
        target_ids.update(targets)
        nodes.append(dict(id='xapp:'+ident, kind='xapp', key=key, label=label,
            paper_id=released['id'], algorithm=released['algorithm'], implementation_id=ident,
            target_kpis=targets, target_ues=metadata.get('target_ues'),
            model_sha256=released['sha256'], evaluation_status=released['evaluation_status'],
            source=metadata_path, contract_source=module_path,
            runtime_parameters=deepcopy(released['runtime_parameters']),
            policy_evaluation=deepcopy(released.get('policy_evaluation')),
            drx_profiles=deepcopy(metadata.get('rcp',{}).get('ue_drx_configs',{}).get('profiles',[])) if key=='x01' else [], state=None))
        raw = metadata.get('affected_kpis', {})
        affected = set()
        for alias in raw:
            if alias in ALIASES:
                affected.add(ALIASES[alias])
        # Include non-objective contract KPI entries as side effects/guards.
        for alias in metadata.get('target_kpis', {}):
            if alias in ALIASES:
                affected.add(ALIASES[alias])
        affected.update(targets)
        # Existing contracts list the energy counter. Window-average power is
        # its physically derived monitoring metric; these remain candidate
        # influence edges until calibrated in W, never relabelled J estimates.
        if 'ue_energy_j' in affected:
            affected.add('ue_power_w')
        for p in params:
            detail = metadata['rcp'][p]
            locked = isinstance(detail, dict) and detail.get('deployed') == 'native_locked'
            edges.append(dict(id='control:'+ident+':'+p, source='xapp:'+ident,
                target='rcp:'+p, kind='controls', evidence='contract', supporting_xapps=[ident],
                declared_values=deepcopy(detail), deployment='native_locked' if locked else 'controllable',
                source_files=[metadata_path, module_path], sensitivity=None))
            for k in sorted(affected):
                edge = relations.setdefault((p, k), dict(id='influence:'+p+':'+k,
                    source='rcp:'+p, target='kpi:'+k, kind='influences', evidence='contract_candidate',
                    causal_status='unidentified', sign='unknown', supporting_xapps=[],
                    objectives={}, source_files=[], sensitivity=None,
                    application_scopes={},
                    note='Candidate from app-level target/affected KPI metadata; individual RCP attribution and sign need interventions.'))
                edge['supporting_xapps'].append(ident)
                edge['application_scopes'][ident] = metadata.get('target_ues', 'network; see app contract')
                if k in targets:
                    edge['objectives'][ident] = targets[k]
                if metadata_path not in edge['source_files']:
                    edge['source_files'].append(metadata_path)
        sources.update((metadata_path, module_path))
    for p, (label, unit, scope, dtype) in PARAMETERS.items():
        nodes.append(dict(id='rcp:'+p, kind='rcp', key=p, label=label,
            unit=unit, scope=scope, dtype=dtype, state=None))
    used = {k for _, k in relations}
    for k, (label, unit, scope, aggregation) in KPIS.items():
        if k in used:
            nodes.append(dict(id='kpi:'+k, kind='kpi', key=k, label=label, unit=unit,
                scope=scope, aggregation=aggregation, primary=k in target_ids,
                preferred_direction=KPI_GOALS.get(k), state=None))
            if k=='urllc_failure_fraction':
                # Canonical inputs/estimates remain ratios. Presentation alone
                # converts 0.07 to 7%; historical events/s is never relabelled.
                nodes[-1].update(display_unit='%',display_multiplier=100.,
                    improvement_unit='percentage_points')
    edges.extend(relations.values())
    sources.update((RELEASE, POOL+'/catalog.json'))
    signatures = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sorted(sources)}
    return dict(schema='oran.intent-graph.v1', pool='chongyu', nodes=nodes, edges=edges,
        contract_sha256=signatures, builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), notes=[
            'Contract candidates are not identified causal effects. Reading a KPI is not an influence edge.',
            'X-01 DRX is controllable in release 20261117; full profile effects are not identified by the native all-UE inactivity-timer fit.',
            'Scheduler mode and UE weights are joint action bundles; observational runs do not identify independent partial derivatives.'])


def validate_influences(graph, path):
    """Import audited intervention/regression results; never infer from logs.

    An estimate carries scope/context, horizon, sample count, units and evidence
    provenance. Categorical controls use a named contrast, not a derivative.
    Validation is atomic and the original graph is never mutated on failure.
    """
    data = load_json(path)
    if data.get('schema') != 'oran.intervention-edges.v1' or not isinstance(data.get('estimates'), list):
        raise ValueError('Expected oran.intervention-edges.v1 estimates')
    result = deepcopy(graph)
    result['influence_input'] = dict(name=Path(path).name, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    lookup = {(e['source'], e['target']): e for e in result['edges'] if e['kind'] == 'influences'}
    nodes = {n['id']: n for n in graph['nodes']}
    seen = set()
    import math
    for row in data['estimates']:
        pair = ('rcp:'+row['parameter'], 'kpi:'+row['kpi'])
        if 'contrast_id' in row and (not isinstance(row['contrast_id'],str)or not row['contrast_id']):
            raise ValueError('A contrast ID must be a nonempty string')
        identity=(pair,row.get('contrast_id',''))
        if identity in seen or pair[0] not in nodes or pair[1] not in nodes:
            raise ValueError('Duplicate/unknown influence estimate')
        seen.add(identity)
        required = {'parameter', 'kpi', 'estimate', 'unit', 'kind', 'sample_count', 'horizon_ms',
                    'scope', 'context', 'method', 'data_sha256', 'design'}
        if not required <= row.keys() or row['design'] != 'controlled_perturbation':
            raise ValueError('Intervention provenance is required')
        vector_estimate=row['estimate'] is None and isinstance(row.get('entity_estimates'),dict) and bool(row['entity_estimates'])
        if vector_estimate:
            if any(type(v)not in(int,float)or not math.isfinite(v)for v in row['entity_estimates'].values()):
                raise ValueError('Finite entity-specific estimates are required')
        for key in (('horizon_ms',) if vector_estimate else ('estimate','horizon_ms')):
            if type(row[key]) not in (int, float) or not math.isfinite(row[key]):
                raise ValueError('Finite estimate/horizon required')
        if row['horizon_ms'] < 0 or type(row['sample_count']) is not int or row['sample_count'] < 2:
            raise ValueError('Positive sample count / nonnegative horizon required')
        if not all(isinstance(row[k], str) and row[k].strip() for k in ('unit', 'scope', 'context', 'method')):
            raise ValueError('Units, scope, context and method are required')
        if len(row['data_sha256']) != 64 or any(c not in '0123456789abcdef' for c in row['data_sha256']):
            raise ValueError('Expected training-data SHA-256')
        dtype=nodes[pair[0]]['dtype']
        scalar_timer_fit=(dtype=='structured'and row['kind']in('range_linear_fit','local_derivative')and row.get('scalar_component')in('inactivity_ms','cycle_ms'))
        categorical = dtype in ('categorical', 'structured') and not scalar_timer_fit
        if row['kind'] not in ('local_derivative', 'finite_contrast','range_linear_fit') or categorical and row['kind'] != 'finite_contrast':
            raise ValueError('Discrete/structured controls require a finite contrast')
        if row['kind']=='range_linear_fit':
            levels=row.get('fit_levels',[]);domain=row.get('fit_domain',[]);diagnostics=row.get('fit_diagnostics',{})
            if (not isinstance(levels,list)or len(levels)<4 or row.get('fit_point_count')!=len(levels)
                or any(type(x)not in(int,float)or not math.isfinite(x)for x in levels)or len(set(levels))!=len(levels)
                or domain!=[min(levels),max(levels)]or row.get('grid_point_count',0)<len(levels)):
                raise ValueError('A range fit requires at least four finite sampled levels and its valid domain')
            if not isinstance(row.get('grid_points'),list)or not diagnostics or diagnostics.get('assessment')not in('constant_response_observed','limited_first_order_approximation','first_order_approximation_supported'):
                raise ValueError('Range-fit observations and residual diagnostics are required')
        if row['kind'] == 'finite_contrast' and (not {'from', 'to'} <= row.keys() or row['from'] == row['to']):
            raise ValueError('Finite contrast requires distinct from/to values')
        if 'confidence_interval' in row and row['confidence_interval'] is not None:
            if vector_estimate:
                raise ValueError('Vector effects require per-entity uncertainty')
            ci = row['confidence_interval']
            if not isinstance(ci, list) or len(ci) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in ci) or not ci[0] <= row['estimate'] <= ci[1]:
                raise ValueError('Invalid confidence interval')
        for uid,details in row.get('entity_uncertainty',{}).items():
            if not vector_estimate or uid not in row['entity_estimates'] or details.get('estimate')!=row['entity_estimates'][uid]:
                raise ValueError('Entity uncertainty does not match its point estimate')
            ci=details.get('confidence_interval')
            if ci is not None and (not isinstance(ci,list)or len(ci)!=2 or any(type(v)not in(int,float)or not math.isfinite(v)for v in ci)or not ci[0]<=row['entity_estimates'][uid]<=ci[1]):
                raise ValueError('Invalid entity confidence interval')
        if pair not in lookup:
            edge = dict(id='influence:'+row['parameter']+':'+row['kpi'], source=pair[0], target=pair[1],
                kind='influences', supporting_xapps=[], objectives={}, source_files=[str(Path(path).name)])
            lookup[pair] = edge
            result['edges'].append(edge)
        edge=lookup[pair]
        edge.setdefault('sensitivities',[]).append(deepcopy(row))
        # Keep the first context as the backwards-compatible default. Other
        # tested scopes/contrasts remain separately selectable, never averaged.
        if not edge.get('sensitivity'):
            values=list(row['entity_estimates'].values())if vector_estimate else[row['estimate']]
            signs={1 if v>0 else -1 if v<0 else 0 for v in values}
            sign='mixed'if len(signs)>1 else'positive'if 1 in signs else'negative'if -1 in signs else'zero'
            edge.update(evidence='controlled_perturbation',causal_status='range_linear_approximation'if row['kind']=='range_linear_fit'else'local_estimate',sign=sign,
                sensitivity=deepcopy(row),note='Range-wide first-order approximation; inspect tested domain and residuals before using it as a local control model.'if row['kind']=='range_linear_fit'else'Local estimate conditional on the recorded scope, context and horizon; not a global causal law.')
        if Path(path).name not in lookup[pair]['source_files']:
            lookup[pair]['source_files'].append(Path(path).name)
    return result
