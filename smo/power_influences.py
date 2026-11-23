"""Derive mean UE-power effects from audited fixed-window energy effects.

The retained energy study measured a *counter difference* over a scoring window.
Its affine fits and paired contrasts can be scaled linearly by 1/window_seconds.
This module refuses cumulative terminal energy or an unverified window. Source
measurements, model files and original coefficients are never modified.
"""
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tarfile


_ENERGY_DELTA = "sum(last['ue_services'][uid]['ue_energy_total_j']-first['ue_services'][uid]['ue_energy_total_j'] for uid in energy_ids)"
_ENERGY_TARGET = "values['ue_energy_j']"
_SCALE_FIELDS = ('estimate', 'standard_error', 'reference_kpi_mean',
                 'intercept_at_reference', 'center_linearity_residual')
_SCALE_INTERVALS = ('confidence_interval', 'replicate_range')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def _scaled(value, factor):
    if value is None:
        return None
    if isinstance(value, dict):
        return {key:_scaled(number, factor) for key, number in value.items()}
    if isinstance(value, (tuple, list)):
        return [_scaled(number, factor) for number in value]
    if not _numeric(value):
        raise ValueError('An energy statistic is not finite numeric data.')
    return value*factor


def _verify_delta_source(source):
    """Recognize the archived estimator's exact first/last counter definition."""
    tree = ast.parse(source)
    expected = ast.dump(ast.parse(_ENERGY_DELTA, mode='eval').body)
    target = ast.dump(ast.parse(_ENERGY_TARGET+'=None').body[0].targets[0])
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and any(ast.dump(t)==target for t in node.targets)]
    if len(assignments)!=1 or ast.dump(assignments[0].value)!=expected:
        raise ValueError('Archived analysis does not prove scoring-window energy counter differences.')
    for name, key in (('start','scoring_start_ms'),('end','scoring_end_ms')):
        expected_value=ast.dump(ast.parse("round(plan['"+key+"']*2)",mode='eval').body)
        if not any(isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
                   and ast.dump(node.value)==expected_value for node in ast.walk(tree)):
            raise ValueError('Archived scoring time-to-slot conversion is not established.')
    # First and last must be snapshots at the scoring boundaries, not terminal
    # cumulative counters or a window starting at a substituted zero.
    for name, boundary in (('first','start'), ('last','end')):
        found = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.If) or ast.dump(node.test)!=ast.dump(ast.parse('rows == '+boundary, mode='eval').body):
                continue
            for statement in node.body:
                if isinstance(statement, ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in statement.targets):
                    found |= ast.dump(statement.value)==ast.dump(ast.parse('deepcopy(row)', mode='eval').body)
        if not found:
            raise ValueError('Archived scoring-boundary snapshots are not established.')


def _source_proof(folder, rows):
    dataset_bytes = (folder/'measurement-dataset.json').read_bytes()
    dataset = json.loads(dataset_bytes)
    if dataset.get('metric_scope')!='ue_population.v1' or not dataset.get('cases'):
        raise ValueError('Power derivation requires an audited population measurement dataset.')
    digest = _sha(dataset_bytes)
    if any(row.get('data_sha256')!=digest for row in rows):
        raise ValueError('Energy coefficients do not match the retained measurement dataset.')
    manifest = json.loads((folder/'analysis-source-files.json').read_text())
    archive_bytes = (folder/'analysis-source.tgz').read_bytes()
    with tarfile.open(folder/'analysis-source.tgz') as archive:
        stream = archive.extractfile('smo/identify.py')
        if stream is None:
            raise ValueError('Archived energy analysis source is missing.')
        source_bytes = stream.read()
    if _sha(source_bytes)!=manifest.get('smo/identify.py'):
        raise ValueError('Archived energy analysis source fingerprint differs.')
    _verify_delta_source(source_bytes.decode())
    durations = set()
    for row in rows:
        start,end,window = (row.get(k) for k in ('scoring_start_ms','scoring_end_ms','scoring_window_ms'))
        if not all(map(_numeric,(start,end,window))) or end<=start or window!=end-start:
            raise ValueError('A consistent positive scoring window is required for power derivation.')
        durations.add((start,end,window))
    # One fixed scored interval applies to the audited experiment's cases.
    if len(durations)!=1:
        raise ValueError('A shared scoring window has not been established across this dataset.')
    start,end,window = next(iter(durations))
    if any(case.get('proof',{}).get('verified_held_parameter_slots')!=round(window*2)
           for case in dataset['cases']):
        raise ValueError('Retained cases do not prove the declared scored duration.')
    return dict(measurement_definition='scoring_window_energy_counter_delta',
        scoring_start_ms=start,scoring_end_ms=end,scoring_window_ms=window,
        duration_s=window/1000, measurement_dataset_sha256=digest,
        archived_analysis_source_sha256=_sha(source_bytes),
        analysis_source_archive_sha256=_sha(archive_bytes))


def derive_power_estimate(row, proof, *, source_estimates_sha256):
    """Scale energy-valued statistics, leaving x coordinates and ratios intact."""
    if row.get('kpi')!='ue_energy_j' or proof.get('measurement_definition')!='scoring_window_energy_counter_delta':
        raise ValueError('Only proven scoring-window UE energy differences can become power effects.')
    duration = proof.get('duration_s')
    if not _numeric(duration) or duration<=0 or row.get('scoring_window_ms')!=duration*1000:
        raise ValueError('Power requires the energy measurement duration, not run length or response horizon.')
    if row.get('data_sha256')!=proof.get('measurement_dataset_sha256'):
        raise ValueError('Power derivation proof and effect dataset differ.')
    unit = row.get('unit','')
    if not unit.startswith('J / '):
        raise ValueError('Expected a joule-valued intervention coefficient.')
    factor=1/duration
    result=deepcopy(row)
    result.update(kpi='ue_power_w',unit='W'+unit[1:])
    for key in _SCALE_FIELDS+_SCALE_INTERVALS:
        if key in result:
            result[key]=_scaled(result[key],factor)
    for key,value in result.get('entity_estimates',{}).items():
        result['entity_estimates'][key]=_scaled(value,factor)
    for entity in result.get('entity_uncertainty',{}).values():
        for key in _SCALE_FIELDS+_SCALE_INTERVALS:
            if key in entity:
                entity[key]=_scaled(entity[key],factor)
    for fit in result.get('seed_fits',[]):
        for key in ('beta','intercept','mean'):
            if key in fit:
                fit[key]=_scaled(fit[key],factor)
    for point in result.get('grid_points',[]):
        point['mean']=_scaled(point['mean'],factor)
        for observation in point.get('by_seed',[]):
            observation['value']=_scaled(observation['value'],factor)
    for pair in result.get('seed_pairs',[]):
        for key in ('low','high','reference'):
            if key in pair:
                pair[key]=_scaled(pair[key],factor)
    for key in ('rmse','leave_level_out_rmse','max_abs_residual'):
        if key in result.get('fit_diagnostics',{}):
            result['fit_diagnostics'][key]=_scaled(result['fit_diagnostics'][key],factor)
    result['derived_from']=dict(source_kpi='ue_energy_j',target_kpi='ue_power_w',
        source_estimates_sha256=source_estimates_sha256,
        source_estimate_sha256=_sha(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()),
        formula='mean_power_W = scored_energy_counter_delta_J / scoring_window_seconds',
        normalization_factor_per_s=factor,**deepcopy(proof))
    result['interpretation']=row.get('interpretation','')+' Derived average-power response over the same fixed scoring window; not an instantaneous or cumulative-power measurement.'
    return result


def derive_power_influences(source_path, output_path=None):
    """Append audited UE-power effects without modifying original data.

    Invoke separately for influences.json and all-effects.json into a new
    directory when both are needed by GraphService's matrix validation. Their
    row-level source is the same canonical original all-effects matrix, so an
    identical published estimate derives an exactly identical power estimate.
    Missing or incompatible measurement proof fails rather than inventing W.
    """
    source_path=Path(source_path)
    source_bytes=source_path.read_bytes()
    data=json.loads(source_bytes)
    if data.get('schema')!='oran.intervention-edges.v1' or data.get('metric_scope')!='ue_population.v1':
        raise ValueError('Expected population intervention estimates.')
    if any(row.get('kpi')=='ue_power_w' for row in data.get('estimates',[])):
        raise ValueError('Source already contains UE-power coefficients.')
    energy=[row for row in data['estimates'] if row.get('kpi')=='ue_energy_j']
    if not energy:
        raise ValueError('Source has no UE energy effects to derive.')
    proof=_source_proof(source_path.parent,energy)
    digest=_sha(source_bytes)
    canonical_path=source_path.parent/'all-effects.json'
    canonical_digest=digest
    if canonical_path.exists():
        canonical_bytes=canonical_path.read_bytes()
        canonical=json.loads(canonical_bytes)
        canonical_digest=_sha(canonical_bytes)
        identity=lambda row:(row.get('parameter'),row.get('kpi'),row.get('contrast_id'))
        rows=canonical.get('estimates',[])
        indexed={identity(row):row for row in rows}
        if len(indexed)!=len(rows) or any(indexed.get(identity(row))!=row for row in energy):
            raise ValueError('Published energy effects differ from the canonical all-effects matrix.')
    result=deepcopy(data)
    result['estimates'] += [derive_power_estimate(row,proof,source_estimates_sha256=canonical_digest) for row in energy]
    result['power_derivation']=dict(source_file=source_path.name,source_sha256=digest,
        canonical_matrix_file=canonical_path.name if canonical_path.exists() else source_path.name,
        canonical_matrix_sha256=canonical_digest,derived_count=len(energy),**proof)
    if output_path is not None:
        output_path=Path(output_path)
        if output_path.resolve()==source_path.resolve():
            raise ValueError('Power derivation must not overwrite original coefficients.')
        output_path.parent.mkdir(parents=True,exist_ok=True)
        output_path.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    return result
