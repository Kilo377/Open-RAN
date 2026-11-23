"""Exact network-record parity for a CPU job-scheduling change; never launch."""
from itertools import zip_longest
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import compute_parity as exact
from smo.evaluation_reference import _model_pins, origin_configuration

APP_STREAMS = {
    'xapp:ue_energy_rf': 'energy_rf_events.jsonl',
    'xapp:ap_energy_rf': 'ap_rf_events.jsonl',
    'xapp:throughput_scheduler_mlp': 'throughput_mlp_events.jsonl',
    'xapp:latency_scheduler_rf': 'latency_rf_events.jsonl',
    'xapp:latency_urllc_rf': 'urllc_rf_events.jsonl',
    'xapp:mec_offload_budget_qgbdt': 'budget_qgbdt_events.jsonl',
}
CORE_STREAMS = {n+'.jsonl' for n in exact.JSONL}-set(APP_STREAMS.values())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def policy():
    return dict(schema='oran.parallel-network-parity-policy.v2',
        physical_source_sha256='cbe114446814007b275856e4bae92a2fe609abf4f4bcbbf8c32394a722a36af0',
        numeric_comparison='Exact recorded Decimal values; 1 and 1.0 agree; no tolerance or rounding.',
        jsonl_streams=list(exact.JSONL)+list(exact.OPTIONAL_JSONL),
        csv_streams=list(exact.CSV), unknown_network_streams='fail',
        required_core_jsonl=sorted(CORE_STREAMS),
        required_active_app_streams=APP_STREAMS,
        inactive_app_streams='Lazy producers: absence on both sides is allowed; any present stream must exist on both sides and compare completely.',
        arrays='Complete ordered sequences, including empty streams.',
        record_exclusions=[],
        configuration='Same origin_configuration normalization already used by native OriginGuard; source/model pins independently exact.',
        purpose='Prove CPU scheduling/cache isolation did not change the simulated network or autonomous controllers.',
        operational_files='Run timing, launch wrappers, cache locations, origin UUID/proof identity and rendered HTML are not compared as network records.',
        parity_is_not_an_independent_sample=True)


def inventory(run, app_ids=()):
    run = Path(run)
    result = {}
    known = {n+'.jsonl' for n in (*exact.JSONL, *exact.OPTIONAL_JSONL)} | {n+'.csv' for n in exact.CSV}
    for p in run.iterdir():
        name = p.name.removesuffix('.gz')
        if name.endswith(('.jsonl', '.csv')):
            if name not in known or name in result:
                raise ValueError('Unknown or duplicate physical stream: '+p.name)
            result[name] = p
    if any(a not in APP_STREAMS for a in app_ids):
        raise ValueError('Unknown active controller')
    mandatory = CORE_STREAMS | {n+'.csv' for n in exact.CSV} | {APP_STREAMS[a] for a in app_ids}
    if not mandatory <= result.keys():
        raise ValueError('Missing physical streams: '+str(sorted(mandatory-result.keys())))
    return result


def verify_seal(checkpoint):
    checkpoint = Path(checkpoint)
    seal = json.loads(checkpoint.read_text())
    if seal.get('completed') is not True or not seal.get('files_sha256'):
        raise ValueError('A complete nonempty checkpoint is required')
    job = checkpoint.parent.parent/'jobs'/seal['case']['id']
    for relative, pin in seal['files_sha256'].items():
        p = (job/relative).resolve()
        if not p.is_relative_to(job.resolve()) or not p.is_file() or sha(p) != pin:
            raise ValueError('Sealed file changed: '+relative)
    if seal.get('source_sha256') != policy()['physical_source_sha256']:
        raise ValueError('Unexpected physical source')
    return seal, job/'data/run'


def compare(reference_checkpoint, candidate_checkpoint):
    proof = dict(schema='oran.parallel-network-parity.v2', passed=False,
        policy=policy(), comparator_sha256=sha(__file__),
        reference_checkpoint=str(Path(reference_checkpoint).resolve()),
        candidate_checkpoint=str(Path(candidate_checkpoint).resolve()),
        differences=[], streams={})
    try:
        old, reference = verify_seal(reference_checkpoint)
        new, candidate = verify_seal(candidate_checkpoint)
        if reference.resolve() == candidate.resolve():
            raise ValueError('Independent physical runs are required')
        if old['worker_id'] != new['worker_id'] or old['case']['seed'] != new['case']['seed']:
            raise ValueError('Parity must remain on the same physical host and seed')
        if old['case']['app_ids'] != new['case']['app_ids']:
            raise ValueError('Autonomous controller selection differs')
        configs = [json.loads((p/'config_snapshot.json').read_text()) for p in (reference, candidate)]
        if origin_configuration(configs[0]) != origin_configuration(configs[1]):
            raise ValueError('Normalized physical configurations differ')
        models = [_model_pins(p) for p in (reference, candidate)]
        if len(models[0]) != 6 or models[0] != models[1] or old['frozen_models'] != new['frozen_models']:
            raise ValueError('All six frozen model identities must agree')
        for p, seal in ((reference, old), (candidate, new)):
            status = json.loads((p/'run_status.json').read_text())
            if (status.get('completed') is not True or not status.get('replay', {}).get('audits_passed')
                    or status.get('source_sha256') != policy()['physical_source_sha256']):
                raise ValueError('A complete audited physical run is required')
            if seal['case']['app_ids']:
                guard = json.loads((p/'evaluation_origin_verified.json').read_text())
                if guard.get('verified_before_activation') is not True or guard.get('origin_id') != seal['origin_id']:
                    raise ValueError('The native prefix was not independently verified')
        inventories = [inventory(p, old['case']['app_ids']) for p in (reference, candidate)]
        if inventories[0].keys() != inventories[1].keys():
            raise ValueError('Physical stream inventories differ')
        total_slots = configs[0]['manifest']['total_slots']
        sentinel = object()
        total_differences = 0
        for name in sorted(inventories[0]):
            paths = [d[name] for d in inventories]
            hashes = [hashlib.sha256(), hashlib.sha256()]
            counts = [0, 0]
            different = 0
            for index, pair in enumerate(zip_longest(*(exact._records(name, p) for p in paths), fillvalue=sentinel)):
                for side, value in enumerate(pair):
                    if value is sentinel:
                        continue
                    if name == 'rt_sys_trace.jsonl' and value.get('slot_id') != index+1:
                        raise ValueError('Every absolute slot must be present in order')
                    hashes[side].update(exact._encoded(value)+b'\n')
                    counts[side] += 1
                issues = [('/', 'record_count_difference')] if any(v is sentinel for v in pair) else exact._differences(*pair)
                for pointer, reason in issues:
                    different += 1
                    total_differences += 1
                    if len(proof['differences']) < 100:
                        proof['differences'].append(dict(stream=name, record_index=index, pointer=pointer, reason=reason))
            if name == 'rt_sys_trace.jsonl' and counts != [total_slots, total_slots]:
                raise ValueError('Full configured trajectory is required')
            proof['streams'][name] = dict(exact_equal=different == 0, differences=different,
                reference=dict(records=counts[0], raw_sha256=sha(paths[0]), semantic_sha256=hashes[0].hexdigest()),
                candidate=dict(records=counts[1], raw_sha256=sha(paths[1]), semantic_sha256=hashes[1].hexdigest()))
        proof.update(passed=total_differences == 0, exact_difference_count=total_differences,
            source_equal=True, models_equal=True, configuration_equal=True, host_equal=True,
            reference_checkpoint_sha256=sha(reference_checkpoint), candidate_checkpoint_sha256=sha(candidate_checkpoint))
    except (OSError, ValueError, KeyError, TypeError) as error:
        proof['differences'].append(dict(reason='fail_closed_input_or_record_error', error=str(error)))
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = compare(a.reference, a.candidate)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise ValueError('Retain a prospective parity proof instead of overwriting it')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output), passed=result['passed'], differences=result.get('exact_difference_count'))))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
