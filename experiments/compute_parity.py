"""Prospectively fixed, fail-closed exact comparison for compute-only replay.

This module never launches, stops or migrates a simulation.  It compares the
complete ordered physical records, not just an Evaluation summary.  Decimal
parsing preserves the numbers actually recorded: there is no tolerance,
rounding, field-name heuristic, sorting of event arrays, or KPI selection.

The small file/path whitelist below is part of the protocol.  Unknown fields
remain compared and unknown files fail admission.  Nonphysical exclusions are
reported by exact path, count, and SHA256 of their original ordered values.
Changing the whitelist after observing parity failures requires a new protocol;
it must never be used to rescue an already failed prospective parity check.
"""
from __future__ import annotations

from collections import Counter
import csv
from decimal import Decimal
import gzip
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re
import sys
from typing import Any
import zipfile


SCHEMA = 'oran.compute-only-exact-parity.v1'
POLICY = 'source_v2_complete_physical_exact_v1'
JSONL = (
    'rt_sys_trace', 'packet_events', 'harq_events', 'olla_decisions',
    'olla_feedback', 'control_receipts', 'diagnostic_trace', 'rule_decisions',
    'state_snapshots', 'xapp_selection', 'xapp_decision_previews',
    'xapp_test_events', 'ap_rf_events', 'energy_rf_events',
    'throughput_mlp_events',
)
CSV = ('cell_metrics', 'ue_metrics', 'slot_metrics')
JSON = (
    'config_snapshot', 'initial_conditions', 'connection_events',
    'orchestration_epoch_history', 'evaluation_summary', 'final_summary',
    'steady_state', 'xapp_switch_timeline', 'xapp_test_spec', 'xapp_hot_start',
    'evaluation_origin_verified', 'performance', 'runtime_diagnostics',
    'run_status',
)
MODEL_IDS = (
    'ue_energy_rf', 'ap_energy_rf', 'throughput_scheduler_mlp',
    'latency_scheduler_rf', 'latency_urllc_rf', 'mec_offload_budget_qgbdt',
)
OPTIONAL_JSONL = ('latency_rf_events', 'urllc_rf_events', 'budget_qgbdt_events')
OPTIONAL_JSON = ('live', 'replay_export', 'public_kpi_inputs')
OPTIONAL_NPZ = ('live.mesh.npz', 'replay.mesh.npz', 'scene.mesh.npz')
SUMMARY_APPS = ('ue_energy_xapp', 'ap_xapp', 'throughput_xapp', 'latency_xapp',
                'urllc_xapp', 'budget_xapp')
MAX_DIFFERENCES = 100
_MISSING = object()
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_NUM = re.compile(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z')


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _number(value: Any) -> bool:
    return type(value) is int or isinstance(value, Decimal)


def _canonical(value: Any) -> Any:
    """Typed canonical numbers; 1 and 1.0 agree, true and 1 never agree."""
    if _number(value):
        d = Decimal(value)
        if not d.is_finite():
            raise ValueError('Nonfinite recorded number')
        # Do not Decimal.normalize(): it rounds through the ambient context.
        sign, digits, exponent = d.as_tuple()
        digits = list(digits)
        if not any(digits):
            return ['number', '0']
        while len(digits) > 1 and digits[-1] == 0:
            digits.pop(); exponent += 1
        return ['number', ('-' if sign else '') + ''.join(map(str, digits)) + 'e' + str(exponent)]
    if isinstance(value, dict):
        return ['object', [[k, _canonical(v)] for k, v in sorted(value.items())]]
    if isinstance(value, list):
        return ['array', [_canonical(v) for v in value]]
    if value is None or type(value) in (bool, str):
        return [type(value).__name__, value]
    raise ValueError('Unsupported recorded value type: ' + type(value).__name__)


def _encoded(value: Any) -> bytes:
    return json.dumps(_canonical(value), separators=(',', ':'), ensure_ascii=True).encode()


def _load(text: str) -> Any:
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d:
                raise ValueError('Duplicate JSON key: ' + k)
            d[k] = v
        return d

    def invalid(value):
        raise ValueError('Nonfinite JSON number: ' + value)

    return json.loads(text, parse_float=Decimal, parse_int=int,
                      parse_constant=invalid, object_pairs_hook=pairs)


def _read(path: Path) -> Any:
    return _load(path.read_text())


def _pointer(path) -> str:
    return '/' + '/'.join(str(x).replace('~', '~0').replace('/', '~1') for x in path)


def _matches(pattern: tuple, path: tuple) -> bool:
    return len(pattern) == len(path) and all(a == '*' or a == b for a, b in zip(pattern, path))


def _rule(file: str, path: str, kind: str, reason: str) -> dict:
    return dict(file=file, pointer=path, type=kind, reason=reason)


def _rules() -> list[dict]:
    r = []
    add = lambda f, p, t, why: r.append(_rule(f, p, t, why))
    wall = 'Host elapsed time or throughput of computation; never virtual time or network goodput'
    owned = 'Owned output/interpreter location; referenced content is separately compared or pinned'
    for k in ('initialization_s', 'simulation_s', 'output_and_telemetry_s',
              'steady_wall_s', 'steady_slots_per_s', 'steady_actual_tbs_per_s',
              'replay_export_s'):
        add('performance.json', '/' + k, 'number', wall)
    add('performance.json', '/stages/*/wall_s', 'number', wall)
    cache = 'Cache/prefetch producer scheduling metadata; channel identity, epochs and anchor ages stay compared'
    for k, kind in (
        ('completed', 'bool'), ('cancelled', 'bool'), ('generated_blocks', 'number'),
        ('skipped_blocks', 'number'), ('lock_wait_s', 'number'),
        ('worker_wall_s', 'number'), ('worker_wall_sum_s', 'number'),
        ('foreground_generated_blocks', 'number'), ('foreground_lock_wait_s', 'number'),
        ('total_generated_blocks', 'number'), ('worker_statuses', 'worker_status_list'),
    ):
        add('performance.json', '/rt_prefetch_statistics/' + k, kind, cache)
    for k in ('elapsed_s', 'simulation_wall_s_so_far', 'peak_rss_native',
              'peak_rss_bytes', 'peak_cuda_bytes', 'cupy_pool_reserved_bytes', 'output_bytes'):
        add('run_status.json', '/' + k, 'number', 'Host runtime/resource/output-size measurement, not a network counter')
    for f in ('run_status.json', 'replay_export.json'):
        prefix = '/replay' if f == 'run_status.json' else ''
        for k, kind in (('html', 'str'), ('html_bytes', 'number'), ('html_sha256', 'str')):
            add(f, prefix + '/' + k, kind, 'Rendered container identity; HTML template and embedded physical data compared independently')
    for k in ('cache_hits', 'cache_misses'):
        add('final_summary.json', '/rt_sys/' + k, 'number', cache)
    for app in SUMMARY_APPS:
        for k in ('fit_s', 'predict_s', 'decision_s'):
            add('final_summary.json', '/' + app + '/' + k, 'number', wall)
        add('final_summary.json', '/' + app + '/model_path', 'model_path', 'Model bytes and app-to-SHA binding verified using all six run checkpoints')
    add('final_summary.json', '/rule_based_v1/*/decision_seconds', 'number', wall)
    nonce = 'LiveXAppSession UUID4 communication nonce; not RNG/feature/arbiter/lease/request identity'
    add('xapp_hot_start.json', '/timeline/*/run_id', 'nonce', nonce)
    for f in ('final_summary.json', 'live.json'):
        add(f, '/xapp_test/run_id', 'nonce', nonce)
        for k in ('command_dir', 'comparison_spec', 'simulation_python'):
            add(f, '/xapp_test/' + k, 'owned_path', owned)
        add(f, '/xapp_test/model_artifacts/*/path', 'model_path', 'App-to-model SHA and copied checkpoint verified')
        for k in ('model_path', 'log_path'):
            add(f, '/xapp_test/app/parameters/' + k, k, 'Model checkpoint or output log content compared independently')
    for k in ('model_path', 'log_path'):
        add('config_snapshot.json', '/configs/ric/xapps/*/parameters/' + k, k, 'Pinned model content or compared owned event log')
        add('final_summary.json', '/xapp_library/instances/*/parameters/' + k, k, 'Pinned model content or compared owned event log')
    add('xapp_hot_start.json', '/models/*/path', 'model_path', 'App-to-model SHA and copied checkpoint verified')
    add('xapp_hot_start.json', '/timeline_source/path', 'timeline_path', 'Timeline bytes independently compared and SHA verified')
    for name in ('ap_services_profile', 'ue_services_profile'):
        add('config_snapshot.json', '/' + name + '/path', 'owned_path', 'Profile materialized values and identity/SHA remain compared')
    for k in ('record_compression',):
        add('config_snapshot.json', '/' + k, 'str', 'Recording compression only; decoded complete rows compared')
        add('run_status.json', '/' + k, 'str', 'Recording compression only; decoded complete rows compared')
        add('xapp_test_spec.json', '/arguments/' + k, 'str', 'Recording compression only; decoded complete rows compared')
    for k in ('config', 'channel_cache_root'):
        add('xapp_test_spec.json', '/arguments/' + k, 'owned_path', 'Materialized complete configuration/cache identity compared independently')
    add('xapp_test_spec.json', '/arguments/xapp_model_dir', 'model_dir', 'Complete six-model identity and content pinned independently')
    add('xapp_test_spec.json', '/arguments/xapp_toggle_timeline', 'timeline_path', 'Timeline bytes independently compared and SHA verified')
    for k in ('energy', 'ap', 'throughput', 'latency', 'urllc', 'budget'):
        add('xapp_test_spec.json', '/arguments/' + k + '_xapp_model', 'model_path', 'Frozen model content pinned independently')
    for model in ('x01', 'x02', 'x08', 'x12_mec', 'x12_urllc', 'x13'):
        add('preview_models/runtime-arguments.json', '/' + model + '/3', 'preview_model_path',
            'Installer argv model location only; model identity/content and all other argv tokens stay exact')
    for k in ('updated_unix_s', 'wall_s'):
        add('live.json', '/' + k, 'number', wall)
    add('live.json', '/mesh_path', 'mesh_path', 'Uncompressed NPZ arrays compared independently')
    add('network-replay.html', '/meta/wall_s', 'number_or_null', wall)
    add('network-replay.html', '/meta/source_run', 'str', 'Owned output directory name only')
    return r


RULES = _rules()
_PATTERNS = [(r, tuple(r['pointer'].strip('/').split('/'))) for r in RULES]


def comparison_spec() -> dict:
    """Serializable whitelist to seal BEFORE either prospective parity run."""
    return dict(schema=SCHEMA, policy=POLICY, number_equality='exact_recorded_decimal_no_tolerance_no_rounding',
        array_equality='same_length_same_order', unknown_fields='compare', unknown_files='fail',
        required_jsonl=list(JSONL), required_csv=list(CSV), required_json=list(JSON),
        required_model_checkpoints=list(MODEL_IDS), optional_jsonl=list(OPTIONAL_JSONL),
        optional_json=list(OPTIONAL_JSON), optional_npz=list(OPTIONAL_NPZ),
        optional_html=['network-replay.html'], metadata_rules=RULES,
        exclusions_preserve_field_presence=True, rendered_html='compare_template_and_embedded_physical_JSON',
        model_artifacts='all_six_checkpoint_bytes_and_bindings_exact_plus_preview_release_manifest',
        transport='gzip_suffix_and_archive_timestamps_not_physical; decoded records/NPY bytes exact',
        nonce_evidence=dict(generator='experiments/live_xapp_test.py:82 UUID4',
            uses='experiments/hot_xapps.py:160,234 session-equality guard only; not seed, policy feature or arbiter order',
            third_copy='experiments/rt_live.py:129 copies session.snapshot into live.xapp_test'),
        no_automatic_action=True, parity_is_not_an_independent_sample=True)


class _Audit:
    def __init__(self, root: Path):
        self.root = root
        self.counts = Counter()
        self.hashes = {}
        self.model_paths = {}
        self.model_pins = {}

    def note(self, file, path, value):
        name = file + ':' + _pointer(path)
        self.counts[name] += 1
        self.hashes.setdefault(name, hashlib.sha256()).update(_encoded(value) + b'\n')

    def transform(self, file, value, path=()):
        for rule, pattern in _PATTERNS:
            if rule['file'] == file and _matches(pattern, path):
                kind = rule['type']
                if kind == 'number' and not _number(value):
                    raise ValueError('Wrong excluded numeric type: ' + file + ':' + _pointer(path))
                if kind == 'number_or_null' and value is not None and not _number(value):
                    raise ValueError('Wrong excluded nullable number')
                if kind == 'bool' and type(value) is not bool:
                    raise ValueError('Wrong excluded boolean type')
                if kind == 'str' and type(value) is not str:
                    raise ValueError('Wrong excluded string type')
                if kind in ('str', 'nonce', 'model_path', 'log_path', 'owned_path', 'model_dir', 'timeline_path', 'mesh_path', 'preview_model_path'):
                    if value is not None and type(value) is not str:
                        raise ValueError('Wrong excluded string type')
                replacement = kind
                if kind == 'nonce' and (type(value) is not str or not re.fullmatch(r'[0-9a-f]{32}', value)):
                    raise ValueError('Invalid live communication nonce')
                if kind == 'model_path' and value is not None:
                    if value not in self.model_paths:
                        raise ValueError('Unpinned model path at ' + file + ':' + _pointer(path))
                    replacement = kind + ':' + self.model_paths[value]
                if kind == 'model_dir':
                    replacement = kind + ':' + _hash(_encoded(self.model_pins))
                if kind == 'preview_model_path':
                    relative = 'models/' + path[0] + '.json'
                    if type(value) is not str or not value.endswith('/preview_models/' + relative):
                        raise ValueError('Unexpected preview installer model argument')
                    replacement = kind + ':' + _hash((self.root / 'preview_models' / relative).read_bytes())
                if kind == 'log_path' and value is not None:
                    leaf = Path(value).name.removesuffix('.gz')
                    if leaf not in [s + '.jsonl' for s in JSONL + OPTIONAL_JSONL]:
                        raise ValueError('Unregistered event-log path')
                    replacement = kind + ':' + leaf
                if kind == 'timeline_path' and value is not None:
                    replacement = kind + ':xapp_switch_timeline.json'
                if kind == 'mesh_path' and value is not None:
                    leaf = Path(value).name
                    if leaf not in OPTIONAL_NPZ or not (self.root / leaf).is_file():
                        raise ValueError('Missing corresponding mesh content')
                    replacement = kind + ':' + leaf
                if kind == 'owned_path' and value is not None:
                    leaf = Path(value).name
                    if path[-1] == 'comparison_spec' and (leaf != 'xapp_test_spec.json' or not (self.root / leaf).is_file()):
                        raise ValueError('Missing comparison-spec content')
                    if path[-1] == 'command_dir' and not leaf.endswith('.commands'):
                        raise ValueError('Unexpected command-directory shape')
                if kind == 'worker_status_list':
                    allowed = {'completed', 'cancelled', 'assigned_blocks', 'processed_blocks', 'generated_blocks',
                               'skipped_blocks', 'lock_wait_s', 'worker_wall_s'}
                    if not isinstance(value, list) or any(not isinstance(x, dict) or set(x) - allowed for x in value):
                        raise ValueError('Unknown prefetch status fields')
                    for row in value:
                        for k, v in row.items():
                            if (k in ('completed', 'cancelled') and type(v) is not bool) or (k not in ('completed', 'cancelled') and not _number(v)):
                                raise ValueError('Invalid prefetch status type')
                self.note(file, path, value)
                return {'__parity_metadata__': replacement, 'is_null': value is None}
        if isinstance(value, dict):
            return {k: self.transform(file, v, path + (k,)) for k, v in value.items()}
        if isinstance(value, list):
            return [self.transform(file, v, path + (str(i),)) for i, v in enumerate(value)]
        return value

    def exclusions(self):
        return {k: dict(count=v, ordered_original_value_sha256=self.hashes[k].hexdigest())
                for k, v in sorted(self.counts.items())}


def _inventory(root: Path) -> dict[str, Path]:
    result = {}
    known = ({s + '.jsonl' for s in JSONL + OPTIONAL_JSONL}
             | {s + '.csv' for s in CSV} | {s + '.json' for s in JSON + OPTIONAL_JSON}
             | set(OPTIONAL_NPZ) | {'network-replay.html'}
             | {'xapp_model_checkpoints/' + m + '.json' for m in MODEL_IDS})
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError('Symlink not admitted: ' + str(p.relative_to(root)))
        if not p.is_file():
            continue
        name = p.relative_to(root).as_posix()
        logical = name[:-3] if name.endswith('.gz') and name[:-3].endswith(('.jsonl', '.csv')) else name
        if logical not in known and not logical.startswith('preview_models/'):
            raise ValueError('Unknown extra file: ' + name)
        if logical in result:
            raise ValueError('Duplicate compressed/uncompressed stream: ' + logical)
        result[logical] = p
    required = ({s + '.jsonl' for s in JSONL} | {s + '.csv' for s in CSV}
                | {s + '.json' for s in JSON}
                | {'xapp_model_checkpoints/' + m + '.json' for m in MODEL_IDS})
    missing = sorted(required - result.keys())
    if missing:
        raise ValueError('Missing required files: ' + ', '.join(missing))
    return result


def _verify(audit: _Audit, files, source, origin):
    config = _read(files['config_snapshot.json'])
    status = _read(files['run_status.json'])
    proof = _read(files['evaluation_origin_verified.json'])
    if config.get('source_sha256') != source or status.get('source_sha256') != source:
        raise ValueError('Physical source SHA differs from expected frozen source')
    if status.get('completed') is not True or status.get('simulation_completed') is not True or status.get('stage') != 'completed':
        raise ValueError('Run is not successfully complete')
    if (proof.get('schema') != 'oran.evaluation-origin-verification.v1'
            or proof.get('origin_id') != origin
            or any(proof.get(k) is not True for k in ('verified_before_activation', 'observed_prefix_equal',
                'configuration_equal', 'models_equal', 'network_dynamics_code_equal'))
            or proof.get('after_kpis_used') is not False):
        raise ValueError('Verified common OFF origin is missing or differs')
    ric = config.get('configs', {}).get('ric', {})
    if ric.get('freeze_learning') is not True or ric.get('persistent_memory_enabled') is True:
        raise ValueError('Parity requires frozen learning and no persistent memory')
    hot = _read(files['xapp_hot_start.json'])
    models = hot.get('models', {})
    if set(models) != set(MODEL_IDS):
        raise ValueError('Complete six-model binding required')
    for app, item in models.items():
        pin = item.get('sha256')
        checkpoint = 'xapp_model_checkpoints/' + app + '.json'
        if (not isinstance(pin, str) or not _SHA.fullmatch(pin) or item.get('mode') != 'frozen'
                or item.get('checkpoint') != checkpoint or _hash(files[checkpoint].read_bytes()) != pin
                or type(item.get('path')) is not str):
            raise ValueError('Frozen model/checkpoint pin mismatch: ' + app)
        audit.model_paths[item['path']] = pin
        audit.model_pins[app] = pin
    configured_apps = ric.get('xapps', [])
    if (not isinstance(configured_apps, list) or len(configured_apps) != len(MODEL_IDS)
            or {a.get('implementation_id') for a in configured_apps} != set(MODEL_IDS)):
        raise ValueError('Complete unique six-app physical recipe required')
    for app in configured_apps:
        app_id = app.get('implementation_id')
        params = app.get('parameters', {})
        if app_id not in models or params.get('mode') != 'frozen' or params.get('model_path') != models[app_id]['path']:
            raise ValueError('Config app-to-model binding mismatch')
    timeline = hot.get('timeline_source')
    if timeline is not None and timeline.get('sha256') != _hash(files['xapp_switch_timeline.json'].read_bytes()):
        raise ValueError('Copied control timeline SHA mismatch')
    release_name = 'preview_models/release.json'
    if release_name not in files:
        raise ValueError('Missing complete preview release')
    release = _read(files[release_name])
    manifest = release.get('files_sha256')
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError('Invalid preview release manifest')
    expected = {release_name, 'preview_models/models.json', 'preview_models/runtime-arguments.json'}
    for name, pin in manifest.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or not isinstance(pin, str) or not _SHA.fullmatch(pin):
            raise ValueError('Invalid preview artifact path/pin')
        key = 'preview_models/' + name
        if key not in files or _hash(files[key].read_bytes()) != pin:
            raise ValueError('Preview release artifact pin mismatch: ' + name)
        expected.add(key)
    if expected != {k for k in files if k.startswith('preview_models/')}:
        raise ValueError('Unknown or missing preview release artifact')
    argv = _read(files['preview_models/runtime-arguments.json'])
    release_models = release.get('models', {})
    if set(argv) != set(release_models):
        raise ValueError('Preview installer model identities differ')
    for name, tokens in argv.items():
        mode = release_models[name].get('cli_mode')
        if not isinstance(mode, str) or not isinstance(tokens, list) or len(tokens) < 4 or tokens[:3] != ['--' + mode + '-xapp-mode', 'frozen', '--' + mode + '-xapp-model']:
            raise ValueError('Unexpected preview installer argv layout')
    total = config.get('manifest', {}).get('total_slots')
    if type(total) is not int or total < 1:
        raise ValueError('Explicit positive complete trace length required')
    return total


def _open_text(path):
    return gzip.open(path, 'rt', newline='') if path.name.endswith('.gz') else path.open(newline='')


def _records(file, path):
    if file.endswith('.jsonl'):
        with _open_text(path) as stream:
            for line in stream:
                if not line.strip():
                    raise ValueError('Blank JSONL physical record')
                yield _load(line)
    elif file.endswith('.csv'):
        csv.field_size_limit(sys.maxsize)
        with _open_text(path) as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            if header is None or len(header) != len(set(header)):
                raise ValueError('Missing/duplicate CSV header')
            yield {'header': header}
            for row in reader:
                if len(row) != len(header):
                    raise ValueError('CSV physical row width differs')
                values = []
                for cell in row:
                    if _NUM.fullmatch(cell):
                        values.append(Decimal(cell))
                    else:
                        if cell.lower() in ('nan', 'inf', '+inf', '-inf', 'infinity', '+infinity', '-infinity'):
                            raise ValueError('Nonfinite CSV number')
                        values.append(cell)
                yield values
    elif file.endswith('.npz'):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or any(not n.endswith('.npy') or '/' in n for n in names):
                raise ValueError('Unknown NPZ entries')
            for name in sorted(names):
                # NPY bytes include exact shape/dtype/data, not ZIP timestamps.
                yield dict(array=name, exact_npy_sha256=_hash(archive.read(name)))
    elif file == 'network-replay.html':
        raw = path.read_text()
        pattern = r'(<script id="replay-data" type="application/json">)(.*?)(</script>)'
        matches = list(re.finditer(pattern, raw, re.DOTALL))
        if len(matches) != 1:
            raise ValueError('Missing/ambiguous embedded replay data')
        m = matches[0]
        yield _load(m.group(2))
        yield {'html_template_sha256': _hash((raw[:m.start(2)] + '<PHYSICAL_DATA>' + raw[m.end(2):]).encode())}
    elif file == 'preview_models/runtime-arguments.json':
        yield _read(path)
    elif file.startswith('preview_models/') or file.startswith('xapp_model_checkpoints/'):
        yield {'exact_artifact_bytes_sha256': _hash(path.read_bytes())}
    else:
        yield _read(path)


def _differences(a, b, path=()):
    if _number(a) and _number(b):
        if Decimal(a) != Decimal(b):
            yield _pointer(path), 'exact_number_difference'
        return
    if type(a) is not type(b):
        yield _pointer(path), 'type_difference'
    elif isinstance(a, dict):
        for k in sorted(a.keys() | b.keys()):
            if k not in a or k not in b:
                yield _pointer(path + (k,)), 'field_missing'
            else:
                yield from _differences(a[k], b[k], path + (k,))
    elif isinstance(a, list):
        if len(a) != len(b):
            yield _pointer(path), 'sequence_length_difference'
        for i, (x, y) in enumerate(zip(a, b)):
            yield from _differences(x, y, path + (str(i),))
    elif a != b:
        yield _pointer(path), 'value_difference'


def compare_runs(reference_run, candidate_run, *, expected_source_sha256, expected_origin_id) -> dict:
    """Read-only proof; ``passed`` never authorizes or executes migration."""
    proof = dict(schema=SCHEMA, policy=POLICY, passed=False, physical_equal=False,
        expected_source_sha256=expected_source_sha256, expected_origin_id=expected_origin_id,
        tolerance=0, rounding=False, parity_is_not_an_independent_sample=True,
        no_automatic_action=True, streams={}, differences=[], counts={})
    proof['comparator_sha256'] = _hash(Path(__file__).read_bytes())
    spec = comparison_spec()
    proof['comparison_spec_sha256'] = _hash(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode())
    audits = [_Audit(Path(reference_run).resolve()), _Audit(Path(candidate_run).resolve())]
    try:
        if any(not isinstance(v, str) or not _SHA.fullmatch(v) for v in (expected_source_sha256, expected_origin_id)):
            raise ValueError('Expected source/origin SHA must be explicit SHA256')
        if audits[0].root == audits[1].root:
            raise ValueError('Parity must compare two independent run directories')
        inventories = [_inventory(a.root) for a in audits]
        total_slots = [_verify(a, files, expected_source_sha256, expected_origin_id)
                       for a, files in zip(audits, inventories)]
        if inventories[0].keys() != inventories[1].keys():
            raise ValueError('Physical/auxiliary file inventories differ')
        if audits[0].model_pins != audits[1].model_pins:
            raise ValueError('App-to-model identity differs')
        counts = Counter()
        for file in sorted(inventories[0]):
            paths = [files[file] for files in inventories]
            hashes = [hashlib.sha256(), hashlib.sha256()]
            n = [0, 0]
            different = False
            streams = [_records(file, p) for p in paths]
            for row_index, pair in enumerate(zip_longest(*streams, fillvalue=_MISSING)):
                values = []
                for side, value in enumerate(pair):
                    if value is _MISSING:
                        values.append(value); continue
                    if file == 'rt_sys_trace.jsonl' and (not isinstance(value, dict)
                            or type(value.get('slot_id')) is not int or value['slot_id'] != row_index + 1):
                        raise ValueError('Complete RT trace must contain every absolute slot in order')
                    value = audits[side].transform(file, value)
                    hashes[side].update(_encoded(value) + b'\n');n[side] += 1
                    values.append(value)
                if _MISSING in values:
                    issues = [('/', 'record_count_difference')]
                else:
                    issues = _differences(*values)
                for pointer, reason in issues:
                    different = True;counts['differences'] += 1
                    if len(proof['differences']) < MAX_DIFFERENCES:
                        proof['differences'].append(dict(file=file, record_index=row_index, pointer=pointer, reason=reason))
            proof['streams'][file] = dict(reference=dict(raw_sha256=_hash(paths[0].read_bytes()), records=n[0], physical_sha256=hashes[0].hexdigest()),
                candidate=dict(raw_sha256=_hash(paths[1].read_bytes()), records=n[1], physical_sha256=hashes[1].hexdigest()), exact_equal=not different)
            if file == 'rt_sys_trace.jsonl' and n != total_slots:
                raise ValueError('RT trace does not cover the complete configured simulation')
            counts['streams'] += 1;counts['reference_records'] += n[0];counts['candidate_records'] += n[1]
        proof['counts'] = dict(counts)
        proof['physical_equal'] = counts['differences'] == 0
        proof['passed'] = proof['physical_equal']
        proof['physical_sequence_sha256'] = {
            side: _hash(_encoded([[file, row[side]['physical_sha256']]
                                  for file, row in sorted(proof['streams'].items())]))
            for side in ('reference', 'candidate')}
    except (OSError, ValueError, TypeError, KeyError, csv.Error, zipfile.BadZipFile) as error:
        proof['differences'].append(dict(reason='fail_closed_admission_or_parse_error', error=str(error)))
    proof['metadata_excluded'] = dict(reference=audits[0].exclusions(), candidate=audits[1].exclusions())
    return proof
