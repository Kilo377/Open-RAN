"""Synthetic streams only: never opens an experimental run or a held-out label."""
from copy import deepcopy
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from experiments.compute_parity import (
    CSV, JSON, JSONL, MODEL_IDS, SCHEMA, _canonical, _differences,
    compare_runs, comparison_spec,
)

SOURCE = 'a' * 64
ORIGIN = 'b' * 64


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(folder, name, value):
    p = folder / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value))


def update(folder, name, edit):
    p = folder / name
    value = json.loads(p.read_text())
    edit(value)
    write(folder, name, value)


def make_run(root):
    root.mkdir()
    for name in JSONL:
        (root / (name + '.jsonl')).write_text('')
    for name in CSV:
        (root / (name + '.csv')).write_text('slot_id,ue_id,power\n1,ue_0,1.00000000000000001\n')
    for name in JSON:
        write(root, name + '.json', {})
    (root / 'rt_sys_trace.jsonl').write_text(json.dumps(dict(slot_id=1,
        system_gain_epoch=0, system_gain_statistics={'max_anchor_age_slots': 0},
        ues=[dict(ue_id='ue_0', tb_id='tb:1', ack=True, bler=0.02)],
        controls=[dict(request_id='timeline:0', action='light_sleep')])) + '\n')
    models = {}
    apps = []
    for name in MODEL_IDS:
        raw = json.dumps({'synthetic_model': name, 'weights': [1, 2, 3]}).encode()
        path = 'xapp_model_checkpoints/' + name + '.json'
        p = root / path;p.parent.mkdir(exist_ok=True);p.write_bytes(raw)
        source_path = str(root / 'owned_models' / (name + '.json'))
        models[name] = dict(path=source_path, sha256=sha(raw), checkpoint=path, mode='frozen', release='synthetic')
        apps.append(dict(implementation_id=name, parameters=dict(mode='frozen', model_path=source_path,
            log_path=str(root / 'ap_rf_events.jsonl'))))
    timeline = [dict(request_id='timeline:0', operation='set', apps=list(MODEL_IDS[:3]), time_ms=40)]
    write(root, 'xapp_switch_timeline.json', timeline)
    write(root, 'xapp_hot_start.json', dict(models=models,
        timeline=[dict(run_id='1' * 32, request_id='timeline:0', apps=list(MODEL_IDS[:3]), scheduled_slot=80)],
        timeline_source=dict(path=str(root / 'owned_timeline.json'), sha256=sha((root/'xapp_switch_timeline.json').read_bytes()))))
    write(root, 'config_snapshot.json', dict(source_sha256=SOURCE,
        record_compression='none', manifest={'total_slots':1}, configs={'ric':dict(freeze_learning=True,
            persistent_memory_enabled=False, xapps=apps)}))
    write(root, 'run_status.json', dict(source_sha256=SOURCE, completed=True,
        simulation_completed=True, stage='completed', record_compression='none',
        elapsed_s=1.2, output_bytes=1234))
    write(root, 'evaluation_origin_verified.json', dict(schema='oran.evaluation-origin-verification.v1',
        origin_id=ORIGIN, verified_before_activation=True, observed_prefix_equal=True,
        configuration_equal=True, models_equal=True, network_dynamics_code_equal=True,
        after_kpis_used=False, baseline_slot=80))
    write(root, 'final_summary.json', dict(xapp_test=dict(run_id='1' * 32,
        comparison_spec=str(root/'xapp_test_spec.json'), command_dir=str(root/'live.commands'),
        simulation_python='/owned/python3', model_artifacts=models),
        rt_sys=dict(cache_key='fixed', cache_hits=40, cache_misses=1),
        packet_metrics={'ue_0':{'delivered_bytes':200}}))
    write(root, 'xapp_test_spec.json', dict(arguments=dict(xapp_model_dir=str(root/'owned_models'),
        xapp_toggle_timeline=str(root/'owned_timeline.json'), record_compression='none'), slots=80))
    write(root, 'performance.json', dict(stages={'channel':{'calls':80,'wall_s':1}},
        actual_decodes=2, rt_prefetch_statistics=dict(workers=2, completed=False, generated_blocks=1,
        worker_statuses=[]), simulation_s=1.0))
    names = ['x01','x02','x08','x12_mec','x12_urllc','x13']
    release = dict(files_sha256={}, models={})
    argv = {}
    for i, name in enumerate(names):
        raw = json.dumps({'synthetic_preview_model':name,'weight':i}).encode()
        key = 'models/' + name + '.json'
        p = root/'preview_models'/key;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        release['files_sha256'][key] = sha(raw)
        release['models'][name] = dict(cli_mode=name)
        argv[name] = ['--'+name+'-xapp-mode','frozen','--'+name+'-xapp-model',
                      str(root/'preview_models'/key)]
    write(root, 'preview_models/release.json', release)
    write(root, 'preview_models/models.json', {'frozen':'pins'})
    write(root, 'preview_models/runtime-arguments.json', argv)
    return root


@pytest.fixture
def pair(tmp_path):
    return make_run(tmp_path/'reference'), make_run(tmp_path/'candidate')


def compare(pair):
    return compare_runs(*pair, expected_source_sha256=SOURCE, expected_origin_id=ORIGIN)


def test_identical_physics_relocation_is_exact_and_no_action(pair):
    result = compare(pair)
    assert result['passed'] and result['physical_equal']
    assert result['schema'] == SCHEMA and result['tolerance'] == 0
    assert result['rounding'] is False and result['no_automatic_action'] is True
    assert result['parity_is_not_an_independent_sample'] is True
    assert result['counts']['streams'] >= len(JSONL)+len(CSV)+len(JSON)+len(MODEL_IDS)
    assert result['metadata_excluded']['reference']['xapp_hot_start.json:/timeline/0/run_id']['count'] == 1


def test_compression_containers_and_numeric_spellings_do_not_round(pair):
    candidate = pair[1]
    for name in ('rt_sys_trace.jsonl','cell_metrics.csv'):
        p=candidate/name
        with gzip.open(p.with_name(p.name+'.gz'),'wb') as stream:stream.write(p.read_bytes())
        p.unlink()
    update(candidate,'config_snapshot.json',lambda d:d.update(record_compression='gzip'))
    update(candidate,'run_status.json',lambda d:d.update(record_compression='gzip'))
    update(candidate,'xapp_test_spec.json',lambda d:d['arguments'].update(record_compression='gzip'))
    assert compare(pair)['passed']
    with gzip.open(candidate/'cell_metrics.csv.gz','wt') as stream:
        stream.write('slot_id,ue_id,power\n1.0,ue_0,1.00000000000000001\n')
    assert compare(pair)['passed']
    with gzip.open(candidate/'cell_metrics.csv.gz','wt') as stream:
        stream.write('slot_id,ue_id,power\n1,ue_0,1.00000000000000002\n')
    result=compare(pair)
    assert not result['passed'] and any(x['reason']=='exact_number_difference' for x in result['differences'])


@pytest.mark.parametrize('file,key,value',[
    ('performance.json','simulation_s',55),('run_status.json','elapsed_s',31),
    ('run_status.json','peak_rss_bytes',1300000),
])
def test_only_predeclared_wall_and_resource_metadata_may_differ(pair,file,key,value):
    update(pair[0],file,lambda d:d.setdefault(key,1))
    update(pair[1],file,lambda d:d.update({key:value}))
    assert compare(pair)['passed']


def test_cache_producer_split_nonce_and_wall_changes_have_auditable_exclusions(pair):
    update(pair[1],'xapp_hot_start.json',lambda d:d['timeline'][0].update(run_id='2'*32))
    update(pair[1],'final_summary.json',lambda d:(d['xapp_test'].update(run_id='2'*32), d['rt_sys'].update(cache_hits=0,cache_misses=41)))
    update(pair[1],'performance.json',lambda d:d['rt_prefetch_statistics'].update(completed=True,generated_blocks=4,
        worker_statuses=[dict(completed=True,generated_blocks=4,worker_wall_s=5)]))
    result=compare(pair)
    assert result['passed']
    path='xapp_hot_start.json:/timeline/0/run_id'
    assert result['metadata_excluded']['reference'][path]['ordered_original_value_sha256'] != result['metadata_excluded']['candidate'][path]['ordered_original_value_sha256']


@pytest.mark.parametrize('edit',[
    lambda d:d.update(system_gain_epoch=1),
    lambda d:d['system_gain_statistics'].update(max_anchor_age_slots=1),
    lambda d:d['controls'][0].update(request_id='timeline:1'),
    lambda d:d['ues'][0].update(ue_id='ue_1'),
    lambda d:d['ues'][0].update(tb_id='tb:2'),
    lambda d:d['ues'][0].update(ack=False),
    lambda d:d['controls'][0].update(action='deep_sleep'),
    lambda d:d.update(unknown_future_numeric_field=0),
])
def test_no_physical_numeric_or_identity_or_unknown_field_is_excluded(pair,edit):
    p=pair[1]/'rt_sys_trace.jsonl';d=json.loads(p.read_text());edit(d);p.write_text(json.dumps(d)+'\n')
    assert not compare(pair)['passed']


def test_control_sequence_order_is_physical(pair):
    for root in pair:
        (root/'control_receipts.jsonl').write_text('{"request_id":"a"}\n{"request_id":"b"}\n')
    (pair[1]/'control_receipts.jsonl').write_text('{"request_id":"b"}\n{"request_id":"a"}\n')
    assert not compare(pair)['passed']


@pytest.mark.parametrize('mutation', ['missing','extra','duplicate','symlink','model','source','origin','incomplete','nonce_type','new_cache_field','new_wall_field'])
def test_admission_and_unknown_metadata_fail_closed(pair,mutation):
    root=pair[1]
    if mutation=='missing':(root/'harq_events.jsonl').unlink()
    elif mutation=='extra':write(root,'unregistered_physical_events.json',{'energy':2})
    elif mutation=='duplicate':
        with gzip.open(root/'harq_events.jsonl.gz','wb') as stream:stream.write(b'')
    elif mutation=='symlink':(root/'unregistered_link').symlink_to(root/'harq_events.jsonl')
    elif mutation=='model':(root/'xapp_model_checkpoints'/f'{MODEL_IDS[0]}.json').write_text('{}')
    elif mutation=='source':update(root,'config_snapshot.json',lambda d:d.update(source_sha256='c'*64))
    elif mutation=='origin':update(root,'evaluation_origin_verified.json',lambda d:d.update(origin_id='c'*64))
    elif mutation=='incomplete':update(root,'run_status.json',lambda d:d.update(completed=False))
    elif mutation=='nonce_type':update(root,'xapp_hot_start.json',lambda d:d['timeline'][0].update(run_id=2))
    elif mutation=='new_cache_field':
        update(root,'performance.json',lambda d:d['rt_prefetch_statistics'].update(worker_statuses=[dict(physical_gain=0)]))
    else:update(root,'performance.json',lambda d:d.update(new_wall_s=2))
    assert not compare(pair)['passed']


@pytest.mark.parametrize('payload',['{"slot":1,"x":1,"x":2}\n','{"slot":1,"x":NaN}\n','\n'])
def test_ambiguous_and_nonfinite_jsonl_rejected(pair,payload):
    (pair[1]/'harq_events.jsonl').write_text(payload)
    assert not compare(pair)['passed']


def test_exact_decimal_canonicalization_never_uses_decimal_context_rounding():
    x=Decimal('1.123456789012345678901234567890123456789')
    y=Decimal('1.123456789012345678901234567890123456788')
    assert _canonical(x)!=_canonical(y)
    assert list(_differences(x,y))
    assert _canonical(Decimal('1.00'))==_canonical(1)
    assert _canonical(False)!=_canonical(0)
    assert _canonical(Decimal('-0.000'))==_canonical(0)


def test_live_payload_physical_fields_are_kept_third_nonce_is_explicit(pair):
    for i,root in enumerate(pair):
        # Arbitrary bytes suffice here; real NPZ comparison is separately tested.
        with zipfile.ZipFile(root/'live.mesh.npz','w') as stream:stream.writestr('vertices.npy',b'EXACT NPY fixture')
        write(root,'live.json',dict(wall_s=i+1,updated_unix_s=i+20,mesh_path=str(root/'live.mesh.npz'),
            xapp_test=dict(run_id=str(i+1)*32),cells=[dict(cell_id='cell_0',power=1)]))
    assert compare(pair)['passed']
    update(pair[1],'live.json',lambda d:d['cells'][0].update(power=2))
    assert not compare(pair)['passed']


def test_npz_zip_metadata_ignored_but_uncompressed_array_bytes_exact(pair):
    for i,root in enumerate(pair):
        with zipfile.ZipFile(root/'replay.mesh.npz','w',compression=zipfile.ZIP_DEFLATED if i else zipfile.ZIP_STORED) as stream:
            info=zipfile.ZipInfo('vertices.npy',date_time=(2020+i,1,1,0,0,0));stream.writestr(info,b'EXACT NPY')
    assert compare(pair)['passed']
    with zipfile.ZipFile(pair[1]/'replay.mesh.npz','w') as stream:stream.writestr('vertices.npy',b'CHANGED NPY')
    assert not compare(pair)['passed']


def test_replay_embedded_numbers_and_template_are_compared(pair):
    for i,root in enumerate(pair):
        data=dict(meta=dict(wall_s=i+1,source_run=root.name),frames=[dict(power=2)])
        (root/'network-replay.html').write_text('<html><script id="replay-data" type="application/json">'+json.dumps(data)+'</script></html>')
    assert compare(pair)['passed']
    p=pair[1]/'network-replay.html';p.write_text(p.read_text().replace('"power": 2','"power": 3'))
    assert not compare(pair)['passed']


def test_metadata_field_presence_and_type_cannot_hide_differences(pair):
    update(pair[1],'performance.json',lambda d:d.pop('simulation_s'))
    assert not compare(pair)['passed']


@pytest.mark.parametrize('trace',['','{"slot_id":2}\n','{"slot_id":1}\n{"slot_id":1}\n'])
def test_even_matching_incomplete_or_reordered_trace_cannot_pass(pair,trace):
    for root in pair:(root/'rt_sys_trace.jsonl').write_text(trace)
    assert not compare(pair)['passed']


def test_whitelist_is_serializable_and_contains_only_three_exact_nonce_locations():
    spec=comparison_spec()
    assert json.loads(json.dumps(spec))==spec
    nonce={(r['file'],r['pointer']) for r in spec['metadata_rules'] if r['type']=='nonce'}
    assert nonce=={('xapp_hot_start.json','/timeline/*/run_id'),
                   ('final_summary.json','/xapp_test/run_id'),('live.json','/xapp_test/run_id')}
    assert all(r['pointer'].startswith('/') and r['reason'] for r in spec['metadata_rules'])
