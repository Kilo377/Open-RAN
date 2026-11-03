"""Long runs retain all records with bounded CSV-export scratch memory."""
import gzip
import json
import weakref
import pytest
from core.record_io import open_record
from core.result_writer import ResultWriter


def test_plain_and_compressed_csv_are_byte_identical(tmp_path):
    records=[dict(slot_id=1,true_metrics=dict(queue_bytes=100),metadata=['UE α',2]),
             dict(slot_id=2,true_metrics=dict(queue_bytes=90,new_field=None),extra=True)]
    a=ResultWriter(str(tmp_path/'plain'));b=ResultWriter(str(tmp_path/'gzip'));b.record_compression='gzip'
    a.write_slot_metrics(records);b.write_slot_metrics(records)
    expected=(a.output_dir/'slot_metrics.csv').read_bytes()
    assert gzip.decompress((b.output_dir/'slot_metrics.csv.gz').read_bytes())==expected
    with open_record(b.output_dir/'slot_metrics.csv') as f:assert 'true_queue_bytes' in f.readline()


def test_csv_export_does_not_retain_flattened_rows(tmp_path):
    live=peak=0
    class Row(dict):pass
    def released():
        nonlocal live
        live-=1
    class Writer(ResultWriter):
        def _flatten_record(self,record,**kwargs):
            nonlocal live,peak
            row=Row(record);live+=1;peak=max(peak,live)
            weakref.finalize(row,released)
            return row
    Writer(str(tmp_path)).write_slot_metrics([dict(slot_id=i) for i in range(1000)])
    assert peak<=2 and live==0


def test_compressed_replay_preserves_frames_and_audits(tmp_path):
    from test_rt_replay import fixture_run
    from experiments.replay import build_data,export_replay
    run=fixture_run(tmp_path/'run',sampled=True)
    expected=build_data(run)
    p=run/'rt_sys_trace.jsonl';original=p.read_bytes()
    with open_record(p,'r') as f:assert len(f.readlines())==17
    p.unlink()
    with open_record(p,'w',compression='gzip') as f:f.write(original.decode())
    actual=build_data(run)
    assert actual==expected
    info=export_replay(run)
    assert info['audits_passed'] and info['self_contained']


def test_record_reader_rejects_ambiguous_stale_formats(tmp_path):
    p=tmp_path/'events.jsonl';p.write_text('{}\n')
    with pytest.raises(FileExistsError):open_record(p,'w',compression='gzip')
    with gzip.open(str(p)+'.gz','wt') as f:f.write('{}\n')
    with pytest.raises(ValueError,match='Ambiguous'):open_record(p)


def test_live_child_keeps_compression_argument():
    from experiments.live_xapp_test import ARGUMENTS
    assert 'record_compression' in ARGUMENTS
