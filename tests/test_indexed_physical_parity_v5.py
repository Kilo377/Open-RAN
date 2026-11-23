from copy import deepcopy
import pytest
from experiments import indexed_physical_parity_v5 as p


def values():
    proof=dict(meshes=[])
    for name in p.MESHES:
        proof['meshes'].append(dict(file=name,reference=dict(path='old',sha256='a'),candidate=dict(path='new',sha256='b'),raw_array_bytes_equal=False,
            geometry_equivalent_under_verified_reindexing=True,explicit_vertex_bijection_verified=True,
            world_coordinate_bits_preserved=True,mapped_literal_faces_equal=True,ordered_vertex_incidence_equal=True,
            topology_and_normal_accumulation_order_verified=True,object_block_partition_matches_actual_RT_fingerprint=True))
    cached=dict(passed=False,exact_difference_count=1,non_metadata_differences=[dict(file='live.mesh.npz',pointer='/exact_npy_sha256',reason='value_difference',record_index=0)],
        streams={'live.mesh.npz':dict(exact_equal=False),'replay.mesh.npz':dict(exact_equal=True),'rt_sys_trace.jsonl':dict(exact_equal=True)},checks={})
    return cached,proof


def test_mesh_semantics_retains_original_exact_byte_failure_and_hashes():
    cached,proof=values();before=deepcopy(cached);r=p.merge_verified_pair(cached,proof)
    assert r['passed'] and all(r['checks'].values())and r['non_metadata_differences']==[]
    assert r['streams']['live.mesh.npz']['exact_equal']is False
    assert r['streams']['live.mesh.npz']['geometry_equivalent_under_verified_reindexing']is True
    assert r['mesh_encoding_differences'][0]['original_v4_exact_issues']==before['non_metadata_differences']
    assert cached==before and r['mesh_encoding_differences'][0]['not_a_metadata_exclusion']


@pytest.mark.parametrize('file,pointer,reason',[('rt_sys_trace.jsonl','/power','exact_number_difference'),
    ('control_receipts.jsonl','/request_id','value_difference'),('final_summary.json','/prediction','exact_number_difference'),
    ('live.mesh.npz','/unknown','value_difference')])
def test_no_network_control_confidence_or_unknown_difference_is_discharged(file,pointer,reason):
    cached,proof=values();cached['non_metadata_differences']=[dict(file=file,pointer=pointer,reason=reason,record_index=0)]
    with pytest.raises(ValueError,match='cannot be discharged'):p.merge_verified_pair(cached,proof)


def test_non_render_stream_must_still_be_exact():
    cached,proof=values();cached['streams']['rt_sys_trace.jsonl']['exact_equal']=False
    with pytest.raises(ValueError,match='Non-render stream'):p.merge_verified_pair(cached,proof)


@pytest.mark.parametrize('key',['explicit_vertex_bijection_verified','world_coordinate_bits_preserved','mapped_literal_faces_equal',
    'ordered_vertex_incidence_equal','topology_and_normal_accumulation_order_verified','object_block_partition_matches_actual_RT_fingerprint'])
def test_every_strong_semantic_condition_is_required(key):
    cached,proof=values();proof['meshes'][0][key]=False
    with pytest.raises(ValueError,match='Missing exact mesh'):p.merge_verified_pair(cached,proof)


def test_no_missing_or_extra_mesh_proof():
    cached,proof=values();proof['meshes'].pop()
    with pytest.raises(ValueError,match='Complete mesh proof'):p.merge_verified_pair(cached,proof)
