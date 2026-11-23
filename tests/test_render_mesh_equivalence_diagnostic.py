import numpy as np
import pytest
from experiments.render_mesh_equivalence_diagnostic import diagnose


def mesh(path,v,f,**extra):
    np.savez_compressed(path,vertices=np.asarray(v,dtype='<f4'),faces=np.asarray(f,dtype='<i4'),**extra)
    return path


def test_permuted_independent_blocks_have_exact_topology_and_incidence(tmp_path):
    v=[[0,0,0],[1,0,0],[0,1,0],[3,0,0],[4,0,0],[3,1,0]];f=[[0,1,2],[3,4,5]]
    a=mesh(tmp_path/'a.npz',v,f);b=mesh(tmp_path/'b.npz',v[3:]+v[:3],f)
    p=diagnose(a,b,witness=tmp_path/'bijection.npy')
    assert not p['raw_array_bytes_equal'] and p['geometry_equivalent_under_verified_reindexing']
    assert p['ordered_vertex_incidence_equal'] and p['explicit_vertex_bijection_verified']


def test_coordinate_bit_change_is_not_rounded(tmp_path):
    v=np.asarray([[0,0,0],[1,0,0],[0,1,0]],dtype='<f4');a=mesh(tmp_path/'a.npz',v,[[0,1,2]])
    v[1,0]=np.nextafter(v[1,0],np.float32(2));b=mesh(tmp_path/'b.npz',v,[[0,1,2]])
    p=diagnose(a,b);assert not p['raw_vertex_multiset_equal'] and not p['geometry_equivalent_under_verified_reindexing']


def test_winding_reversal_is_not_accepted(tmp_path):
    v=[[0,0,0],[1,0,0],[0,1,0]];a=mesh(tmp_path/'a.npz',v,[[0,1,2]]);b=mesh(tmp_path/'b.npz',v,[[0,2,1]])
    p=diagnose(a,b);assert p['raw_vertex_multiset_equal'] and not p['oriented_world_triangle_multiset_equal']


def test_unreferenced_vertex_cannot_be_deleted(tmp_path):
    v=[[0,0,0],[1,0,0],[0,1,0],[9,9,9]];a=mesh(tmp_path/'a.npz',v,[[0,1,2]]);b=mesh(tmp_path/'b.npz',v[:3],[[0,1,2]])
    assert not diagnose(a,b)['geometry_equivalent_under_verified_reindexing']


def test_attributes_are_not_silently_ignored(tmp_path):
    v=[[0,0,0],[1,0,0],[0,1,0]];a=mesh(tmp_path/'a.npz',v,[[0,1,2]]);b=mesh(tmp_path/'b.npz',v,[[0,1,2]],normals=np.ones((3,3)))
    with pytest.raises(ValueError,match='attribute'):diagnose(a,b)


def test_duplicate_coordinates_do_not_hide_a_topology_change(tmp_path):
    # Both world triangle multisets contain two identical geometric triangles,
    # but a shares vertices across the faces and b uses disjoint duplicate IDs.
    v=[[0,0,0],[1,0,0],[0,1,0],[0,0,0],[1,0,0],[0,1,0]]
    a=mesh(tmp_path/'a.npz',v,[[0,1,2],[0,1,2]]);b=mesh(tmp_path/'b.npz',v,[[0,1,2],[3,4,5]])
    p=diagnose(a,b)
    assert p['raw_vertex_multiset_equal'] and p['oriented_world_triangle_multiset_equal']
    assert not p['explicit_vertex_bijection_verified'] and not p['geometry_equivalent_under_verified_reindexing']


def test_face_reordering_that_changes_normal_accumulation_order_is_separate(tmp_path):
    v=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]];f=[[0,1,2],[0,2,3]]
    a=mesh(tmp_path/'a.npz',v,f);b=mesh(tmp_path/'b.npz',v,list(reversed(f)))
    p=diagnose(a,b);assert p['explicit_vertex_bijection_verified'] and p['mapped_literal_faces_equal']
    assert not p['ordered_vertex_incidence_equal'] and not p['geometry_equivalent_under_verified_reindexing']


def test_signed_zero_bits_are_retained(tmp_path):
    v=np.asarray([[0,0,0],[1,0,0],[0,1,0]],dtype='<f4');a=mesh(tmp_path/'a.npz',v,[[0,1,2]])
    v[0,0]=-0.0;b=mesh(tmp_path/'b.npz',v,[[0,1,2]])
    assert not diagnose(a,b)['raw_vertex_multiset_equal']
