"""Exact, read-only diagnostic of a render-mesh vertex-index permutation.

This does not change a parity policy or authorize a simulation.  A vertex and
triangle multiset alone is insufficient: the candidate index bijection is also
verified against complete face connectivity and ordered per-vertex incidence.
"""
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counter_sha(values):
    h=hashlib.sha256()
    for key,count in sorted(values.items()):
        h.update(len(key).to_bytes(8,'big'));h.update(key);h.update(count.to_bytes(8,'big'))
    return h.hexdigest()


def _load(path):
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'vertices','faces'}:raise ValueError('Exactly vertices/faces required; no attribute may be ignored')
        v,f=z['vertices'],z['faces']
    if v.dtype!=np.dtype('<f4') or f.dtype!=np.dtype('<i4'):
        raise ValueError('Expected original float32 coordinates and int32 face indices')
    if v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3 or not len(v) or not len(f):
        raise ValueError('Invalid complete mesh dimensions')
    if not np.isfinite(v).all() or f.min()<0 or f.max()>=len(v):raise ValueError('Invalid coordinates or face range')
    return v,f


def _rows(vertices):
    return [np.ascontiguousarray(v).tobytes() for v in vertices]


def _world_faces(rows,faces):
    result=Counter()
    for face in faces:
        a,b,c=(rows[int(i)] for i in face)
        result[min(a+b+c,b+c+a,c+a+b)]+=1
    return result


def _signatures(rows,faces):
    incident=[[] for _ in rows]
    for face in faces:
        a,b,c=map(int,face)
        incident[a].append(rows[b]+rows[c]);incident[b].append(rows[c]+rows[a]);incident[c].append(rows[a]+rows[b])
    return [(row,tuple(sorted(around))) for row,around in zip(rows,incident)]


def _cyclic_face(face):
    a,b,c=map(int,face)
    return min((a,b,c),(b,c,a),(c,a,b))


def _ordered_incidence(faces,vertices):
    result=[[] for _ in range(vertices)]
    for face in faces:
        record=tuple(map(int,face))
        for v in record:result[v].append(record)
    return result


def diagnose(reference,candidate,*,witness=None):
    paths=[Path(reference),Path(candidate)];arrays=[_load(p) for p in paths]
    (va,fa),(vb,fb)=arrays
    result=dict(schema='oran.exact-render-mesh-index-bijection-diagnostic.v1',
        diagnostic_only=True,parity_policy_modified=False,primary_admission_authorized=False,
        reference=dict(path=str(paths[0]),sha256=sha(paths[0])),candidate=dict(path=str(paths[1]),sha256=sha(paths[1])),
        arrays=['vertices','faces'],colors_or_normals_or_material_attributes_present=False,
        no_recorded_attribute_omitted=True,unrecorded_RT_materials_must_be_verified_separately=True,
        dtype_equal=va.dtype==vb.dtype and fa.dtype==fb.dtype,shape_equal=va.shape==vb.shape and fa.shape==fb.shape,
        raw_array_bytes_equal=va.tobytes()==vb.tobytes() and fa.tobytes()==fb.tobytes(),
        vertices=dict(reference_shape=list(va.shape),candidate_shape=list(vb.shape),dtype=str(va.dtype)),
        faces=dict(reference_shape=list(fa.shape),candidate_shape=list(fb.shape),dtype=str(fa.dtype)),
        no_tolerance=True,no_rounding=True,no_vertex_removed=True,no_winding_reversal=True,
        raw_vertex_multiset_equal=False,oriented_world_triangle_multiset_equal=False,
        explicit_vertex_bijection_verified=False,world_coordinate_bits_preserved=False,
        mapped_oriented_faces_equal=False,mapped_literal_faces_equal=False,ordered_vertex_incidence_equal=False,
        topology_and_normal_accumulation_order_verified=False,geometry_equivalent_under_verified_reindexing=False)
    rows=[_rows(v) for v,_ in arrays]
    counters=[Counter(r) for r in rows];triangles=[_world_faces(r,f) for r,(_,f) in zip(rows,arrays)]
    result.update(raw_vertex_multiset_equal=counters[0]==counters[1],oriented_world_triangle_multiset_equal=triangles[0]==triangles[1],
        raw_vertex_multiset_sha256=[counter_sha(c) for c in counters],oriented_world_triangle_multiset_sha256=[counter_sha(c) for c in triangles])
    if not result['shape_equal']:return result
    result['vertices']['different_raw_component_bits']=int(np.count_nonzero(va.view('<u4')!=vb.view('<u4')))
    result['faces']['different_index_components']=int(np.count_nonzero(fa!=fb))
    if not result['raw_vertex_multiset_equal'] or not result['oriented_world_triangle_multiset_equal']:return result
    buckets=[]
    for r,(_,f) in zip(rows,arrays):
        groups=defaultdict(list)
        for i,sig in enumerate(_signatures(r,f)):groups[sig].append(i)
        buckets.append(groups)
    if buckets[0].keys()!=buckets[1].keys() or any(len(v)!=len(buckets[1][k]) for k,v in buckets[0].items()):
        result['failure_reason']='No coordinate-and-oriented-incidence-preserving candidate bijection'
        return result
    mapping=np.empty(len(va),dtype='<u4')
    for sig,old in buckets[0].items():
        for i,j in zip(old,buckets[1][sig]):mapping[i]=j
    bijective=len(np.unique(mapping))==len(va) and int(mapping.min())==0 and int(mapping.max())==len(vb)-1
    coordinates=all(rows[0][i]==rows[1][int(j)] for i,j in enumerate(mapping))
    mapped=mapping[fa]
    oriented=Counter(_cyclic_face(f) for f in mapped)==Counter(_cyclic_face(f) for f in fb)
    literal=Counter(tuple(map(int,f)) for f in mapped)==Counter(tuple(map(int,f)) for f in fb)
    old_incident=_ordered_incidence(mapped,len(vb));new_incident=_ordered_incidence(fb,len(vb))
    ordered=old_incident==new_incident
    result.update(explicit_vertex_bijection_verified=bool(bijective and coordinates and oriented),
        world_coordinate_bits_preserved=coordinates,mapped_oriented_faces_equal=oriented,mapped_literal_faces_equal=literal,
        ordered_vertex_incidence_equal=ordered,topology_and_normal_accumulation_order_verified=bool(bijective and coordinates and literal and ordered),
        geometry_equivalent_under_verified_reindexing=bool(bijective and coordinates and oriented and literal and ordered),
        vertex_bijection_sha256=hashlib.sha256(mapping.tobytes()).hexdigest(),
        candidate_assignment='Stable coordinate+oriented-incidence-signature buckets; acceptance requires full independent bijection/face/incidence verification, not bucket equality alone')
    if witness is not None:
        path=Path(witness)
        if path.exists():raise ValueError('Preserve prior diagnostic witness')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as out:np.save(out,mapping,allow_pickle=False)
        path.chmod(0o444);result['explicit_bijection_witness']=dict(path=str(path),sha256=sha(path),dtype='<u4',shape=[len(mapping)])
    return result
