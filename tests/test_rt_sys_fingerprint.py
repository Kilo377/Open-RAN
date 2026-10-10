from types import SimpleNamespace
import numpy as np
from phy.rt_sys.channel import SionnaRayTracer


def test_cache_ignores_runtime_shape_counters_but_tracks_materials(tmp_path,monkeypatch):
    asset=tmp_path/'scene.xml';asset.write_text('<scene/>')
    (tmp_path/'building.ply').write_bytes(b'fixed mesh bytes')
    monkeypatch.setattr('phy.rt_sys.channel.version',lambda package:'test')
    tracer=SionnaRayTracer.__new__(SionnaRayTracer)
    tracer.m={'scene':'test'}
    tracer.rt=SimpleNamespace(scene=SimpleNamespace(test=str(asset)))
    tracer.mi=SimpleNamespace(variant=lambda:'fixture',traverse=lambda mesh:mesh)
    def obj(conductivity):
        return SimpleNamespace(mi_mesh={'vertex_positions':np.array([conductivity,0,0]),'faces':np.array([0])},radio_material=SimpleNamespace(name='concrete',
            relative_permittivity=np.array([5.]),conductivity=np.array([conductivity])))
    tracer.scene=SimpleNamespace(objects={'no-name-9':obj(.2),'no-name-10':obj(.5)})
    first=tracer.fingerprint()
    tracer.scene.objects={'no-name-102':obj(.5),'no-name-101':obj(.2)}
    assert tracer.fingerprint()==first
    tracer.scene.objects['no-name-102'].radio_material.conductivity=np.array([.6])
    assert tracer.fingerprint()!=first
