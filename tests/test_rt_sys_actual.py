"""Optional real PyTorch/SYS checks; no TensorFlow or approximate fallback."""
import importlib.metadata
import numpy as np
import pytest

try:
    installed=importlib.metadata.version('sionna')
except importlib.metadata.PackageNotFoundError:
    installed=None
pytestmark=pytest.mark.skipif(installed != '2.2.0',reason='Dedicated Sionna 2.2 environment required')


def test_real_torch_cluster_matches_numpy_reference():
    from test_rt_sys import channel,ue,phy,FixtureAbstraction
    from phy.rt_sys.adapter import RtSysPhyAdapter
    snap=channel(np.array([[[[1e-5,0]]],[[[0,1e-5]]]]),('u0','u1'),('c0',))
    inputs=[phy('c0',[ue('u0'),ue('u1')])]
    reference=RtSysPhyAdapter({'device':'numpy'},abstraction=FixtureAbstraction()).run_cluster_slot(inputs,snap)
    actual=RtSysPhyAdapter({'device':'cpu'},abstraction=FixtureAbstraction()).run_cluster_slot(inputs,snap)
    np.testing.assert_allclose([u.sinr_db for u in actual['c0'].ue_results],
                               [u.sinr_db for u in reference['c0'].ue_results],atol=1e-4)


def test_real_sys_mask_calibration_and_keyed_ack():
    from phy.rt_sys.abstraction import SionnaAbstraction
    model=SionnaAbstraction('cpu')
    values=np.ones((2,24,1))*10
    values[:,12:]=0
    a=model.evaluate(values,'qam64',0,7,1,'ue_0')
    model.evaluate(values,'qam64',4,7,1,'ue_1')
    b=model.evaluate(values,'qam64',0,7,1,'ue_0')
    assert a==b
    assert 0<=a['tbler']<=1 and a['capacity_bytes']>0
    assert a['mcs']==3 and a['mcs_projection']=='calibrated_table_projection:0->3'
    c=model.evaluate(values[:,:12],'qam64',0,7,1,'ue_0')
    assert a==c
    with pytest.raises(ValueError,match='EESM'):
        model.evaluate(values,'qam64_low_se',9,7,1,'ue_0')
