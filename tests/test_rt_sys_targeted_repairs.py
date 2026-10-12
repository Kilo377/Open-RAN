from types import SimpleNamespace
import json
import numpy as np
import pytest
from phy.rt_sys.channel import geometry_blocks
from phy.rt_sys.calibration import artifact_digest,load_artifact,monotone


def test_motion_cap_and_direction_change_force_geometry_refresh():
    m={'slot_duration_ms':.5,'path_update_ms':10,'total_slots':100,'max_geometry_displacement_m':.01,
       'ues':[{'trajectory':{'times_s':[0,.0025,.05],'positions_m':[[0,0,0],[.025,0,0],[.025,.475,0]]}}]}
    blocks,step=geometry_blocks(m)
    assert step<=2 and 5 in blocks
    m['ues'][0]['trajectory']['positions_m'][1]=[1,0,0]
    with pytest.raises(ValueError,match='Per-slot'):geometry_blocks(m)


def test_default_uses_real_geometry_every_slot():
    from phy.rt_sys.scenario import build_case
    for scene in ('munich','simple_street_canyon'):
        m,_,_=build_case(scene,total_slots=100)
        blocks,step=geometry_blocks(m)
        assert step==1 and blocks==list(range(101))


def test_fitted_artifact_cannot_activate_without_independent_validation(tmp_path):
    a={'schema':'nr_tb_exact_re_v1','curves':{'1:3:324':{'snr_db':[-20,0,40],'tbler':[1,.5,0],'num_cb':1}},'beta':{'1:3:324':1.},'waveform':{},'validation':{'passed':False}}
    a['digest']=artifact_digest(a);f=tmp_path/'a.json';f.write_text(json.dumps(a))
    with pytest.raises(ValueError,match='independent'):load_artifact(f)
    assert load_artifact(f,True)['digest']==a['digest']
    a['validation']={'passed':True,'fitted_digest':a['digest']};f.write_text(json.dumps(a))
    with pytest.raises(ValueError,match='actual RT-profile'):load_artifact(f)
    a['validation']['validated_profiles']=['held_out_rt_lmmse_profiles'];f.write_text(json.dumps(a));assert load_artifact(f)
    a['beta']['1:3:324']=2.;f.write_text(json.dumps(a))
    with pytest.raises(ValueError,match='digest'):load_artifact(f)


def test_isotonic_bler_keeps_probability_bounds():
    y=monotone([1.,.8,.9,.2,0]);np.testing.assert_allclose(y,[1,.85,.85,.2,0]);assert np.all(np.diff(y)<=0)


def test_exact_resource_sys_bridge_rejects_unknown_code_and_preserves_tbler():
    import torch
    from phy.rt_sys.calibration import calibrated_sys_class
    class Base:
        def __init__(self,**kw):self.device='cpu';self.dtype=torch.float32
        def call(self,mcs,sinr,eff,count,**kw):return self.get_bler(mcs,kw['mcs_table_index'],1,None,eff)
    a={'curves':{'1:3:324':{'snr_db':[-10,0,10],'tbler':[1,.36,0],'num_cb':2}}}
    phy=calibrated_sys_class(Base)(a)
    cb=phy.call(torch.tensor([[3]]),sinr_eff=torch.tensor([[1.]]),num_allocated_re=torch.tensor([[324]]),mcs_table_index=1)
    np.testing.assert_allclose(1-(1-cb.numpy())**2,[[.36]],atol=1e-6)
    with pytest.raises(ValueError,match='Unvalidated'):phy.call(torch.tensor([[3]]),sinr_eff=torch.tensor([[1.]]),num_allocated_re=torch.tensor([[325]]),mcs_table_index=1)
    assert phy._counts is None


def test_stable_eesm_preserves_flat_channel_without_exponential_underflow():
    import torch
    from phy.rt_sys.calibration import stable_eesm_class
    class Base:
        def __init__(self,**kw):pass
    e=stable_eesm_class(Base)({'beta':{'1:3:324':.1}})
    x=torch.ones((1,1,324,1,1))*10000
    np.testing.assert_allclose(e.call(x,torch.tensor([[3]]),1).numpy(),[[10000]],rtol=1e-6)


def test_adaptive_eesm_remains_monotone_under_power_scaling():
    from phy.rt_sys.calibration import beta_value
    from scipy.special import logsumexp
    profile=np.array([.1,.1,1.9,1.9])
    for slope in (0.,.5,1.,4.):
        spec={'cv_knots':[0.,.9],'parameters':[{'log_beta':0.,'slope':slope,'reference_db':0.}]*2}
        effective=[]
        for power in np.logspace(-4,4,100):
            values=profile*power;beta=beta_value(spec,values)
            effective.append(-beta*(logsumexp(-values/beta)-np.log(len(values))))
        assert np.all(np.diff(effective)>0)


def test_qam_information_mapping_preserves_flat_and_erasure_ceiling():
    from phy.rt_sys.mutual_information import effective_sinr,qam_information_table
    for q in (2,4,6,8):
        spec={'qam_bits':q}
        for db in (-5.,10.,20.):
            np.testing.assert_allclose(effective_sinr(spec,np.ones(100)*10**(db/10)),10**(db/10),rtol=1e-7)
        grid,mi=qam_information_table(q)
        assert np.all(np.diff(mi)>=0) and np.max(mi)<=q
        profile=np.array([1e-12,1e-12,1e8,1e8])
        eff=effective_sinr(spec,profile)
        np.testing.assert_allclose(np.interp(10*np.log10(eff),grid,mi),q/2,atol=.001)
