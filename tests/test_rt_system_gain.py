"""System abstraction retains dynamic interference, power and report isolation."""
from copy import deepcopy
import numpy as np
import pytest
from phy.rt_sys.adapter import RtSysPhyAdapter
from phy.rt_sys.fast_profile import configure_fast_profile
from phy.rt_sys.report_copy import copy_report


def run(profile,snapshot,plans,**config):
    from test_rt_sys import FixtureAbstraction
    a=RtSysPhyAdapter(dict(device='numpy',feedback_mode='fast',fast_profile=profile,**config),abstraction=FixtureAbstraction())
    return a,a.run_cluster_slot(plans,snapshot)


@pytest.mark.parametrize('mode',['mu','overlap','separate','silent','no_path','rank2'])
def test_system_anchor_matches_reference_mimo_and_resource_energy(mode):
    from test_rt_sys import channel,phy,ue
    rng=np.random.default_rng(123)
    if mode in ('mu','rank2'):
        h=(rng.normal(size=(2,1,2,4))+1j*rng.normal(size=(2,1,2,4)))*1e-5
        snap=channel(h,('u0','u1'),('c0',))
        plans=[phy('c0',[ue('u0',layers=2 if mode=='rank2' else 1),ue('u1',scheduled=mode!='rank2')])]
    else:
        h=(rng.normal(size=(2,2,2,4))+1j*rng.normal(size=(2,2,2,4)))*1e-5
        if mode=='no_path':h[:]=0
        snap=channel(h)
        plans=[phy('c0',[ue('u0')]),phy('c1',[ue('u1',scheduled=mode!='silent',start=1 if mode=='separate' else 0)])]
    old,x=run('efficient',snap,deepcopy(plans));new,y=run('system',snap,deepcopy(plans))
    for cid in x:
        for a,b in zip(x[cid].ue_results,y[cid].ue_results):
            assert a.sinr_db==pytest.approx(b.sinr_db,abs=1e-5)
            assert a.delivered_bytes==b.delivered_bytes
            assert a.effective_data_re_count==b.effective_data_re_count
    assert old.trace['cells']==new.trace['cells']
    for a,b in zip(old.trace['ues'],new.trace['ues']):
        assert (a['mu_data_re'],a['su_data_re'])==(b['mu_data_re'],b['su_data_re'])


def test_system_updates_power_sleep_pairing_beams_and_has_bounded_channel_age():
    from test_rt_sys import channel,phy,ue,FixtureAbstraction
    from dataclasses import replace
    h=np.ones((2,2,1,2))*1e-5;snapshot=channel(h)
    a=RtSysPhyAdapter(dict(device='numpy',feedback_mode='fast',fast_profile='system'),abstraction=FixtureAbstraction())
    def apply(slot,neighbor=0,scheduled=True,codebook=0):
        plans=[phy('c0',[ue('u0')]),phy('c1',[ue('u1',scheduled)])]
        for p in plans:p.slot_id=slot
        plans[1].tx_power_dbm=neighbor;plans[0].current_codebook_id=codebook
        return a.run_cluster_slot(plans,replace(snapshot,slot_id=slot))['c0'].ue_results[0].sinr_db
    low=apply(1);fields=a.system_gains.stats['field_builds']
    high=apply(2,neighbor=10)
    assert high<low-5 and a.system_gains.stats['field_builds']==fields
    silent=apply(3,scheduled=False);assert silent>low+5
    apply(4,codebook=1);assert a.system_gains.stats['field_builds']>fields
    builds=a.system_gains.stats['field_builds'];apply(10)
    assert a.system_gains.stats['field_builds']>builds
    assert len(a.system_gains.fields)<=64 and len(a.system_gains.combiners)<=64


def test_report_copy_keeps_delayed_mutable_reports_isolated():
    first=dict(ue={'u':dict(subband=[1,2],alloc=dict(prbs=[3]))})
    delayed=copy_report(first);delivered=copy_report(delayed)
    first['ue']['u']['subband'][0]=9
    delivered['ue']['u']['alloc']['prbs'].append(4)
    assert delayed==dict(ue={'u':dict(subband=[1,2],alloc=dict(prbs=[3]))})


def test_system_options_cannot_leak_into_full_or_reference():
    c=dict(feedback_mode='fast',device='cpu')
    configure_fast_profile(c,'system');assert c['system_phy_enabled']
    configure_fast_profile(c,'efficient');assert 'system_phy_enabled' not in c
    configure_fast_profile(c,'system');c['feedback_mode']='nr_direct'
    configure_fast_profile(c);assert 'reference_gain_update_ms' not in c


def test_reference_gain_cache_applies_power_sleep_and_beams_immediately():
    from types import SimpleNamespace as N
    from dataclasses import replace
    from test_rt_sys import channel
    from phy.rt_sys.paper_radio import ReferenceGainCache,references
    snap=channel(np.ones((1,1,1,2))*1e-5,('u0',),('c0',))
    cell=N(current_codebook_id=0,current_beam_profile_id=0,tx_power_dbm=30,cell_sleep_mode='active')
    state=N(managed_cells={'c0':N(cell=cell,resource_grid=N(n_prbs_total=2))},connections={'u0':N(serving_beam_id=0)})
    state.paper=N(config=dict(beam_reports=False),gain_cache=ReferenceGainCache(5.,.5))
    first=references(snap,state)['u0']['c0'];count=state.paper.gain_cache.builds
    cell.tx_power_dbm=27
    second=references(replace(snap,slot_id=2),state)['u0']['c0']
    assert second['rsrp_dbm']==pytest.approx(first['rsrp_dbm']-3.)
    assert state.paper.gain_cache.builds==count
    cell.cell_sleep_mode='deep_sleep';assert references(replace(snap,slot_id=3),state)['u0']['c0']['no_path']
    cell.cell_sleep_mode='active';cell.current_codebook_id=1
    references(replace(snap,slot_id=4),state);assert state.paper.gain_cache.builds>count
    count=state.paper.gain_cache.builds
    references(replace(snap,slot_id=10),state);assert state.paper.gain_cache.builds>count


def test_frequency_selective_system_anchor_and_per_prb_power_match():
    from test_rt_sys import channel,phy,ue,FixtureAbstraction
    from types import SimpleNamespace as N
    rng=np.random.default_rng(24)
    h=(rng.normal(size=(2,2,2,4))+1j*rng.normal(size=(2,2,2,4)))*1e-5
    snap=channel(h);snap.cfr*=rng.lognormal(0,.3,snap.cfr.shape)
    context=N(n_prbs_total=2,num_ofdm_symbols_per_slot=2,slot_duration_ms=.5,
        carrier_frequency_hz=3.5e9,bandwidth_hz=10e6,subcarrier_spacing_hz=30000,cyclic_prefix_type='normal')
    plans=[phy('c0',[ue('u0')]),phy('c1',[ue('u1')])]
    for p in plans:p.runtime_context=context
    rows=[]
    for profile in ('efficient','system'):
        adapter=RtSysPhyAdapter(dict(device='numpy',feedback_mode='fast',fast_profile=profile),abstraction=FixtureAbstraction())
        cell=lambda:N(prb_power_w=[.0002,.0008],ret_enabled=False)
        adapter.paper=N(state=N(managed_cells={'c0':N(cell=cell()),'c1':N(cell=cell())}))
        result=adapter.run_cluster_slot(deepcopy(plans),snap)
        rows.append(([result[c].ue_results[0].sinr_db for c in ['c0','c1']],adapter.trace['cells']))
    np.testing.assert_allclose(rows[0][0],rows[1][0],atol=1e-5)
    assert rows[0][1]==rows[1][1]


def test_system_gain_interval_cli_and_child_spec(monkeypatch,capsys):
    import json
    from experiments.run_rt_sys import parse_args,run_case
    from experiments.live_xapp_test import ARGUMENTS
    assert 'fast_gain_update_ms' in ARGUMENTS
    base=['runner','--dry-run','--device','cpu','--model','fast','--fast-profile','system']
    monkeypatch.setattr('sys.argv',base)
    default_args=parse_args();run_case(default_args);capsys.readouterr()
    assert default_args.fast_gain_update_ms==2
    monkeypatch.setattr('sys.argv',base+['--fast-gain-update-ms','2'])
    run_case(parse_args());p=json.loads(capsys.readouterr().out)['phy_config']['rt_sys']
    assert p['system_gain_update_ms']==p['reference_gain_update_ms']==2
    for value in ['.1','nan','-1']:
        monkeypatch.setattr('sys.argv',base+['--fast-gain-update-ms',value])
        with pytest.raises(ValueError,match='whole slots'):run_case(parse_args())
    monkeypatch.setattr('sys.argv',['runner','--dry-run','--model','fast','--fast-profile','efficient','--fast-gain-update-ms','2'])
    with pytest.raises(ValueError,match='requires'):run_case(parse_args())
