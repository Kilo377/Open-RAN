"""Sampling boundaries, mode isolation and official SYS probability equivalence."""
import json
import numpy as np
import pytest
from phy.rt_sys.fast_profile import configure_fast_profile
from phy.rt_sys.sampling import sample_coordinates
from phy.rt_sys.efficiency import resource_patterns


def test_sampling_stays_inside_prb_bin_and_exact_stream_pattern():
    ns,nf,offset=14,64,8
    first=np.zeros((ns,nf),dtype=bool);first[:,offset:offset+48]=True
    second=first.copy();second[:7,offset+6:offset+18]=False
    power=first.astype(float);power[:,offset+30:offset+33]=0
    entries=[dict(mask=first,power=power),dict(mask=second,power=1)]
    patterns=resource_patterns(entries,ns,nf)
    for owners,coords in patterns.items():
        selected,inverse=sample_coordinates(coords,4,offset)
        assert len(inverse)==len(coords) and set(selected)<=set(coords)
        for old,new in zip(coords,(selected[i] for i in inverse)):
            assert (old[1]-offset)//3==(new[1]-offset)//3
            assert tuple(i for i,e in enumerate(entries) if e['mask'][new] and np.broadcast_to(e['power'],(ns,nf))[new]>0)==owners
    assert not np.any(first!=entries[0]['mask'])
    coords=[(s,f) for s in range(14) for f in range(offset,offset+48)]
    selected,inverse=sample_coordinates(coords,4,offset)
    assert len(selected)==16 and {s for s,f in selected}=={6}
    assert np.array_equal(inverse,sample_coordinates(coords,4,offset)[1])


def test_sampling_preserves_mu_resource_counts_dmrs_budget_and_rf_power():
    from test_rt_sys import FixtureAbstraction,channel,phy,ue
    from phy.rt_sys.adapter import RtSysPhyAdapter
    snapshot=channel(np.array([[[[1e-5,0]]],[[[0,1e-5]]]]),('u0','u1'),('c0',))
    plan=[phy('c0',[ue('u0'),ue('u1')])]
    outputs=[];traces=[]
    for profile in ('reference','efficient'):
        adapter=RtSysPhyAdapter(dict(device='numpy',feedback_mode='fast',fast_profile=profile),abstraction=FixtureAbstraction())
        outputs.append(adapter.run_cluster_slot(plan,snapshot)['c0'])
        traces.append(adapter.trace)
    for a,b in zip(outputs[0].ue_results,outputs[1].ue_results):
        for key in ('effective_data_re_count','resource_grid_re_count','delivered_bytes','tb_size_bytes'):
            assert getattr(a,key)==getattr(b,key)
        assert a.sinr_db==pytest.approx(b.sinr_db)
        assert a.effective_data_re_count<a.resource_grid_re_count
    assert traces[0]['cells']==traces[1]['cells']
    for a,b in zip(traces[0]['ues'],traces[1]['ues']):
        assert (a['mu_data_re'],a['su_data_re'])==(b['mu_data_re'],b['su_data_re'])


def test_profile_resolution_and_full_isolation():
    c=dict(feedback_mode='fast',device='cuda',algebra_backend='cupy')
    assert configure_fast_profile(c)=='system'
    assert c['algebra_device']=='cpu' and c['spatial_samples_per_prb']==4
    assert c['fast_channel_mode']=='sampled'
    assert configure_fast_profile(c,'reference')=='reference'
    assert c['algebra_device']=='cupy' and c['spatial_samples_per_prb'] is None
    c['feedback_mode']='nr_direct'
    assert configure_fast_profile(c) is None and 'algebra_device' not in c
    with pytest.raises(ValueError,match='requires'):configure_fast_profile(c,'efficient')


def test_default_fast_matches_validated_named_preset_and_freezes_live_arguments(monkeypatch,capsys):
    from experiments.run_rt_sys import parse_args,run_case,ROOT
    base=['runner','--dry-run','--device','cuda','--rt-device','cpu','--scene','munich',
        '--cells','3','--ues','6','--slots','400','--model','fast']
    results=[]
    for config in ['agx_munich_6ue_depth2_fast.json','agx_munich_6ue_depth2_fast_10ms.json']:
        monkeypatch.setattr('sys.argv',base+['--config',str(ROOT/'configs/rt_sys'/config)])
        args=parse_args();run_case(args);results.append(json.loads(capsys.readouterr().out))
        assert args.path_update_ms==10 and args.max_geometry_displacement_m==.015
        assert args.rt_prefetch_workers==2 and args.fast_channel_mode=='sampled'
    assert results[0]['manifest']==results[1]['manifest']
    assert results[0]['phy_config']==results[1]['phy_config']
    from phy.rt_sys.channel import geometry_blocks
    assert geometry_blocks(results[0]['manifest'])[1]==20


@pytest.mark.parametrize('model_args',[[],['--model','full'],['--model','fast','--fast-profile','efficient'],['--model','fast','--fast-profile','reference']])
def test_full_and_older_profiles_keep_configured_geometry(monkeypatch,capsys,model_args):
    from experiments.run_rt_sys import parse_args,run_case,ROOT
    monkeypatch.setattr('sys.argv',['runner','--dry-run','--device','cpu','--scene','munich','--cells','3','--ues','6',
        '--config',str(ROOT/'configs/rt_sys/agx_munich_6ue_depth2_fast.json'),*model_args])
    run_case(parse_args());result=json.loads(capsys.readouterr().out)
    assert result['manifest']['path_update_ms']==5
    assert result['manifest']['max_geometry_displacement_m']==.01
    assert result['phy_config']['rt_sys'].get('fast_channel_mode','dense')=='dense'
    assert 'rt_prefetch_workers' not in result['phy_config']['rt_sys']


def test_fast_default_allows_exact_five_ms_dense_reference_overrides(monkeypatch,capsys):
    from experiments.run_rt_sys import parse_args,run_case
    monkeypatch.setattr('sys.argv',['runner','--dry-run','--device','cpu','--model','fast',
        '--path-update-ms','5','--max-geometry-displacement-m','.01','--fast-channel-mode','dense',
        '--rt-prefetch-workers','1'])
    args=parse_args();run_case(args);result=json.loads(capsys.readouterr().out)
    assert result['manifest']['path_update_ms']==5 and result['manifest']['max_geometry_displacement_m']==.01
    assert result['phy_config']['rt_sys']['fast_channel_mode']=='dense'
    assert result['phy_config']['rt_sys']['rt_prefetch_workers']==1
    assert args.path_update_ms==5 and args.max_geometry_displacement_m==.01


@pytest.mark.parametrize('platform,prefetch',[('agx',True),('dgx',False)])
def test_sampled_default_retains_platform_rt_and_selects_host_channel_execution(monkeypatch,capsys,tmp_path,platform,prefetch):
    from experiments.run_rt_sys import parse_args,run_case,ROOT
    config=json.loads((ROOT/'configs/rt_sys/default.json').read_text())
    config['rt_sys'].update(retain_channel_on_gpu=True,rt_prefetch=True)
    path=tmp_path/'retained.json';path.write_text(json.dumps(config))
    monkeypatch.setattr('sys.argv',['runner','--dry-run','--model','fast','--platform',platform,'--config',str(path)])
    run_case(parse_args());result=json.loads(capsys.readouterr().out);phy=result['phy_config']['rt_sys']
    assert phy['retain_channel_on_gpu'] is False and phy['channel_array_backend']=='numpy'
    assert phy['rt_prefetch'] is prefetch and phy['rt_prefetch_workers']==2
    assert result['execution_devices']['rt']==('cpu' if platform=='agx' else 'cuda')
    assert result['execution_devices']['channel_reconstruction']=='cpu'


def test_config_profile_cli_precedence(monkeypatch,capsys,tmp_path):
    from experiments.run_rt_sys import parse_args,run_case,ROOT
    c=json.loads((ROOT/'configs/rt_sys/default.json').read_text())
    c['rt_sys'].update(model='fast',fast_profile='reference')
    path=tmp_path/'config.json';path.write_text(json.dumps(c))
    for override,expected in [([], 'reference'),(['--fast-profile','efficient'], 'efficient')]:
        monkeypatch.setattr('sys.argv',['runner','--dry-run','--device','cpu','--config',str(path),*override])
        run_case(parse_args());r=json.loads(capsys.readouterr().out)
        assert r['phy_config']['rt_sys']['fast_profile']==expected


@pytest.mark.parametrize('profile',['efficient','reference'])
def test_cli_profile_and_actual_device_metadata(monkeypatch,capsys,profile):
    from experiments.run_rt_sys import parse_args,run_case
    monkeypatch.setattr('sys.argv',['runner','--dry-run','--model','fast','--device','cuda','--rt-device','cpu','--fast-profile',profile])
    run_case(parse_args());r=json.loads(capsys.readouterr().out)
    assert r['phy_config']['rt_sys']['fast_profile']==profile
    assert r['execution_devices']['phy_sys']==('cpu' if profile=='efficient' else 'cuda')
    assert r['execution_devices']['nr_fallback']=='cuda'
    assert r['execution_devices']['linear_algebra']==('cpu' if profile=='efficient' else 'cuda')
    monkeypatch.setattr('sys.argv',['runner','--dry-run','--model','full','--fast-profile',profile])
    with pytest.raises(ValueError,match='requires'):run_case(parse_args())


@pytest.mark.parametrize('platform,rt_device',[('dgx','cuda'),('agx','cpu')])
def test_fast_profile_keeps_platform_ray_tracing_selection(monkeypatch,capsys,platform,rt_device):
    from experiments.run_rt_sys import parse_args,run_case
    monkeypatch.setattr('sys.argv',['runner','--dry-run','--model','fast','--platform',platform])
    run_case(parse_args());r=json.loads(capsys.readouterr().out)
    assert r['execution_devices']['rt']==rt_device
    assert r['execution_devices']['linear_algebra']=='cpu'
    assert r['execution_devices']['phy_sys']=='cpu'


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_batched_cpu_sys_matches_reference_with_aliases_and_unequal_lengths(device):
    pytest.importorskip('sionna')
    import torch
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from phy.rt_sys.fast import FastAbstraction
    old=FastAbstraction(device,config={'fast_profile':'reference'})
    new=FastAbstraction(device)
    rng=np.random.default_rng(9201);items=[]
    for table in (1,3):
        for mcs in (1,3,5,11,16,21,27):
            for count in (324,648,2052):
                wave=new.waveform(table,mcs,count)
                items.append((wave,rng.lognormal(float(rng.uniform(-3,4)),1.1,count)))
    expected=[old.probability(*item) for item in items]
    actual=new.probability_many(items)
    for a,b in zip(expected,actual):
        assert a[2]==b[2]
        if a[0] is not None:
            assert a[0]==pytest.approx(b[0],abs=1e-5)
            assert a[1]==pytest.approx(b[1],rel=1e-5,abs=1e-6)
    assert old.alias_counts==new.alias_counts
    assert new.profile=='system' and new.feedback_device=='cpu'


def test_batched_feedback_preserves_scalar_harq_events_and_counters():
    pytest.importorskip('sionna')
    from phy.rt_sys.fast import FastAbstraction
    scalar=FastAbstraction('cpu');batch=FastAbstraction('cpu')
    def requests(slot,retx):
        return [dict(sinr=np.ones(n)*(.01 if slot==1 else 100),table='qam64',mcs=mcs,seed=501,slot=slot,ue_id=u,
            harq_context=dict(process_id=1,ndi=1,is_retx=retx,cell_id='c0')) for u,mcs,n in [('u',1,324),('v',11,648)]]
    for slot in (1,2):
        rows=requests(slot,slot>1)
        a=[scalar.evaluate(**r) for r in rows];b=batch.evaluate_many(rows)
        for x,y in zip(a,b):
            for k in ('ack','capacity_bytes','actual_decode','ack_source','harq_reference'):
                assert x[k]==y[k]
    assert scalar.model_statistics()==batch.model_statistics()
