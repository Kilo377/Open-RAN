"""Physics and integration invariants; fixtures never claim Sionna execution."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from models.phy_input import PhyInput,PhyUeInput
from phy.rt_sys.adapter import RtSysPhyAdapter
from phy.rt_sys.channel import RtChannelProvider,position_at
from phy.rt_sys.contracts import ChannelSnapshot,keyed_uniform
from phy.rt_sys.control import audit_action
from models.action import ConfigAction
from phy.rt_sys.scenario import build_case


class FixtureAbstraction:
    def evaluate(self,sinr,table,mcs,seed,slot,ue_id,fixed=False):
        used=sinr[sinr>0]
        return dict(mcs=0,qm=2,rate=.5,capacity_bytes=10000,sinr_eff=float(used.mean()),
                    tbler=0,cbler=0,ack=True)


def ue(uid,scheduled=True,start=0,layers=1):
    return PhyUeInput(uid,1000,0,0,.1,0,scheduled,'embb','video',100,1,
        allocated_prbs=1,allocated_symbols=2,start_symbol=0,start_prb=start,
        allocated_subcarrier_indices=list(range(start*12,(start+1)*12)),
        allocated_symbol_indices=[0,1],layer_count=layers,requested_bytes=100,
        grant_bytes_budget=100,mcs_table_id='qam64')


def phy(cid,ues,power=0):
    return PhyInput(1,1,cid,power,0,0,ue_inputs=ues)


def channel(h,ue_ids=('u0','u1'),cell_ids=('c0','c1')):
    # h [UE,cell,RX,TX]; flat frequency channels for analytic tests
    full=np.broadcast_to(np.asarray(h)[...,None,None],np.asarray(h).shape+(2,24)).copy()
    return ChannelSnapshot(1,ue_ids,cell_ids,full,np.zeros((len(ue_ids),3)),'fixture',{'normalized':False})


class PhysicsTests(unittest.TestCase):
    def adapter(self): return RtSysPhyAdapter({'device':'numpy'},abstraction=FixtureAbstraction())

    def test_neighbor_silence_and_resource_separation_reduce_interference(self):
        snap=channel(np.ones((2,2,1,2))*1e-5)
        adapter=self.adapter()
        first=[phy('c0',[ue('u0')]),phy('c1',[ue('u1')])]
        overlap=adapter.run_cluster_slot(first,snap)['c0'].ue_results[0].sinr_db
        separate=adapter.run_cluster_slot([first[0],phy('c1',[ue('u1',start=1)])],snap)['c0'].ue_results[0].sinr_db
        silent=adapter.run_cluster_slot([first[0],phy('c1',[ue('u1',False)])],snap)['c0'].ue_results[0].sinr_db
        self.assertGreater(separate,overlap+5);self.assertAlmostEqual(separate,silent,places=5)

    def test_power_conservation_and_power_response(self):
        snap=channel(np.ones((1,1,1,2))*1e-5,('u0',),('c0',))
        adapter=self.adapter()
        low=adapter.run_cluster_slot([phy('c0',[ue('u0')],0)],snap)['c0'].ue_results[0]
        high=adapter.run_cluster_slot([phy('c0',[ue('u0')],10)],snap)['c0'].ue_results[0]
        self.assertAlmostEqual(high.sinr_db-low.sinr_db,10,places=5)
        t=adapter.trace['cells']['c0'];self.assertAlmostEqual(t['budget_w'],t['average_used_w'])
        self.assertEqual(high.delivered_bytes,100)

    def test_idle_no_path_and_sleep_zero_delivery(self):
        adapter=self.adapter()
        snap=channel(np.zeros((1,1,1,2)),('u0',),('c0',))
        self.assertEqual(adapter.run_cluster_slot([phy('c0',[ue('u0')])],snap)['c0'].ue_results[0].delivered_bytes,0)
        snap.cfr[:]=1e-5;snap.metadata['sleep_modes']={'c0':'deep_sleep'}
        self.assertEqual(adapter.run_cluster_slot([phy('c0',[ue('u0')])],snap)['c0'].ue_results[0].delivered_bytes,0)
        snap.metadata['sleep_modes']={}
        self.assertEqual(adapter.run_cluster_slot([phy('c0',[ue('u0',False)])],snap)['c0'].ue_results[0].delivered_bytes,0)
        self.assertEqual(adapter.trace['cells']['c0']['average_used_w'],0)

    def test_disconnected_scheduled_stream_emits_power_and_neighbor_interference(self):
        h=np.zeros((2,2,1,2));h[1,:,:,:]=1e-5
        snap=channel(h)
        adapter=self.adapter()
        plan=[phy('c0',[ue('u0')]),phy('c1',[ue('u1')])]
        active=adapter.run_cluster_slot(plan,snap)
        self.assertEqual(active['c0'].ue_results[0].delivered_bytes,0)
        self.assertAlmostEqual(adapter.trace['ues'][0]['precoder_column_norm_mean'],1)
        self.assertAlmostEqual(adapter.trace['cells']['c0']['average_used_w'],.001)
        plan[0].ue_inputs[0].is_scheduled=False
        idle=adapter.run_cluster_slot(plan,snap)
        self.assertGreater(idle['c1'].ue_results[0].sinr_db,active['c1'].ue_results[0].sinr_db+5)

    def test_mu_orthogonality_and_correlation(self):
        orth=channel(np.array([[[[1e-5,0]]],[[[0,1e-5]]]]),('u0','u1'),('c0',))
        corr=channel(np.array([[[[1e-5,0]]],[[[1e-5,0]]]]),('u0','u1'),('c0',))
        adapter=self.adapter(); inputs=[phy('c0',[ue('u0'),ue('u1')])]
        good=adapter.run_cluster_slot(inputs,orth)['c0'].ue_results
        bad=adapter.run_cluster_slot(inputs,corr)['c0'].ue_results
        self.assertGreater(good[0].sinr_db,bad[0].sinr_db+5)
        self.assertAlmostEqual(adapter.trace['ues'][0]['precoder_column_norm_mean'],1)
        solo=adapter.run_cluster_slot([phy('c0',[ue('u0'),ue('u1',False)])],orth)['c0'].ue_results[0]
        def rate(r):return np.log2(1+10**(r.sinr_db/10))
        self.assertGreater(sum(rate(r) for r in good),rate(solo))

    def test_association_selects_new_channel(self):
        snap=channel(np.array([[[[1e-6,0]],[[1e-4,0]]]]),('u0',),('c0','c1'))
        adapter=self.adapter()
        old=adapter.run_cluster_slot([phy('c0',[ue('u0')]),phy('c1',[])],snap)['c0'].ue_results[0]
        new=adapter.run_cluster_slot([phy('c0',[]),phy('c1',[ue('u0')])],snap)['c1'].ue_results[0]
        self.assertGreater(new.sinr_db,old.sinr_db+30)

    def test_multilayer_and_invalid_grants(self):
        snap=channel(np.eye(2)[None,None]*1e-5,('u0',),('c0',))
        adapter=self.adapter()
        result=adapter.run_cluster_slot([phy('c0',[ue('u0',layers=2)])],snap)['c0'].ue_results[0]
        self.assertEqual(result.num_layers,2);self.assertEqual(result.delivered_bytes,100)
        invalid=ue('u0');invalid.allocated_subcarrier_indices=[99]
        with self.assertRaises(ValueError):adapter.run_cluster_slot([phy('c0',[invalid])],snap)
        with self.assertRaises(ValueError):adapter.run_cluster_slot([phy('c0',[ue('u0')]),phy('c1',[ue('u0')])],snap)

    def test_masks_and_budget(self):
        adapter=self.adapter();snap=channel(np.ones((1,1,1,2))*1e-5,('u0',),('c0',))
        item=ue('u0');item.queue_bytes=7;item.grant_bytes_budget=13
        r=adapter.run_cluster_slot([phy('c0',[item])],snap)['c0'].ue_results[0]
        self.assertEqual(r.delivered_bytes,7);self.assertEqual(r.effective_data_re_count,18)
        self.assertLess(r.effective_data_re_count,r.resource_grid_re_count)
        np.testing.assert_allclose(adapter.run_cluster_slot([phy('c0',[item])],snap)['c0'].ue_results[0].sinr_db,r.sinr_db)

    def test_no_fallback_and_unknown_controls(self):
        with self.assertRaises(ValueError):RtSysPhyAdapter({'device':'numpy'})
        with self.assertRaises(RuntimeError):self.adapter().run_slot(phy('c0',[]))
        audit=audit_action(ConfigAction(power_cap_dbm=30,muting_ratio=.2))
        self.assertEqual({x['field']:x['route'] for x in audit},{'power_cap_dbm':'phy','muting_ratio':'unsupported'})


    def test_neighbor_power_and_beam_weights_change_physics(self):
        adapter=self.adapter()
        snap=channel(np.ones((2,2,1,4))*1e-5)
        a=adapter.run_cluster_slot([phy('c0',[ue('u0')]),phy('c1',[ue('u1')],0)],snap)['c0'].ue_results[0].sinr_db
        b=adapter.run_cluster_slot([phy('c0',[ue('u0')]),phy('c1',[ue('u1')],10)],snap)['c0'].ue_results[0].sinr_db
        self.assertLess(b,a-5)
        from phy.rt_sys.numerics import beam_subspace
        row=beam_subspace(4,0,0,0)[:,2].conj()[None,None,None,:]*1e-5
        snap=channel(row,('u0',),('c0',))
        wide=phy('c0',[ue('u0')]);narrow=phy('c0',[ue('u0')]);narrow.current_codebook_id=1
        a=adapter.run_cluster_slot([wide],snap)['c0'].ue_results[0].sinr_db
        b=adapter.run_cluster_slot([narrow],snap)['c0'].ue_results[0].sinr_db
        self.assertGreater(a,b+40)


    def test_cross_frequency_and_guard_band_rejected(self):
        adapter=self.adapter();snap=channel(np.ones((2,2,1,2))*1e-5)
        def context(fc):return SimpleNamespace(carrier_frequency_hz=fc,bandwidth_hz=10e6,
            subcarrier_spacing_hz=30000,n_prbs_total=1,num_ofdm_symbols_per_slot=2,
            cyclic_prefix_type='normal',slot_duration_ms=.5)
        first=phy('c0',[ue('u0')]);second=phy('c1',[ue('u1')])
        first.runtime_context=context(3.5e9);second.runtime_context=context(3.6e9)
        with self.assertRaisesRegex(ValueError,'common frequency'):
            adapter.run_cluster_slot([first,second],snap)
        first.ue_inputs[0].allocated_subcarrier_indices=[13]
        with self.assertRaisesRegex(ValueError,'active PRB'):
            adapter.run_cluster_slot([first],snap)


class FakeTracer:
    def __init__(self,m):self.m=m;self.calls=0
    def fingerprint(self):return {'fixture':True,'version':1}
    def validate_geometry(self):pass
    def generate(self,first,last,update_slots):
        self.calls+=1
        return {'cfr':np.ones((last-first,len(self.m['ues']),len(self.m['cells']),2,4,self.m['num_symbols'],self.m['fft_size']),np.complex64)*1e-5,
                'positions':np.asarray([[position_at(u['trajectory'],s*self.m['slot_duration_ms']/1000)[0] for u in self.m['ues']] for s in range(first,last)])}


class CacheTests(unittest.TestCase):
    def test_cache_replay_and_fingerprint(self):
        manifest,_,_=build_case(total_slots=2)
        with tempfile.TemporaryDirectory() as root:
            tracer=FakeTracer(manifest);a=RtChannelProvider(manifest,root,backend=tracer)
            first=a.snapshot(1)
            tracer2=FakeTracer(manifest);b=RtChannelProvider(manifest,root,backend=tracer2)
            np.testing.assert_equal(first.cfr,b.snapshot(1).cfr);self.assertEqual(tracer2.calls,0)
            changed=json.loads(json.dumps(manifest));changed['carrier_frequency_hz']+=100
            c=RtChannelProvider(changed,root,backend=FakeTracer(changed))
            self.assertNotEqual(a.key,c.key)

    def test_trajectory_and_keyed_random(self):
        traj={'times_s':[0,1],'positions_m':[[0,0,1.5],[10,0,1.5]]}
        p,v=position_at(traj,.5);np.testing.assert_equal(p,[5,0,1.5]);np.testing.assert_equal(v,[10,0,0])
        with self.assertRaises(ValueError):position_at(traj,2)
        a=keyed_uniform(1,2,'ue_4');keyed_uniform(1,2,'ue_1');self.assertEqual(a,keyed_uniform(1,2,'ue_4'))


class ClosedLoopTests(unittest.TestCase):
    def test_real_simulator_with_injected_channel(self):
        from core.simulator import Simulator
        from core.factory import build_controller_registry,build_scheduler
        from experiments.scenario_scan_utils import build_runtime_scenario_config
        from core.state_updater import StateUpdater
        manifest,configs,_=build_case(total_slots=3)
        configs['ric'].update(rapps=[],xapps=[])
        with tempfile.TemporaryDirectory() as root:
            provider=RtChannelProvider(manifest,root,backend=FakeTracer(manifest))
            adapter=RtSysPhyAdapter({**configs['phy']['rt_sys'],'device':'numpy'},abstraction=FixtureAbstraction(),channel_provider=provider)
            runtime=build_runtime_scenario_config(scenario_name='fixture',scenario_type='rt_scene',sim_config=configs['sim'],cell_config=configs['cell'],ric_config=configs['ric'])
            sim=Simulator(runtime,build_controller_registry(configs['ric']),build_scheduler(configs['sim']),adapter,
                          configs['cell'],configs['ue'],configs['beamforming'],root)
            for _ in range(3):sim.step()
            self.assertEqual(sim.state.slot_id,3);self.assertIn('rt_sys',sim.slot_records[-1])
            self.assertEqual(set(adapter.trace['cells']),{'cell_0','cell_1'})
            self.assertTrue(all(0<=u.queue_bytes for u in sim.state.ues))
            self.assertTrue(any(u['delivered_bytes']>0 for u in adapter.trace['ues']))


    def test_nack_pending_payload_matches_sys_tb_then_ack_clears(self):
        from core.simulator import Simulator
        from core.factory import build_controller_registry,build_scheduler
        from experiments.scenario_scan_utils import build_runtime_scenario_config
        class SmallTb(FixtureAbstraction):
            fail=True
            def evaluate(self,*args,**kwargs):
                result=super().evaluate(*args,**kwargs)
                result.update(capacity_bytes=7,ack=not self.fail,tbler=1.0 if self.fail else 0.0)
                return result
        manifest,configs,_=build_case(num_cells=1,num_ues=1,total_slots=2)
        configs['ric'].update(rapps=[],xapps=[])
        with tempfile.TemporaryDirectory() as root:
            provider=RtChannelProvider(manifest,root,backend=FakeTracer(manifest))
            model=SmallTb()
            adapter=RtSysPhyAdapter({'device':'numpy'},abstraction=model,channel_provider=provider)
            runtime=build_runtime_scenario_config(scenario_name='harq',scenario_type='rt_scene',sim_config=configs['sim'],cell_config=configs['cell'],ric_config=configs['ric'])
            sim=Simulator(runtime,build_controller_registry(configs['ric']),build_scheduler(configs['sim']),adapter,
                          configs['cell'],configs['ue'],configs['beamforming'],root)
            sim.step()
            first=adapter.trace['ues'][0]
            self.assertEqual(first['tb_bytes'],7)
            self.assertFalse(first['ack'])
            self.assertEqual(sim.state.ues[0].harq.pending_retx_bytes,7)
            self.assertEqual(sim.state.ues[0].queue_bytes,first['queue_bytes'])
            model.fail=False
            sim.step()
            second=adapter.trace['ues'][0]
            self.assertEqual(second['delivered_bytes'],7)
            self.assertEqual(sim.state.ues[0].harq.pending_retx_bytes,0)
            self.assertEqual(sim.state.ues[0].queue_bytes,second['queue_bytes']-7)


if __name__=='__main__':unittest.main()
