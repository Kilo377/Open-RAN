"""Reuse handover state machine with delayed RT-derived candidate measurements."""
from collections import deque
import math
import numpy as np
from core.mobility_manager import MobilityManager
from state.candidate_cell_state import CandidateCellState
from phy.rt_sys.numerics import beam_subspace
from phy.rt_sys.contracts import keyed_uniform


class RtMobilityManager(MobilityManager):
    def __init__(self,adapter,*,slot_duration_ms,rng_seed,delay_slots,noise_std_db=2):
        super().__init__(slot_duration_ms=slot_duration_ms,rng_seed=rng_seed)
        self.adapter=adapter; self.delay=max(0,int(delay_slots)); self.noise=noise_std_db
        self.history=deque(maxlen=self.delay+2); self.observed={}

    def _advance_geometry(self,link):
        pass  # RtChannelProvider owns the preconfigured trajectory.

    def update(self,state):
        snapshot=self.adapter.current_snapshot
        values={}
        for uid in snapshot.ue_ids:
            values[uid]={}
            for ci,cid in enumerate(snapshot.cell_ids):
                runtime=state.managed_cells[cid]
                cell=runtime.cell
                channel=snapshot.link(uid,cid)
                nt=channel.shape[1]
                gains=[]
                for beam in range(nt):
                    vector=beam_subspace(nt,1,1,beam)[:,0]
                    signal=np.einsum('rtsf,t->rsf',channel,vector)
                    gains.append(float(np.mean(np.sum(abs(signal)**2,axis=0))))
                best=int(np.argmax(gains)); gain=max(gains[best],1e-30)
                # RSRP here is reference received power per occupied subcarrier,
                # before load-dependent instantaneous data interference.
                nsc=int(runtime.resource_grid.n_prbs_total)*12
                rsrp=float(cell.tx_power_dbm)-10*math.log10(nsc)+10*math.log10(gain)
                if cell.cell_sleep_mode=='deep_sleep': rsrp=-300.0
                u1=max(keyed_uniform(self.rng_seed,snapshot.slot_id,uid,'measurement_'+cid+'_1'),1e-12)
                u2=keyed_uniform(self.rng_seed,snapshot.slot_id,uid,'measurement_'+cid+'_2')
                noise=self.noise*math.sqrt(-2*math.log(u1))*math.cos(2*math.pi*u2)
                values[uid][cid]=(rsrp+noise,gain,best)
        self.history.append(values)
        self.observed=self.history[max(0,len(self.history)-1-self.delay)]
        super().update(state)

    def _compute_candidate_cells(self,state,ue,link,connection,cell_load_view):
        profiles=self._cell_radio_profiles(state)
        positions=self._managed_cells(state)
        serving=str(connection.serving_cell_id)
        values=self.observed[ue.ue_id]
        candidates=[]
        for cid,(rsrp,gain,beam) in values.items():
            profile=profiles[cid]; position=positions[cid]
            runtime=state.managed_cells[cid]
            load=cell_load_view[cid]
            noise=-174+10*math.log10(profile['subcarrier_spacing_hz'])+7
            interference=sum(10**(other[0]/10)*cell_load_view[oid]['activity_factor']
                             for oid,other in values.items() if oid!=cid)
            denominator=10*math.log10(10**(noise/10)+interference)
            score=rsrp-denominator-load['load_penalty_db']
            if cid!=serving:
                preference=state.cell.traffic_steering_preference; bias=state.cell.mobility_bias_db
                score+=bias if preference=='edge_offload' else -abs(bias) if preference=='stay_put' else 0
            dx=link.x_m-position['x_m']; dy=link.y_m-position['y_m']; dz=link.z_m-position['z_m']
            candidates.append(CandidateCellState(cell_id=cid,x_m=position['x_m'],y_m=position['y_m'],z_m=position['z_m'],
                distance_2d_m=math.hypot(dx,dy),distance_3d_m=math.sqrt(dx*dx+dy*dy+dz*dz),
                carrier_frequency_hz=profile['carrier_frequency_hz'],bandwidth_hz=profile['bandwidth_hz'],
                subcarrier_spacing_hz=profile['subcarrier_spacing_hz'],tx_power_dbm=runtime.cell.tx_power_dbm,
                pathloss_db=-10*math.log10(gain),shadowing_db=0,rsrp_dbm=rsrp,
                estimated_interference_db=denominator,signal_margin_db=rsrp-denominator,
                quality_score_db=score,steering_score_db=score,candidate_beam_id=beam,
                is_serving=cid==serving,estimated_load=load['estimated_load'],load_penalty_db=load['load_penalty_db'],
                connected_ue_count=int(load['connected_ue_count']),scheduled_ue_count=int(load['scheduled_ue_count']),
                queued_bytes=int(load['queued_bytes'])))
        return sorted(candidates,key=lambda c:c.quality_score_db,reverse=True)
