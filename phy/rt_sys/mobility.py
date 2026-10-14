"""Reuse handover state machine with delayed RT-derived candidate measurements."""
from collections import deque
from copy import copy
from types import SimpleNamespace
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
        cfg=adapter.config.get('mobility',{})
        self.filter_ms=float(cfg.get('measurement_filter_ms',20))
        self.load_filter_ms=float(cfg.get('load_filter_ms',50))
        self.ttt_slots=math.ceil(float(cfg.get('time_to_trigger_ms',40))/slot_duration_ms)
        self.dwell_slots=math.ceil(float(cfg.get('min_dwell_ms',100))/slot_duration_ms)
        if min(self.filter_ms,self.load_filter_ms,self.ttt_slots,self.dwell_slots)<0:
            raise ValueError('Mobility time constants must be nonnegative')
        self.filtered={}; self.filtered_load={}; self.trigger_since={}

    def _build_cell_load_view(self,state,profiles):
        view_state=copy(state)
        view_state.scheduler=SimpleNamespace(scheduled_ue_ids=sorted({uid
            for runtime in state.managed_cells.values() for uid in runtime.scheduler.scheduled_ue_ids}))
        view=super()._build_cell_load_view(view_state,profiles)
        alpha=1-math.exp(-self.slot_duration_s*1000/max(self.load_filter_ms,1e-12))
        for cid,load in view.items():
            previous=self.filtered_load.get(cid,load)
            for key in ('estimated_load','load_penalty_db','activity_factor'):
                load[key]=previous[key]+alpha*(load[key]-previous[key])
            self.filtered_load[cid]=dict(load)
        return view

    def _compute_measurement_event_met(self,**kwargs):
        base=super()._compute_measurement_event_met(**kwargs)
        state,ue,connection=self._event_context
        slot=int(state.slot_id); target=kwargs['best_candidate'].cell_id
        key=(ue.ue_id,target)
        serving_sleep=state.managed_cells[connection.serving_cell_id].cell.cell_sleep_mode=='deep_sleep'
        policy=state.managed_cells[connection.serving_cell_id].cell
        threshold=policy.handover_margin_db+policy.steering_hysteresis_db+policy.handover_cost_db
        last=connection.last_handover_slot
        dwell_ok=serving_sleep or last is None or slot-last>=self.dwell_slots
        if not base or kwargs['trigger_margin_db']<threshold or not dwell_ok:
            self.trigger_since.pop(key,None)
            return False
        started=self.trigger_since.setdefault(key,slot)
        return serving_sleep or slot-started>=self.ttt_slots

    def _advance_geometry(self,link):
        pass  # RtChannelProvider owns the preconfigured trajectory.

    def initialize_association(self, state):
        # Initial attachment is not a handover and must not wait for TTT.
        # Use actual RT reference gain (no noise/load bias) at snapshot zero.
        for ue in state.ues:
            connection = state.connections[ue.ue_id]
            candidates = connection.candidate_cells
            best = max(candidates, key=lambda c: c.tx_power_dbm - 10*math.log10(state.managed_cells[c.cell_id].resource_grid.n_prbs_total*12) - c.pathloss_db
                if state.managed_cells[c.cell_id].cell.cell_sleep_mode!='deep_sleep' else -math.inf) if candidates else None
            if best is None or best.pathloss_db >= 300:
                continue  # Explicit outage remains visible; do not invent a path.
            self._apply_serving_candidate(ue, state.links[ue.ue_id], connection, best)
            connection.connection_state = 'connected'
            connection.handover_pending_target_cell_id = None
            connection.handover_started_slot = None
            connection.last_trigger_reason = 'initial_rt_attachment'
        self.trigger_since.clear()

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
                raw=rsrp+noise
                previous=self.filtered.get((uid,cid),raw)
                alpha=1-math.exp(-self.slot_duration_s*1000/max(self.filter_ms,1e-12))
                filtered=raw if cell.cell_sleep_mode=='deep_sleep' else previous+alpha*(raw-previous)
                self.filtered[uid,cid]=filtered
                values[uid][cid]=(filtered,gain,best)
        self.history.append(values)
        self.observed=self.history[max(0,len(self.history)-1-self.delay)]
        self.observed_slot=int(snapshot.slot_id)-min(self.delay,len(self.history)-1)
        previous={uid:(c.serving_cell_id,c.handover_count) for uid,c in state.connections.items()}
        super().update(state,scoped_cell_policy=True)
        self.events=[]
        for uid,c in state.connections.items():
            if previous[uid][0]!=c.serving_cell_id:
                self.events.append({'ue_id':uid,'from_cell':previous[uid][0],'to_cell':c.serving_cell_id,
                    'handover_completed':c.handover_count>previous[uid][1],
                    'completed_handover_count':c.handover_count,'slot_id':state.slot_id})
                self.trigger_since={key:start for key,start in self.trigger_since.items() if key[0]!=uid}

    def _compute_candidate_cells(self,state,ue,link,connection,cell_load_view):
        self._event_context=(state,ue,connection)
        profiles=self._cell_radio_profiles(state)
        positions=self._managed_cells(state)
        serving=str(connection.serving_cell_id)
        policy=state.managed_cells[serving].cell
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
                preference=policy.traffic_steering_preference; bias=policy.mobility_bias_db
                score+=bias if preference=='edge_offload' else -abs(bias) if preference=='stay_put' else 0
            dx=link.x_m-position['x_m']; dy=link.y_m-position['y_m']; dz=link.z_m-position['z_m']
            candidates.append(CandidateCellState(measurement_produced_at_slot=self.observed_slot,
                measurement_delivered_at_slot=int(state.slot_id),measurement_window_slots=max(1,math.ceil(self.filter_ms/(self.slot_duration_s*1000))),cell_id=cid,x_m=position['x_m'],y_m=position['y_m'],z_m=position['z_m'],
                distance_2d_m=math.hypot(dx,dy),distance_3d_m=math.sqrt(dx*dx+dy*dy+dz*dz),
                carrier_frequency_hz=profile['carrier_frequency_hz'],bandwidth_hz=profile['bandwidth_hz'],
                subcarrier_spacing_hz=profile['subcarrier_spacing_hz'],tx_power_dbm=runtime.cell.tx_power_dbm,
                pathloss_db=-10*math.log10(gain),shadowing_db=0,rsrp_dbm=rsrp,
                estimated_interference_db=denominator,signal_margin_db=rsrp-denominator,
                quality_score_db=score,steering_score_db=score,candidate_beam_id=beam,
                is_serving=cid==serving,estimated_load=load['estimated_load'],load_penalty_db=load['load_penalty_db'],
                connected_ue_count=int(load['connected_ue_count']),scheduled_ue_count=int(load['scheduled_ue_count']),
                queued_bytes=int(load['queued_bytes'])))
        ranked=sorted(candidates,key=lambda c:c.quality_score_db,reverse=True)
        # TTT is continuous for the current best target. Returning to serving
        # or changing target breaks all previous candidate timers.
        best=ranked[0].cell_id
        self.trigger_since={key:start for key,start in self.trigger_since.items()
            if key[0]!=ue.ue_id or (key[1]==best and best!=serving)}
        return ranked
