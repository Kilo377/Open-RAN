"""Reference measurement, mobility and telemetry for opt-in paper interfaces.

No learned policy, synthetic conflict labels, or data-CRC-driven RLF. All timing
and thresholds below are engineering assumptions, not an RRC implementation.
"""
from collections import deque, Counter
from copy import deepcopy
import math
import random
import numpy as np
from phy.rt_sys.contracts import digest
from phy.rt_sys.paper_controls import DEFAULTS, COMMANDS
from phy.rt_sys.paper_radio import references, quantify_cqi
from state.candidate_cell_state import CandidateCellState


class PaperEnvironment:
    def __init__(self,simulator,config):
        self.sim=simulator;self.state=simulator.state;self.adapter=simulator.phy_adapter
        self.config=deepcopy(config);self.adapter.paper=self
        self.dt=float(simulator.config.slot_duration_ms);self.window=int(getattr(simulator.config,'measurement_throughput_granularity_slots',10))
        self.thresholds=config.get('cqi_thresholds_db',list(range(-7,22,2)))
        quantify_cqi(0,self.thresholds)
        self.events_enabled=bool(config.get('connection_events',True))
        self.filtered={};self.sample_slot=-1;self.committed=-1;self.raw={};self.observed={};self.history=deque(maxlen=4)
        self.triggers={};self.pending={};self.manual=set();self.bad_since={};self.good_since={}
        self.sessions={};self.session_serial=0;self.session_events=[];self.all_events=[]
        self.events=[];self.command_ids=set();self.stats=Counter();self.cell_stats={c:Counter() for c in self.state.managed_cells}
        self.receipts=[];self.samples=deque(maxlen=self.window);self.ue_samples={u.ue_id:deque(maxlen=self.window) for u in self.state.ues}
        self.previous={u.ue_id:dict(u.packet_ledger.stats) for u in self.state.ues}
        self.rf_energy={c:0. for c in self.state.managed_cells};self.published={};self.sla_since={}
        self.delayed=deque(maxlen=max(4,int(getattr(simulator.config,'measurement_queue_delay_slots',0))+3))
        self.last_radio_config=None
        for ue,item in zip(self.state.ues,getattr(simulator,'_paper_ue_config',{}).get('ues',[])):
            value=item.get('requested_dl_rate_bps',self.config.get('service_requirements_bps',{}).get(ue.ue_id,self.config.get('service_requirements_by_slice_bps',{}).get(ue.slice_id)))
            if value is not None and (not np.isfinite(value) or value<0):raise ValueError('Invalid declared service requirement')
            ue.paper_requested_dl_rate_bps=None if value is None else float(value)
        for runtime in self.state.managed_cells.values():
            for k,v in DEFAULTS.items():setattr(runtime.cell,k,deepcopy(v))
            runtime.cell.time_to_trigger_ms=float(self.adapter.config.get('mobility',{}).get('time_to_trigger_ms',40))
            if not np.isfinite(runtime.cell.time_to_trigger_ms) or runtime.cell.time_to_trigger_ms<0:raise ValueError('Invalid initial TTT')
        self.layer_caps={cid:(r.scheduler.max_total_spatial_layers or self.sim.scheduler.max_total_spatial_layers) for cid,r in self.state.managed_cells.items()}
        self.model=dict(bad_rsrp_dbm=-110.,bad_sinr_db=-6.,good_rsrp_dbm=-107.,good_sinr_db=-3.,bad_ms=200.,good_ms=40.,handover_timeout_ms=200.)
        self.model.update(config.get('connection_model',{}))
        if any(not np.isfinite(v) for v in self.model.values()) or min(self.model[k] for k in ('bad_ms','good_ms','handover_timeout_ms'))<0:
            raise ValueError('Invalid connection model timing/threshold')

    def event(self,kind,uid,**kwargs):
        e=dict(kind=kind,slot_id=int(self.state.slot_id),ue_id=uid,**kwargs)
        self.events.append(e);self.all_events.append(e)
        return e

    def reset_triggers(self,cid):
        self.triggers={k:v for k,v in self.triggers.items() if self.state.connections[k[0]].serving_cell_id!=cid}
        for uid,p in list(self.pending.items()):
            if p['from_cell']==cid and p['to_cell'] not in self.neighbors(cid):self.finish_handover(uid,'cancelled','neighbor_removed')

    def neighbors(self,cid):
        nl=self.state.managed_cells[cid].cell.neighbor_cell_ids
        return set(self.state.managed_cells)-{cid} if nl is None else set(nl)

    def prepare_reference(self):
        slot=self.state.slot_id
        if self.sample_slot!=slot:
            self.events=[];self.receipts=[];self.sample_slot=slot
        # Mobility and scheduling ask for the same hypothesis within a slot.
        # Runtime RF changes invalidate it; association/queue changes do not
        # enter references(), which calculates every UE/candidate independently.
        snapshot=self.adapter.current_snapshot
        radio=tuple((cid,r.resource_grid.n_prbs_total,
            r.cell.tx_power_dbm,r.cell.cell_sleep_mode,r.cell.current_codebook_id,
            r.cell.current_beam_profile_id,getattr(r.cell,'ret_enabled',False),
            getattr(r.cell,'ret_tilt_deg',0),getattr(r.cell,'ret_codebook_id',0),
            tuple(r.cell.prb_power_w) if getattr(r.cell,'prb_power_w',None) is not None else None)
            for cid,r in self.state.managed_cells.items())
        key=(slot,id(snapshot),radio,snapshot.metadata.get('tx_rows'),snapshot.metadata.get('tx_cols'),
             self.adapter.config.get('noise_figure_db',7))
        if self.adapter.config.get('reuse_reference_within_slot',False) and getattr(self,'_reference_key',None)==key:return
        self.raw=references(self.adapter.current_snapshot,self.state,self.adapter.config.get('noise_figure_db',7))
        self._reference_key=key

    def serving_references(self):
        if not self.raw:self.prepare_reference()
        source=self.observed if self.committed==self.state.slot_id and self.observed else self.raw
        return {uid:dict(source[uid][c.serving_cell_id],serving_cell_id=c.serving_cell_id,reference_beam_gain_db=None,prefiltered_reference=True) for uid,c in self.state.connections.items()}

    def good(self,r):
        return bool(r and not r['no_path'] and r['rsrp_dbm']>self.model['good_rsrp_dbm'] and r['reference_sinr_db']>self.model['good_sinr_db'])

    def bad(self,r):
        return bool(r and (r['no_path'] or r['rsrp_dbm']<self.model['bad_rsrp_dbm'] or r['reference_sinr_db']<self.model['bad_sinr_db']))

    def cleanup(self,uid,reason):
        ue=next(u for u in self.state.ues if u.ue_id==uid)
        self.reset_good_timer(uid)
        self.triggers={k:v for k,v in self.triggers.items() if k[0]!=uid}
        for p in ue.harq.processes:
            self.sim.scheduler._record_harq_process_flush(ue.harq,reason=reason,slot_id=self.state.slot_id,process=p)
            ue.packet_ledger.release(p.tb_id)
        ue.harq.processes.clear();ue.harq.pending_retx_bytes=0;ue.harq.pending_retx_logical_channels=[]
        self.state.connections[uid].active_harq_processes=[]
        receiver=getattr(self.adapter.abstraction,'harq',None)
        if receiver:
            for key in list(receiver.pending):
                if key[0]==uid:receiver.pending.pop(key)

    def open_session(self,uid,cid):
        self.session_serial+=1;sid=f'{uid}:connection:{self.session_serial}'
        self.sessions[uid]=dict(session_id=sid,start_slot=self.state.slot_id,cell_id=cid)
        self.session_events.append(dict(kind='start',slot=self.state.slot_id,uid=uid,session_id=sid))
        self.event('connection_started',uid,session_id=sid,cell_id=cid)

    def close_session(self,uid,kind):
        s=self.sessions.pop(uid,None)
        if s:
            self.session_events.append(dict(kind=kind,slot=self.state.slot_id,uid=uid,session_id=s['session_id']))
            self.event('connection_'+kind,uid,session_id=s['session_id'],cell_id=self.state.connections[uid].serving_cell_id)

    def reset_good_timer(self,uid):
        self.good_since={k:v for k,v in self.good_since.items() if k!=uid and not (isinstance(k,tuple) and k[0]==uid)}

    def attach(self,uid,cid,reason):
        c=self.state.connections[uid];ue=next(u for u in self.state.ues if u.ue_id==uid)
        candidate=next(x for x in c.candidate_cells if x.cell_id==cid)
        self.sim.mobility_manager._apply_serving_candidate(ue,self.state.links[uid],c,candidate)
        c.connection_state='connected';c.out_of_sync_reason=None;c.consecutive_nack_count=0
        if uid not in self.sessions:self.open_session(uid,cid)
        self.bad_since.pop(uid,None);self.reset_good_timer(uid)

    def start_handover(self,uid,target,manual):
        c=self.state.connections[uid]
        self.pending[uid]=dict(from_cell=c.serving_cell_id,to_cell=target,start=self.state.slot_id,good_since=None,manual=manual)
        c.handover_pending_target_cell_id=target;c.handover_started_slot=self.state.slot_id
        # Keep serving the original cell while target preparation is pending.
        self.event('handover_attempt',uid,from_cell=c.serving_cell_id,to_cell=target)

    def finish_handover(self,uid,outcome,reason):
        p=self.pending.pop(uid,None)
        if p is None:return
        c=self.state.connections[uid];c.handover_pending_target_cell_id=None;c.handover_started_slot=None
        if outcome=='success':
            self.cleanup(uid,'handover');self.attach(uid,p['to_cell'],'handover')
            c.handover_count+=1;c.last_handover_slot=self.state.slot_id
            interruption=int(self.state.managed_cells[p['from_cell']].cell.handover_interruption_slots)
            c.interruption_remaining_slots=interruption
            if interruption:c.connection_state='handover_interruption'
        self.event('handover_'+outcome,uid,from_cell=p['from_cell'],to_cell=p['to_cell'],reason=reason)
        self.cell_stats[p['from_cell']]['handover_'+outcome]+=1
        self.triggers={k:v for k,v in self.triggers.items() if k[0]!=uid}

    def consume_commands(self,commands):
        owned=[c for c in commands if c.command_type in COMMANDS];other=[c for c in commands if c.command_type not in COMMANDS]
        def order(c):
            m=c.metadata
            return (-int(m.priority or 0),-int(m.issued_at_slot or 0),str(c.source_name),str(c.control_request_id or m.decision_id or ''))
        chosen=set()
        for cmd in sorted(owned,key=order):
            uid=cmd.target_ue_id;rid=cmd.control_request_id or cmd.metadata.decision_id;reason=None
            if not rid:reason='missing_request_id'
            elif rid in self.command_ids:reason='duplicate_request'
            elif uid not in self.state.connections:reason='unknown_ue'
            elif cmd.valid_from_slot is not None and self.state.slot_id<cmd.valid_from_slot:reason='not_yet_valid'
            elif cmd.valid_until_slot is not None and self.state.slot_id>cmd.valid_until_slot:reason='expired'
            elif uid in chosen:reason='conflicting_ue_command'
            else:
                c=self.state.connections[uid];target=cmd.target_cell_id
                allowed=getattr(cmd,'_paper_allowed_cells',())
                if allowed and (c.serving_cell_id not in allowed or (target is not None and target not in allowed)):reason='command_execution_scope'
                if reason is None and cmd.command_type=='request_handover':
                    if uid in self.manual or c.connection_state!='connected':reason='ue_not_connected'
                    elif uid in self.pending:reason='handover_in_progress'
                    elif c.last_handover_slot is not None and (self.state.slot_id-c.last_handover_slot)*self.dt<float(self.adapter.config.get('mobility',{}).get('min_dwell_ms',100)):reason='minimum_dwell_time'
                    elif target not in self.neighbors(c.serving_cell_id):reason='target_not_eligible'
                    elif not self.good(self.raw[uid].get(target)):reason='target_not_available'
                elif cmd.command_type=='reconnect' and (uid in self.sessions and uid not in self.manual):reason='already_connected'
                elif cmd.command_type=='reconnect' and target is not None and target not in self.state.managed_cells:reason='unknown_target_cell'
            if rid:self.command_ids.add(rid)
            if reason is None:
                chosen.add(uid)
                if cmd.command_type=='disconnect':
                    self.finish_handover(uid,'cancelled','manual_disconnect');self.cleanup(uid,'manual_disconnect')
                    self.close_session(uid,'normal_end');self.manual.add(uid);self.state.connections[uid].connection_state='disconnected'
                elif cmd.command_type=='reconnect':
                    self.manual.discard(uid);self.state.connections[uid].connection_state='out_of_sync'
                    self.reset_good_timer(uid)
                    self.state.connections[uid]._paper_reconnect_target=cmd.target_cell_id
                else:self.start_handover(uid,cmd.target_cell_id,True)
            receipt=dict(kind='command',control_request_id=rid,source=cmd.source_name,target_ue_id=uid,target_cell_id=cmd.target_cell_id,
                command_type=cmd.command_type,apply_slot=self.state.slot_id,status='rejected' if reason else 'applied',reason=reason)
            self.receipts.append(receipt)
            for old in self.state.control.actuation_outcomes:
                if old.get('kind')=='command' and old.get('control_request_id')==rid:old.update(receipt)
            self.sim._update_e2_control_request_status(rid,status=receipt['status'],reason=reason or '')
        return other

    def before_scheduling(self):
        self.prepare_reference()
        slot=self.state.slot_id
        if self.committed==slot:return
        self.committed=slot
        # One common filtered/noisy reference report for all candidates/subbands.
        alpha=1-math.exp(-self.dt/max(float(self.adapter.config.get('mobility',{}).get('measurement_filter_ms',20)),1e-9))
        for uid,cells in self.raw.items():
            for cid,r in cells.items():
                rng=random.Random(int(digest([self.adapter.seed,slot,uid,cid,'paper_reference_noise'])[:15],16))
                noise=rng.gauss(0,float(getattr(self.sim.config,'measurement_sinr_noise_std_db',2)))
                old=self.filtered.get((uid,cid));new=deepcopy(r)
                for k in ('rsrp_dbm','reference_sinr_db'):
                    value=r[k]+noise;new[k]=value if old is None else old[k]+alpha*(value-old[k])
                value=np.asarray(r['prb_sinr_db'])+noise
                new['prb_sinr_db']=(value if old is None else np.asarray(old['prb_sinr_db'])+alpha*(value-np.asarray(old['prb_sinr_db']))).tolist()
                self.filtered[uid,cid]=new
        self.observed={uid:{cid:self.filtered[uid,cid] for cid in cells} for uid,cells in self.raw.items()}
        for ue in self.state.ues:
            uid=ue.ue_id;c=self.state.connections[uid];source=c.serving_cell_id;policy=self.state.managed_cells[source].cell
            c.candidate_cells=[]
            preference=policy.traffic_steering_preference
            for cid,r in self.observed[uid].items():
                c.candidate_cells.append(CandidateCellState(cell_id=cid,rsrp_dbm=r['rsrp_dbm'],quality_score_db=r['reference_sinr_db'],
                    steering_score_db=r['rsrp_dbm']+float(policy.cio_offsets_db.get(cid,0))+(float(policy.mobility_bias_db) if preference=='edge_offload' and cid!=source else -abs(float(policy.mobility_bias_db)) if preference=='stay_put' and cid!=source else 0),candidate_beam_id=r['candidate_beam_id'],
                    is_serving=cid==source,measurement_produced_at_slot=slot,measurement_delivered_at_slot=slot,measurement_window_slots=max(1,round(20/self.dt))))
            if slot==0:
                target=max(self.observed[uid],key=lambda cid:self.observed[uid][cid]['rsrp_dbm'])
                if not self.events_enabled or self.good(self.observed[uid][target]):self.attach(uid,target,'initial')
                else:c.connection_state='out_of_sync'
                continue
            if uid in self.manual:continue
            r=self.observed[uid].get(c.serving_cell_id)
            if self.events_enabled and self.bad(r):self.bad_since.setdefault(uid,slot)
            else:self.bad_since.pop(uid,None)
            if uid in self.sessions and uid in self.bad_since and (slot-self.bad_since[uid])*self.dt>=self.model['bad_ms']:
                self.finish_handover(uid,'failure','serving_link_lost');self.cleanup(uid,'radio_link_failure');self.close_session(uid,'drop')
                c.connection_state='out_of_sync';c.out_of_sync_reason='reference_link_failure'
            if uid not in self.sessions:
                candidates=[cid for cid,v in self.observed[uid].items() if self.good(v)]
                target=getattr(c,'_paper_reconnect_target',None)
                if target is not None:candidates=[cid for cid in candidates if cid==target]
                if candidates:
                    best=max(candidates,key=lambda cid:self.observed[uid][cid]['rsrp_dbm']);key=(uid,best)
                    start=self.good_since.setdefault(key,slot)
                    self.good_since={k:v for k,v in self.good_since.items() if not isinstance(k,tuple) or k[0]!=uid or k==key}
                    if (slot-start)*self.dt>=self.model['good_ms']:
                        self.attach(uid,best,'recovery');c._paper_reconnect_target=None
                else:self.good_since={k:v for k,v in self.good_since.items() if not isinstance(k,tuple) or k[0]!=uid}
                continue
            if c.connection_state=='handover_interruption':
                c.interruption_remaining_slots=max(0,c.interruption_remaining_slots-1)
                if c.interruption_remaining_slots==0:c.connection_state='connected'
                continue
            if uid in self.pending:
                p=self.pending[uid];target=p['to_cell'];target_good=self.good(self.observed[uid].get(target))
                if not target_good:self.finish_handover(uid,'failure','target_unavailable');continue
                if p['good_since'] is None:p['good_since']=slot
                if (slot-p['start'])*self.dt>=self.model['handover_timeout_ms']:self.finish_handover(uid,'failure','timeout');continue
                delay=max(float(self.model['good_ms']),float(getattr(policy,'handover_delay_slots',2))*self.dt)
                if (slot-p['good_since'])*self.dt>=delay:self.finish_handover(uid,'success','target_confirmed')
                continue
            last=c.last_handover_slot
            if last is not None and (slot-last)*self.dt<float(self.adapter.config.get('mobility',{}).get('min_dwell_ms',100)):continue
            choices=[x for x in c.candidate_cells if x.cell_id in self.neighbors(c.serving_cell_id)]
            if not choices:continue
            best=max(choices,key=lambda x:x.steering_score_db)
            margin=best.steering_score_db-self.observed[uid][c.serving_cell_id]['rsrp_dbm']
            threshold=float(policy.steering_hysteresis_db)+float(policy.handover_cost_db)+float(getattr(policy,'handover_margin_db',3))
            current=next(x for x in c.candidate_cells if x.cell_id==c.serving_cell_id)
            current.signal_margin_db=r['reference_sinr_db'];best.signal_margin_db=self.observed[uid][best.cell_id]['reference_sinr_db']
            from core.mobility_manager import MobilityManager
            event_met=MobilityManager._compute_measurement_event_met(self.sim.mobility_manager,event_trigger=policy.measurement_event_trigger,
                trigger_margin_db=margin,event_offset_db=policy.measurement_event_offset_db,best_candidate=best,current_candidate=current)
            key=(uid,best.cell_id)
            if event_met and margin>=threshold and self.good(self.observed[uid][best.cell_id]):
                self.triggers={k:v for k,v in self.triggers.items() if k[0]!=uid or k==key}
                start=self.triggers.setdefault(key,slot)
                if (slot-start)*self.dt>=policy.time_to_trigger_ms:self.start_handover(uid,best.cell_id,False)
            else:self.triggers={k:v for k,v in self.triggers.items() if k[0]!=uid}

    def after_phy(self):
        slot=self.state.slot_id;dt=self.dt/1000;cell_bytes={c:0 for c in self.state.managed_cells}
        for ue in self.state.ues:
            stats=dict(ue.packet_ledger.stats);prev=self.previous[ue.ue_id]
            delivered=stats['delivered_bytes']-prev.get('delivered_bytes',0);arrived=stats['offered_bytes']-prev.get('offered_bytes',0)
            self.ue_samples[ue.ue_id].append((delivered,arrived));self.previous[ue.ue_id]=stats
            cell_bytes[self.state.connections[ue.ue_id].serving_cell_id]+=delivered
        powers={c:float(t['average_used_w']) for c,t in self.adapter.trace['cells'].items()}
        for c,p in powers.items():self.rf_energy[c]+=p*dt
        self.samples.append(dict(slot=slot,energy={c:p*dt for c,p in powers.items()},bits={c:b*8 for c,b in cell_bytes.items()}))
        self.published=self.build_reports()
        self.delayed.append((slot,deepcopy(self.published)))
        self.adapter.trace['paper_interfaces']=dict(events=deepcopy(self.events),command_receipts=deepcopy(self.receipts),
            config_outcomes=deepcopy(self.state.control.actuation_outcomes),rf_energy_j=dict(self.rf_energy),
            runtime_config={cid:{k:deepcopy(getattr(r.cell,k)) for k in DEFAULTS} for cid,r in self.state.managed_cells.items()},
            reports=deepcopy(self.published))

    def build_reports(self):
        slot=self.state.slot_id;dt=self.dt/1000;reports={'ue':{},'cell':{}};window_start=self.samples[0]['slot'] if self.samples else slot
        for ue in self.state.ues:
            uid=ue.ue_id;c=self.state.connections[uid];seq=self.ue_samples[uid];seconds=len(seq)*dt
            rate=sum(x[0] for x in seq)*8/max(seconds,dt);arrival=sum(x[1] for x in seq)*8/max(seconds,dt)
            demand=getattr(ue,'paper_requested_dl_rate_bps',None)
            unmet=demand is not None and rate<demand if seconds else False
            if unmet:self.sla_since.setdefault(uid,slot)
            else:self.sla_since.pop(uid,None)
            r=self.observed[uid][c.serving_cell_id]
            active_session_ids={e['session_id'] for e in self.session_events if e['uid']==uid and e['kind']=='start' and e['slot']<=slot}
            ended_before={e['session_id'] for e in self.session_events if e['uid']==uid and e['kind']!='start' and e['slot']<window_start}
            denominator=len(active_session_ids-ended_before)
            e=[x for x in self.all_events if x['ue_id']==uid and x['slot_id']>=window_start]
            success=sum(x['kind']=='handover_success' for x in e);failure=sum(x['kind']=='handover_failure' for x in e)
            drops=sum(x['kind']=='connection_drop' for x in e)
            trace=next((x for x in self.adapter.trace['ues'] if x['ue_id']==uid),{})
            alloc=deepcopy(trace.get('resource_allocation',{}))
            group=int(self.state.managed_cells[c.serving_cell_id].cell.power_group_size_prbs)
            subbands=[list(range(i,min(i+group,len(r['prb_sinr_db'])))) for i in range(0,len(r['prb_sinr_db']),group)]
            group_cqi=[quantify_cqi(10*math.log10(float(np.mean(10**(np.asarray(r['prb_sinr_db'])[indices]/10)))),self.thresholds) for indices in subbands]
            reports['ue'][uid]=dict(subband_prb_groups=subbands,reference_sinr_db=r['reference_sinr_db'],rsrp_dbm=r['rsrp_dbm'],association=dict(cell_id=c.serving_cell_id,connection_state=c.connection_state),
                resource_allocation=alloc,wideband_cqi=quantify_cqi(r['reference_sinr_db'],self.thresholds) if not r['no_path'] else None,
                subband_cqi=group_cqi if not r['no_path'] else None,
                candidate_sinr_db={cid:x['reference_sinr_db'] for cid,x in self.observed[uid].items()},
                candidate_rsrp_dbm={cid:x['rsrp_dbm'] for cid,x in self.observed[uid].items()},goodput_utility=math.log(max(rate,1e-6)),
                offered_rate_bps=arrival,requested_dl_rate_bps=demand,sla_status=dict(target_bps=demand,goodput_bps=rate,violated=unmet,duration_ms=(slot-self.sla_since[uid]+1)*self.dt if uid in self.sla_since else 0) if demand is not None else None,
                handover_rate_per_s=success/max(seconds,dt),handover_failure_rate=failure/(success+failure) if success+failure and self.events_enabled else None,
                connection_drop_fraction=drops/denominator if denominator and self.events_enabled else None,
                mobility_counts=dict(success=success,failure=failure,cancelled=sum(x['kind']=='handover_cancelled' for x in e),pending=int(uid in self.pending),drops=drops,sessions_at_risk=denominator,normal_ends=sum(x['kind']=='connection_normal_end' for x in e),active_sessions=int(uid in self.sessions)))
        for cid in self.state.managed_cells:
            energy=sum(s['energy'][cid] for s in self.samples);bits=sum(s['bits'][cid] for s in self.samples)
            reports['cell'][cid]=dict(rf_energy_j=energy,rf_energy_total_j=self.rf_energy[cid],rf_efficiency_bit_per_j=bits/energy if energy>0 else None,whole_device_power_w=None,whole_device_efficiency_bit_per_j=None)
        return reports

    def attach_observation(self,observation):
        # This object goes through the existing E2 observation queue. Independent
        # measurement-report delay is applied here before E2 delivery stamping.
        observation.paper_interfaces_enabled=True
        delay=int(getattr(self.sim.config,'measurement_queue_delay_slots',0));slot=self.state.slot_id
        candidates=[(s,r) for s,r in self.delayed if s<=slot-delay]
        if not candidates:
            observation.paper_report=None;return
        produced,report=candidates[-1]
        observation.paper_report=dict(values=deepcopy(report),produced_at_slot=produced,window_slots=min(self.window,produced),
            source='RT_reference_quantization_and_executed_MAC_RF_event_counters',reference_window_slots=max(1,math.ceil(float(self.adapter.config.get('mobility',{}).get('measurement_filter_ms',20))/self.dt)),cqi_thresholds_db=list(self.thresholds),
            connection_model=deepcopy(self.model),connection_model_enabled=self.events_enabled,rf_energy_scope='emitted modeled data/pilots only; reference transmission is a measurement hypothesis')
