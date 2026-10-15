"""Offline-facing evidence only. Never supplies information to a controller."""
from collections import Counter
from dataclasses import asdict
import math
from phy.rt_sys.reference import probability_interval


def compact_trace(trace):
    """Keep executed TB/resource/byte contracts; omit redundant PHY diagnostics."""
    dropped={'precoders','control_audit','paper_interfaces'}
    result={k:v for k,v in trace.items() if k not in dropped}
    result['recording_mode']='experiment'
    return result


class RuntimeDiagnostics:
    def __init__(self, users, dt, warmup_ms, window_ms=100):
        self.dt=dt;self.warmup=warmup_ms;self.window_slots=max(1,round(window_ms/dt))
        self.users={u.ue_id:u for u in users};self.windows={};self.ue={uid:Counter() for uid in self.users}
        self.last={};self.evaluated_slots=0;self.feedback_count=0

    def observe(self,state,trace):
        slot=state.slot_id;feedback=[];selections=[]
        for uid,u in self.users.items():
            olla=getattr(u,'du_olla',None)
            if olla:
                for event in olla.consumed_events:
                    if event['consumed_slot']!=slot:continue
                    assert event['available_slot']<=slot, 'Future feedback consumed'
                    assert not event['is_retx'] or not event['updated'], 'Retx updated OLLA'
                    feedback.append(dict(ue_id=uid,**event));self.feedback_count+=1
            decision=getattr(u,'olla_selection',None)
            if decision and decision['slot_id']==slot:
                selections.append(dict(ue_id=uid,segment=olla.segment,**decision))
        if state.sim_time_ms<=self.warmup:return feedback,selections
        self.evaluated_slots+=1
        paper=getattr(state,'paper',None)
        for row in trace['ues']:
            uid=row['ue_id'];u=self.users[uid];connection=state.connections[uid];counts=self.ue[uid]
            counts['evaluated_slots']+=1
            connected=connection.connection_state in ('connected','handover_interruption')
            counts['connected_slots']+=connected
            counts['scheduled_slots']+=bool(row['scheduled'])
            counts['allocated_re_streams']+=row['allocated_re']*row['layers']
            counts['actual_decodes']+=bool(row.get('actual_decode'))
            counts['delivered_bytes']+=row['delivered_bytes']
            refs=paper.raw.get(uid,{}) if paper else {}
            best=max(refs,key=lambda c:refs[c]['rsrp_dbm']) if refs else None
            serving=refs.get(connection.serving_cell_id,{})
            counts['serving_no_path_slots']+=bool(serving.get('no_path',False))
            counts['all_candidates_no_path_slots']+=bool(refs) and all(v['no_path'] for v in refs.values())
            counts['best_candidate_below_recovery_slots']+=bool(best) and not paper.good(refs[best])
            counts['retx_wait_events']+=sum(e.get('kind')=='retx_deferred_resource' for e in u.harq.events)
            self.last[uid]=dict(serving_cell_id=connection.serving_cell_id,connection_state=connection.connection_state,
                serving_reference=serving,best_candidate_cell_id=best,best_candidate_reference=refs.get(best),
                ue_position_m=[state.links[uid].x_m,state.links[uid].y_m,state.links[uid].z_m],
                pending_bytes=u.packet_ledger.stats['pending_bytes'],slice_id=u.slice_id,
                traffic_type=u.traffic_type,latency_budget_ms=u.latency_budget_ms)
            if not row.get('actual_decode'):continue
            assert row.get('tb_id'), 'Actual transmission without TB identity'
            decision=getattr(u,'olla_selection',{})
            olla=u.du_olla
            key=(uid,row['cell_id'],row['mcs_table'],olla.segment,(slot-1)//self.window_slots)
            item=self.windows.setdefault(key,dict(counts=Counter(),offset_min_db=math.inf,offset_max_db=-math.inf,
                first_slot=slot,last_slot=slot))
            item['last_slot']=slot;c=item['counts'];retx=row['is_retx']
            c['retx_samples' if retx else 'first_samples']+=1
            c['retx_errors' if retx else 'first_errors']+=not row['ack']
            if not retx:
                fresh=decision.get('slot_id')==slot
                c['stale_or_missing_measurements']+=not fresh or decision.get('reason')!='delivered_reference_sinr_plus_du_olla'
                lower=decision.get('min_mcs',0);upper=decision.get('max_mcs',27)
                offset=decision.get('offset_db',olla.offsets.get(row['mcs_table'],0.))
                c['saturated_samples']+=row['mcs'] in (lower,upper) or abs(offset)>=olla.limit_db-1e-9
                c['measurement_age_slots_sum']+=decision.get('measurement_age_slots',0) if fresh else 0
                item['offset_min_db']=min(item['offset_min_db'],offset)
                item['offset_max_db']=max(item['offset_max_db'],offset)
        return feedback,selections

    def summary(self,steady):
        rows=[];per_link={}
        for (uid,cell,table,segment,window),item in sorted(self.windows.items()):
            c=item['counts'];n=c['first_samples'];e=c['first_errors'];r=c['retx_samples']
            row=dict(ue_id=uid,cell_id=cell,table=table,segment=segment,window_index=window,
                first_slot=item['first_slot'],last_slot=item['last_slot'],**dict(c),
                first_bler=e/n if n else None,wilson_95=probability_interval(e,n) if n else None,
                retx_error_rate=c['retx_errors']/r if r else None,
                saturated_fraction=c['saturated_samples']/n if n else None,
                offset_range_db=[item['offset_min_db'],item['offset_max_db']] if n else None,
                target_status='insufficient_samples' if n<2000 else 'dynamic_conditions_require_review')
            rows.append(row)
            key=f'{uid}|{cell}|{table}|segment_{segment}'
            per_link.setdefault(key,Counter()).update(c)
        users={};groups={}
        for uid,c in self.ue.items():
            m=steady['per_ue'][uid];tags=[]
            if c['best_candidate_below_recovery_slots'] or c['serving_no_path_slots']:tags.append('coverage_limited')
            if m['overflow_bytes'] or m['queue_growth_bytes']>0 or c['retx_wait_events']:tags.append('resource_pressure_evidence')
            links=[v for k,v in per_link.items() if k.startswith(uid+'|')]
            if any(v['stale_or_missing_measurements'] or v['saturated_samples'] for v in links):tags.append('feedback_or_measurement_limited')
            if not tags:tags=['insufficient_evidence_for_bottleneck']
            users[uid]=dict(**dict(c),**self.last.get(uid,{}),steady_state=m,diagnoses=tags,
                interpretation='Evidence tags may overlap; queue growth alone does not prove a scheduler defect.')
            group=groups.setdefault(self.users[uid].traffic_type,Counter())
            for name in ('offered_bytes','delivered_bytes','overflow_bytes','deadline_misses'):group[name]+=m[name]
        connected=[uid for uid,c in self.ue.items() if c['connected_slots']]
        def cohort(ids):
            return dict(ue_ids=ids,delivered_bytes=sum(self.ue[u]['delivered_bytes'] for u in ids),
                offered_bytes=sum(steady['per_ue'][u]['offered_bytes'] for u in ids))
        return dict(per_ue=users,windows=rows,per_link_segment={k:dict(v) for k,v in per_link.items()},
            all_ues=cohort(list(self.users)),ever_connected_ues=cohort(connected),
            by_traffic_type={k:dict(v) for k,v in groups.items()},feedback_events=self.feedback_count,
            target_interpretation='Dynamic RT segments are diagnostic, not fixed-channel target-BLER certificates.',
            confidence_interpretation='Wilson intervals are descriptive binomial intervals; temporal correlation is not removed.')
