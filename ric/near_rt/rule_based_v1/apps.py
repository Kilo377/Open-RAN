"""Measured-only deterministic xApps; all state is derived from delivered reports."""
from collections import Counter, deque
from copy import deepcopy
import math
import time
from models.action import ConfigAction, CommandAction, ControlDecision
from ric.near_rt.xapp_base import XAppBase
from ric.xapp_contract import XAppContract
from .catalog import CATALOG, DEFAULTS


def clip(x,lo,hi):return max(lo,min(hi,x))
def finite(x):return isinstance(x,(int,float)) and math.isfinite(x)
def mean(xs,default=None):
    xs=[x for x in xs if finite(x)]
    return sum(xs)/len(xs) if xs else default

def waterfill_caps(demand,weights,budget):
    """Deterministic integer weighted max-min allocation, ties by UE ID."""
    result={k:0 for k in demand}
    for _ in range(max(0,int(budget))):
        active=[k for k in demand if result[k]<demand[k]]
        if not active:break
        k=min(active,key=lambda k:((result[k]+1)/max(weights[k],1e-9),k))
        result[k]+=1
    return result


class RuleXApp(XAppBase):
    def __init__(self,kind,name=None,priority=None,parameters=None,cell_ids=(),slot_duration_ms=.5):
        if kind not in CATALOG:raise ValueError(f'Unknown rule xApp: {kind}')
        spec=CATALOG[kind];self.kind=kind;self.parameters={**DEFAULTS,**(parameters or {})}
        unknown=set(self.parameters)-set(DEFAULTS)-{'min_action_ms'}
        if unknown:raise ValueError(f'Unknown rule parameters: {sorted(unknown)}')
        if any(not finite(x) for x in self.parameters.values()):raise ValueError('Rule parameters must be finite numbers')
        if any(v<0 for k,v in self.parameters.items() if k.endswith('_ms')):raise ValueError('Negative timing')
        if slot_duration_ms<=0:raise ValueError('Positive slot duration required')
        self.dt=float(slot_duration_ms);self.min_ms=self.parameters.get('min_action_ms',spec[1])
        contract=XAppContract(tuple(spec[2]),frozenset(spec[3]),cell_ids=tuple(cell_ids),
            commands=frozenset(spec[4]),max_age_slots=math.ceil(self.parameters['max_age_ms']/self.dt),partial_entities=True)
        super().__init__(name or 'rule_'+kind,priority if priority is not None else spec[0],list(spec[3]),contract)
        self.state={};self.last_report={};self.last_action={};self.ewma={};self.command_last={}
        self.events=deque(maxlen=1024);self.statistics=Counter();self.last_decisions=[]
        self.decision_seconds=0.;self._event_ids=set();self._event_order=deque(maxlen=4096)
        self.rule_based_v1=True
        self._validate_parameters(self.parameters)

    @staticmethod
    def _validate_parameters(p):
        for lo,hi in [('min_power_dbm','max_power_dbm'),('ret_min_deg','ret_max_deg'),('slice_min_weight','slice_max_weight')]:
            if p[lo]>p[hi]:raise ValueError(f'Inverted bounds: {lo}/{hi}')
        if not 0<=p['ret_min_deg']<=p['ret_max_deg']<=15 or not 0<=p['protected_outer_fraction']<=1:raise ValueError('Invalid physical bounds')
        if any(p[k]<=0 for k in ('power_step_db','ret_step_deg','cio_step_db','minimum_samples','ts_samples','mro_min_attempts','slice_min_weight')):raise ValueError('Nonpositive step/sample bound')

    def reset(self):
        self.state.clear();self.last_report.clear();self.last_action.clear();self.ewma.clear();self.command_last.clear()

    def _valid(self,entity,name):
        metadata=getattr(entity,'kpi_metadata',{}).get(name,{})
        return metadata.get('valid',False) and metadata.get('sample_count',0)>=self.parameters['minimum_samples']

    def _reason(self,cid,reason,report=None,**extra):
        row=dict(xapp=self.name,kind=self.kind,cell_id=cid,reason=reason,report_id=report,**extra)
        self.last_decisions.append(row);self.statistics[reason]+=1

    def _command(self,o,u,target,commands):
        if getattr(u,'handover_pending_target_cell_id',None):return False
        if getattr(u,'connection_state',None) not in ('connected','active'):return False
        if (o.slot_id-self.command_last.get(u.ue_id,-10**12))*self.dt<self.parameters['command_cooldown_ms']:return False
        request=f'{self.name}:{u.ue_id}:{target}:{o.slot_id}'
        commands.append(CommandAction('request_handover',target_ue_id=u.ue_id,target_cell_id=target,
            source_name=self.name,control_request_id=request,valid_from_slot=o.slot_id,
            valid_until_slot=o.slot_id+max(5,math.ceil(self.parameters['command_cooldown_ms']/self.dt))))
        self.command_last[u.ue_id]=o.slot_id
        self._reason(u.serving_cell_id,'command_requested',getattr(o,'rule_report_produced_at_slot',None),
            request_id=request,ue_id=u.ue_id,target_cell_id=target,command_type='request_handover')
        return True

    def _candidate(self,u,cells,source):
        rs=getattr(u,'candidate_rsrp_dbm',None);sn=getattr(u,'candidate_sinr_db',None)
        if not isinstance(rs,dict) or not isinstance(sn,dict):return []
        neighbors=source.get('neighbor_cell_ids')
        return [cid for cid in sorted(cells) if cid!=u.serving_cell_id and
            (neighbors is None or cid in neighbors) and cid in rs and cid in sn and
            finite(rs[cid]) and finite(sn[cid]) and rs[cid]>self.parameters['good_rsrp_dbm'] and
            sn[cid]>self.parameters['good_sinr_db'] and cells[cid].cell_sleep_mode=='active' and
            finite(getattr(cells[cid],'resource_grid_utilization',None)) and
            cells[cid].resource_grid_utilization<self.parameters['target_max_load']]

    def _policy(self,p):
        overrides=getattr(p,'rule_based_parameters',{}) or {}
        params=overrides.get(self.name,overrides.get(self.kind,{}))
        if params:
            if set(params)-set(DEFAULTS)-{'min_action_ms'} or any(not finite(x) for x in params.values()):raise ValueError('Invalid non-RT rule parameters')
        result={**self.parameters,**params}
        cap=getattr(p,'preferred_power_cap_dbm',None)
        if cap is not None:result['max_power_dbm']=min(result['max_power_dbm'],float(cap))
        self._validate_parameters(result)
        return result

    def run(self,observation,policy_snapshot):
        started=time.perf_counter();o=observation;self.last_decisions=[];actions=[];commands=[]
        original=self.parameters;self.parameters=self._policy(policy_snapshot)
        try:
            cells={c.cell_id:c for c in o.cell_observations}
            if self.kind=='ret_coverage' and self.last_action and (o.slot_id-max(self.last_action.values()))*self.dt<self.min_ms:
                self._reason(None,'action_dwell');return ControlDecision(config_action=ConfigAction(source_name=self.name))
            items=sorted(cells.items())
            if self.kind=='ret_coverage' and items:
                index=int(o.slot_id*self.dt/max(self.min_ms,1))%len(items)
                items=items[index:]+items[:index]
                pending=[item for item in items if self.state.get(item[0],{}).get('pending')]
                if pending:items=pending
            for cid,c in items:
                if self.contract.cell_ids and cid not in self.contract.cell_ids:continue
                cfg=getattr(c,'executed_config',None)
                if not cfg:self._reason(cid,'missing_executed_config');continue
                meta=c.kpi_metadata.get('executed_config',{})
                if not meta.get('valid'):self._reason(cid,'invalid_executed_config');continue
                produced=meta['produced_at_slot']
                if produced<=self.last_report.get(cid,-1):self._reason(cid,'no_new_report',produced);continue
                self.last_report[cid]=produced
                members=[u for u in o.ue_observations if u.serving_cell_id==cid]
                valid=[u for u in members if all(self._valid(u,k) for k in self.contract.required_kpis
                    if k not in ('resource_grid_utilization',))]
                skipped=[u.ue_id for u in members if u not in valid]
                if skipped:self._reason(cid,'invalid_entity_kpi',produced,ue_ids=skipped)
                # Internal histories advance once per delivered report, never once per repeated call.
                for u in valid:
                    value=getattr(u,'throughput_mbps',None)
                    if finite(value):
                        prev=self.ewma.get(u.ue_id,(produced,value));alpha=1-math.exp(-(produced-prev[0])*self.dt/max(self.parameters['ewma_ms'],1e-9))
                        self.ewma[u.ue_id]=(produced,prev[1]+alpha*(value-prev[1]))
                interval=self.parameters.get('min_action_ms',self.min_ms)
                if (o.slot_id-self.last_action.get(cid,-10**12))*self.dt<interval:self._reason(cid,'action_dwell',produced);continue
                if self.kind!='cell_sleep' and not valid:self._reason(cid,'no_valid_ue_samples',produced);continue
                if self.kind=='cell_sleep' and skipped:self._reason(cid,'coverage_information_incomplete',produced);continue
                if c.cell_sleep_mode!='active' and self.kind!='cell_sleep':self._reason(cid,'cell_not_active',produced);continue
                action=ConfigAction(target_cell_id=cid,source_name=self.name,apply_scope='cell')
                action.control_request_id=f'{self.name}:{cid}:{produced}'
                changed=self._decide(o,c,cfg,valid,cells,action,commands)
                if changed:
                    actions.append(action);self.last_action[cid]=o.slot_id
                    self._reason(cid,'action_requested',produced,values={k:deepcopy(getattr(action,k)) for k in self.controlled_fields if getattr(action,k) is not None})
                    if self.kind=='ret_coverage':break
                else:self._reason(cid,'hold',produced)
            return ControlDecision(config_actions=actions or [ConfigAction(source_name=self.name)],commands=commands)
        finally:
            self.parameters=original;self.decision_seconds+=time.perf_counter()-started

    def _decide(self,o,c,cfg,ues,cells,a,commands):
        p=self.parameters;cid=c.cell_id;kind=self.kind
        state=self.state.setdefault(cid,{})
        n=int(cfg['n_prbs']);budget=10**((min(cfg['tx_power_dbm'],cfg.get('power_cap_dbm') if cfg.get('power_cap_dbm') is not None else cfg['tx_power_dbm'])-30)/10)
        active=[u for u in ues if (getattr(u,'queue_bytes',None) or 0)>0]
        if kind=='throughput_power':
            scores=[0.]*n
            for u in active:
                for indices,q in zip(u.subband_prb_groups,u.subband_cqi):
                    if q is None:continue
                    for g in indices:
                        if 0<=g<n:scores[g]+=max(1,getattr(u,'priority',1))*(q+1)/max(self.ewma.get(u.ue_id,(0,u.throughput_mbps))[1],.1)
            total=sum(scores)
            if total<=0:return False
            a.prb_power_w=[budget*x/total for x in scores];a.power_group_size_prbs=1;return True
        if kind=='rf_energy':
            sn=min(u.reference_sinr_db for u in ues);unsafe=sn<p['sinr_target_db']-p['sinr_deadband_db']
            for u in ues:
                demand=getattr(u,'requested_dl_rate_bps',None)
                unsafe|=u.hol_delay_ms>getattr(u,'latency_budget_ms',100)*.7 or (finite(demand) and u.throughput_mbps*1e6<demand and u.queue_bytes>0)
            safe=sn>p['sinr_target_db']+p['sinr_deadband_db'] and not unsafe
            if not unsafe and not safe:return False
            value=clip(cfg['tx_power_dbm']+(p['power_step_db'] if unsafe else -p['power_step_db']),p['min_power_dbm'],min(p['max_power_dbm'],cfg.get('nominal_tx_power_dbm',30)))
            if cfg.get('power_cap_dbm') is not None:value=min(value,cfg['power_cap_dbm'])
            if abs(value-cfg['tx_power_dbm'])<1e-9:return False
            a.tx_power_dbm=value;return True
        if kind=='load_balancing':
            if not finite(c.resource_grid_utilization):return False
            offsets=deepcopy(cfg.get('cio_offsets_db',{}));changed=False
            for target in sorted(cells):
                if target==cid:continue
                candidates=[u for u in ues if target in self._candidate(u,cells,cfg)]
                load=getattr(cells[target],'resource_grid_utilization',None)
                if candidates and finite(load) and c.resource_grid_utilization-load>p['load_gap']:
                    old=offsets.get(target,0.);offsets[target]=clip(old+p['cio_step_db'],-p['cio_bound_db'],p['cio_bound_db']);changed|=offsets[target]!=old
            if changed:a.cio_offsets_db=offsets
            return changed
        if kind=='traffic_steering':
            for u in ues:
                demand=getattr(u,'requested_dl_rate_bps',None)
                poor=u.hol_delay_ms>getattr(u,'latency_budget_ms',100)*.5 or (finite(demand) and u.throughput_mbps*1e6<demand)
                rs=u.candidate_rsrp_dbm;sn=u.candidate_sinr_db
                def score(t):return (clip((sn[t]+10)/40,0,1)+clip((rs[t]+120)/60,0,1)+(1-cells[t].resource_grid_utilization))/3
                targets=self._candidate(u,cells,cfg)
                target=max(targets,key=lambda t:(score(t),t)) if targets else None
                good=poor and target is not None and cid in rs and cid in sn and score(target)>score(cid)+p['ts_margin']
                prev=state.get(u.ue_id,(None,0));count=prev[1]+1 if good and prev[0]==target else 1 if good else 0
                state[u.ue_id]=(target,count)
                if count>=p['ts_samples'] and self._command(o,u,target,commands):state[u.ue_id]=(None,0)
            return False
        if kind=='cell_sleep':
            load=c.resource_grid_utilization
            if not finite(load):return False
            if (o.slot_id-state.get('transition',-10**12))*self.dt<p['sleep_dwell_ms']:return False
            if c.cell_sleep_mode!='active':
                needed=any(getattr(x,'resource_grid_utilization',0)>p['wake_high_load'] for x in cells.values()) or any(getattr(u,'queue_bytes',0)>0 and getattr(u,'connection_state','')!='connected' for u in o.ue_observations)
                if needed:a.cell_sleep_mode='active';state['transition']=o.slot_id;return True
                return False
            # Keep at least one live cell; protect unmanaged neighbors from commands.
            if sum(x.cell_sleep_mode=='active' for x in cells.values())<=1:return False
            low=load<p['sleep_low_load'] and all((getattr(u,'hol_delay_ms',None) is not None and u.hol_delay_ms<getattr(u,'latency_budget_ms',100)*.5) for u in ues)
            if not low:state.pop('low_since',None);return False
            state.setdefault('low_since',o.slot_id)
            if (o.slot_id-state['low_since'])*self.dt<p['sleep_low_ms']:return False
            if ues:
                for u in ues:
                    choices=self._candidate(u,cells,cfg)
                    if not choices:return False
                for u in ues:
                    choices=self._candidate(u,cells,cfg);target=min(choices,key=lambda t:(cells[t].resource_grid_utilization,t))
                    self._command(o,u,target,commands)
                return False
            # No served users and no hidden/unconnected user relying on this cell.
            for u in o.ue_observations:
                rs=getattr(u,'candidate_rsrp_dbm',None);sn=getattr(u,'candidate_sinr_db',None)
                if not isinstance(rs,dict) or not isinstance(sn,dict):return False
                alternatives=[t for t in cells if t!=cid and cells[t].cell_sleep_mode=='active' and rs.get(t,-999)>p['good_rsrp_dbm'] and sn.get(t,-999)>p['good_sinr_db']]
                if not alternatives:return False
            a.cell_sleep_mode='deep_sleep';state['transition']=o.slot_id;return True
        if kind=='mobility_robustness':
            ev={}
            for u in ues:
                for e in u.mobility_events:
                    if (o.slot_id-e['slot_id'])*self.dt<=p['mro_window_ms']:ev[(e['slot_id'],e['ue_id'],e['kind'])]=e
            ended=sorted((e for e in ev.values() if e['kind'] in ('handover_success','handover_failure')),key=lambda e:(e['slot_id'],e['ue_id']))
            if len(ended)<p['mro_min_attempts']:return False
            signature=tuple(ev)
            if signature==state.get('signature'):return False
            state['signature']=signature;last={};ping=0
            for e in ended:
                old=last.get(e['ue_id'])
                if e['kind']=='handover_success':
                    if old and old.get('from_cell')==e.get('to_cell') and old.get('to_cell')==e.get('from_cell') and (e['slot_id']-old['slot_id'])*self.dt<=p['pingpong_ms']:ping+=1
                    last[e['ue_id']]=e
            bad=sum(e.get('reason')=='serving_link_lost' for e in ended)
            direction=1 if ping else -1 if bad and min(u.reference_sinr_db for u in ues)<p['good_sinr_db'] else 0
            if not direction:return False
            a.time_to_trigger_ms=clip(cfg['time_to_trigger_ms']+direction*20,20,480)
            a.steering_hysteresis_db=clip(cfg['steering_hysteresis_db']+direction*.5,.5,12);return True
        if kind=='ue_fairness':
            weights={u.ue_id:max(1.,getattr(u,'priority',1))/max(self.ewma.get(u.ue_id,(0,u.throughput_mbps))[1],.1) for u in active}
            demand={u.ue_id:min(n,max(1,math.ceil(n*max(u.offered_rate_bps,getattr(u,'requested_dl_rate_bps',None) or 0)/max(sum(max(x.offered_rate_bps,getattr(x,'requested_dl_rate_bps',None) or 0) for x in active),1)))) for u in active}
            limits=waterfill_caps(demand,weights,n)
            limits.update({u.ue_id:0 for u in ues if u not in active})
            a.ue_prb_limits=limits;return True
        if kind in ('slice_sla','latency_guard'):
            pressure={};urgent=0.
            for u in ues:
                sid=u.slice_id;hol=u.hol_delay_ms/max(getattr(u,'latency_budget_ms',100),1e-6)
                demand=getattr(u,'requested_dl_rate_bps',None)
                deficit=max(0.,1-u.throughput_mbps*1e6/demand) if finite(demand) and demand>0 and u.queue_bytes>0 else 0.
                miss=getattr(u,'deadline_misses_window',None);count=getattr(u,'deadline_packets_window',None)
                deadline=miss/count if finite(miss) and finite(count) and count>0 else 0.
                pressure[sid]=max(pressure.get(sid,0.),hol+deficit+deadline);urgent=max(urgent,hol+deadline)
            weights={s:clip(1+x,p['slice_min_weight'],p['slice_max_weight']) for s,x in pressure.items()}
            a.slice_weight_adjustment=weights
            if kind=='slice_sla':a.scheduler_mode='slice_aware'
            if kind=='latency_guard':
                a.scheduler_mode='latency_aware' if urgent>.7 else state.setdefault('baseline_scheduler',cfg['scheduler_mode'])
                a.latency_bias=clip(1+urgent,.1,4.) if urgent>.7 else 1.
            return True
        if kind=='interference_coordination':
            ordered=sorted(cells);edges={x:set() for x in ordered}
            for u in o.ue_observations:
                rs=getattr(u,'candidate_rsrp_dbm',None)
                if not isinstance(rs,dict):continue
                source=u.serving_cell_id
                if source not in edges:continue
                for target in ordered:
                    if target!=source and rs.get(target,-999)>p['interference_edge_rsrp_dbm']:edges[source].add(target);edges[target].add(source)
            colors={}
            for x in sorted(ordered,key=lambda x:(-len(edges[x]),x)):
                used={colors[y] for y in edges[x] if y in colors};colors[x]=next(z for z in range(len(ordered)) if z not in used)
            count=max(colors.values(),default=0)+1;epoch=int(o.slot_id*self.dt/max(self.min_ms,1))
            a.prb_power_w=[budget/n*(1 if (g+epoch)%count==colors[cid] else p['protected_outer_fraction']) for g in range(n)]
            a.power_group_size_prbs=1;return True
        if kind=='ret_coverage':
            value=mean([u.reference_sinr_db for u in ues]);tilt=cfg['ret_tilt_deg']
            # Configuration excludes this controller's own RET change.
            signature=deepcopy(getattr(o,'executed_configs',{}))
            for x in signature.values():x.pop('ret_tilt_deg',None);x.pop('ret_enabled',None);x.pop('ret_codebook_id',None)
            pending=state.get('pending')
            if pending:
                if not cfg['ret_enabled'] or abs(tilt-pending['requested'])>1e-8:return False  # Not executed yet / lost arbitration.
                if getattr(c,'config_produced_at_slot',0)<=pending['issued']:return False
                # Evaluate only a window formed after actuation and settling.
                if (o.slot_id-pending['issued'])*self.dt<self.min_ms*2:return False
                if signature!=pending['signature']:
                    self._reason(cid,'ret_effect_not_attributable');state.pop('pending');return False
                state.pop('pending')
                if value<pending['quality']+p['ret_gain_db']:
                    a.ret_enabled=True;a.ret_codebook_id=0;a.ret_tilt_deg=pending['before'];state['direction']=-state.get('direction',1);return True
            direction=state.get('direction',1);target=clip(tilt+direction*p['ret_step_deg'],p['ret_min_deg'],p['ret_max_deg'])
            if target==tilt:state['direction']=-direction;return False
            a.ret_enabled=True;a.ret_codebook_id=0;a.ret_tilt_deg=target
            state['pending']=dict(before=tilt,requested=target,quality=value,signature=signature,issued=o.slot_id);return True
        if kind=='beam_management':
            reports={}
            for u in ues:
                for r in u.beam_reference_reports:
                    if r['cell_id']==cid and r['valid']:
                        key=(r['codebook_id'],r['beam_profile_id']);reports.setdefault(key,[]).append((r['reference_sinr_db'],max(1,getattr(u,'priority',1))))
            if cfg['ret_enabled']:self._reason(cid,'ret_owns_analog_subspace');return False
            if not reports:return False
            scores={k:sum(v*w for v,w in x)/sum(w for _,w in x) for k,x in reports.items()}
            current=(cfg['codebook_id'],cfg['beam_profile_id']);best=max(sorted(scores),key=scores.get)
            if scores[best]<=scores.get(current,-999)+p['beam_gain_db']:return False
            a.codebook_id,a.beam_profile_id=best;return True
        raise AssertionError(kind)


class KpiMonitor(XAppBase):
    """Read-only aggregation. No conflict inference and no raw history retention."""
    rule_based_v1=True
    def __init__(self,name='rule_kpi_monitor'):
        super().__init__(name,0,[],XAppContract((),frozenset(),partial_entities=True,read_only=True))
        self.statistics=Counter();self.last_decisions=[];self.latest={};self.last=-1
    def run(self,o,p):
        self.last_decisions=[]
        report=getattr(o,'rule_report_produced_at_slot',-1)
        if report>self.last:
            self.last=report;self.statistics['reports']+=1
            self.latest=dict(slot_id=o.slot_id,produced_at_slot=report,
                valid=sum(r['valid'] for r in o.kpi_records),invalid=sum(not r['valid'] for r in o.kpi_records),
                sla_alerts=[u.ue_id for u in o.ue_observations if (getattr(u,'sla_status',None) or {}).get('violated')])
            self.last_decisions=[dict(xapp=self.name,kind='monitor',reason='report',**self.latest)]
        return ControlDecision(config_action=ConfigAction(source_name=self.name))
