"""Deterministic objective baselines. No predictors or extra PHY executions."""
from collections import deque
from copy import deepcopy
import math
import numpy as np
from .algorithms import Algorithm
from .objectives import OBJECTIVE_DEFAULTS, SEARCH_ALGORITHMS
from ..rule_based_v1.apps import finite,clip,rsrp_map


class ObjectiveAlgorithm(Algorithm):
    def __init__(self,category,algorithm,**kwargs):
        parameters=kwargs.pop('parameters',{}) or {}
        base={k:v for k,v in parameters.items() if k not in OBJECTIVE_DEFAULTS}
        super().__init__(category,algorithm,parameters=base,**kwargs)
        self.parameters.update({**OBJECTIVE_DEFAULTS,**parameters});self.validate(self.parameters)
        self.samples=deque();self.seen_samples={};self.completed_events={};self.weak_reports={};self.weak_windows={}

    def run(self,o,policy_snapshot):
        if self.algorithm not in ('pingpong_guard','failure_guard'):return super().run(o,policy_snapshot)
        # Keep actual departed cohorts visible without rewriting their association.
        from models.action import ConfigAction,ControlDecision
        self.last_decisions=[];actions=[];original=self.parameters
        try:
            self.parameters=self._policy(policy_snapshot);self.capture_reports(o,policy_snapshot)
            for c in o.cell_observations:
                cid=c.cell_id
                if cid not in self.contract.cell_ids:continue
                meta=c.kpi_metadata.get('executed_config',{});produced=meta.get('produced_at_slot',-1)
                if not meta.get('valid') or produced<=self.last_report.get(cid,-1):continue
                self.last_report[cid]=produced
                ues=[u for u in o.ue_observations if (u.serving_cell_id==cid or any(e.get('from_cell')==cid for e in u.mobility_events)) and all(self._valid(u,key) for key in self.contract.required_kpis)]
                if not ues:continue
                if (o.slot_id-self.last_action.get(cid,-10**12))*self.dt<self.parameters.get('min_action_ms',self.min_ms):continue
                a=ConfigAction(target_cell_id=cid,source_name=self.name)
                if self._decide(o,c,c.executed_config,ues,{x.cell_id:x for x in o.cell_observations},a,[]):
                    actions.append(a);self.last_action[cid]=o.slot_id
                    self._reason(cid,'action_requested',produced,values={f:deepcopy(getattr(a,f)) for f in self.controlled_fields if getattr(a,f) is not None})
            return ControlDecision(config_actions=actions or [ConfigAction(source_name=self.name)])
        finally:self.parameters=original

    @staticmethod
    def validate(p):
        Algorithm.validate(p)
        if p.get('objective_window_ms',400)<400:raise ValueError('Objective evaluation window must be >=400ms')
        for key in ('service_floor_bps','starvation_ms','continuity_bad_ms'):
            if p.get(key,1)<=0:raise ValueError('Positive objective parameter required: '+key)
        for key in ('concentrated_power_fraction','top_group_fraction','weak_user_fraction'):
            if not 0<p.get(key,.5)<1:raise ValueError('Objective fraction must lie in (0,1): '+key)

    def capture_reports(self,o,policy_snapshot=None):
        # This path runs even while the wrapper waits for its RET action interval.
        super().capture_reports(o,policy_snapshot)
        for c in o.cell_observations:
            if c.cell_id not in self.contract.cell_ids:continue
            members=[u for u in o.ue_observations if u.serving_cell_id==c.cell_id]
            if not members:continue
            keys=['throughput_mbps','reference_sinr_db']
            if self.algorithm in SEARCH_ALGORITHMS:keys+=['delivered_bits_total'] if self.algorithm!='coverage_fraction_grid' else ['rsrp_dbm']
            if any(not all(self._valid(u,k) for k in keys) for u in members):continue
            produced=min(u.kpi_metadata[keys[-1]]['produced_at_slot'] for u in members)
            if produced<=self.seen_samples.get(c.cell_id,-1):continue
            self.seen_samples[c.cell_id]=produced
            self.samples.append(dict(cell=c.cell_id,slot=produced,users=tuple(sorted(u.ue_id for u in members)),
                bits=sum(getattr(u,'delivered_bits_total',0) for u in members),
                rates={u.ue_id:u.throughput_mbps*1e6 for u in members},
                demands={u.ue_id:getattr(u,'requested_dl_rate_bps',None) for u in members},
                coverage=[min(u.rsrp_dbm-self.parameters['good_rsrp_dbm'],u.reference_sinr_db-self.parameters['good_sinr_db']) for u in members],
                energy=getattr(c,'rf_energy_total_j',None)))
        for u in o.ue_observations:
            if self._valid(u,'reference_sinr_db'):
                produced=u.kpi_metadata['reference_sinr_db']['produced_at_slot'];key=(u.serving_cell_id,u.ue_id)
                old=self.weak_reports.get(key)
                if u.reference_sinr_db<self.parameters['good_sinr_db']:
                    begin=old[0] if old and (produced-old[1])*self.dt<=self.dt*5 else produced
                    self.weak_reports[key]=(begin,produced)
                    if (produced-begin)*self.dt>=40:self.weak_windows[key]=produced
                else:self.weak_reports.pop(key,None)
            if self._valid(u,'mobility_events'):
                for e in u.mobility_events:self.completed_events[(e['slot_id'],e['ue_id'],e['kind'])]=deepcopy(e)
        params=self._policy(policy_snapshot) if policy_snapshot is not None else self.parameters
        horizon=max(params['objective_window_ms']*3,params['mro_window_ms']*2)
        while self.samples and (o.slot_id-self.samples[0]['slot'])*self.dt>horizon:self.samples.popleft()
        self.completed_events={key:e for key,e in self.completed_events.items() if (o.slot_id-e['slot_id'])*self.dt<=horizon}

    def _evidence(self,cid,ues,value=None,proxy=None,**extra):
        self._reason(cid,'objective_observation',objective_value=value,proxy_score=proxy,
            valid_samples=len(ues) if value is not None or proxy is not None else 0,**extra)

    def objective_value(self,ues,c):
        k=self.algorithm
        if k in ('sum_service_power','sum_service_caps','sum_goodput_grid','sum_goodput_beam'):
            return sum(u.throughput_mbps*1e6 for u in ues)
        if k=='weak_user_power':return float(np.quantile([u.throughput_mbps*1e6 for u in ues],.2)) if ues else None
        if k=='proportional_utility_caps':return sum(math.log(max(u.throughput_mbps*1e6,1e-6)) for u in ues)
        if k=='rf_efficiency_search':return getattr(c,'rf_efficiency_bit_per_j',None)
        if k in ('sla_admission_power','sla_count_beam'):
            return sum(u.throughput_mbps*1e6>=u.requested_dl_rate_bps for u in ues) if all(finite(getattr(u,'requested_dl_rate_bps',None)) for u in ues) else None
        if k=='deadline_loss_weights':return sum(u.deadline_misses_window for u in ues)
        return None

    @staticmethod
    def quality(u):
        q=[q+1 for q in u.subband_cqi if finite(q)]
        return sum(q)/len(q) if q else None

    def _decide(self,o,c,cfg,ues,cells,a,commands):
        p=self.parameters;k=self.algorithm;cid=c.cell_id;s=self.state.setdefault(cid,{})
        active=[u for u in ues if u.queue_bytes>0];n=int(cfg['n_prbs'])
        # Metric-specific observations below; unavailable objectives remain None.
        self._evidence(cid,ues,value=self.objective_value(ues,c))
        if k in SEARCH_ALGORITHMS:
            if k!='coverage_fraction_grid' and not active and not any(finite(getattr(u,'offered_rate_bps',None)) and u.offered_rate_bps>0 for u in ues):
                self._reason(cid,'no_active_business');return False
            return self.objective_search(o,c,cfg,ues,a)
        if k in ('sum_service_power','weak_user_power'):
            if not active:return False
            selected=active
            if k=='weak_user_power':selected=sorted(active,key=lambda u:(u.throughput_mbps,u.ue_id))[:max(1,math.ceil(len(active)*p['weak_user_fraction']))]
            scores=[0.]*n
            for u in selected:
                for group,q in zip(u.subband_prb_groups,u.subband_cqi):
                    if finite(q):
                        for g in group:
                            if 0<=g<n:scores[g]+=u.queue_bytes*(q+1)
            valid=[g for g,v in enumerate(scores) if v>0]
            if not valid:return False
            budget=10**((min(cfg['tx_power_dbm'],cfg.get('power_cap_dbm') if cfg.get('power_cap_dbm') is not None else cfg['tx_power_dbm'])-30)/10)
            if k=='sum_service_power':
                top=sorted(valid,key=lambda g:(-scores[g],g))[:max(1,math.ceil(len(valid)*p['top_group_fraction']))]
                rest=[g for g in valid if g not in top];weights=[0.]*n
                for group,fraction in ((top,p['concentrated_power_fraction'] if rest else 1.),(rest,1-p['concentrated_power_fraction'])):
                    if group:
                        total=sum(scores[g] for g in group)
                        for g in group:weights[g]=fraction*scores[g]/total
            else:weights=[v/sum(scores) for v in scores]
            a.prb_power_w=[budget*v for v in weights];a.power_group_size_prbs=1
            self._evidence(cid,selected,value=self.objective_value(selected,c),proxy=scores);return True
        if k=='sla_admission_power':
            if not active:return False
            if any(not finite(u.requested_dl_rate_bps) or u.requested_dl_rate_bps<=0 for u in active):return False
            ratios=[u.throughput_mbps*1e6/u.requested_dl_rate_bps for u in active]
            unmet=[r for r in ratios if r<1]
            # Concentrate recovery on nearly admitted users; protect severe HOL.
            raise_power=any(.5<=r<1 for r in ratios) or any(u.hol_delay_ms>=u.latency_budget_ms for u in active)
            if unmet and not raise_power:self._reason(cid,'no_near_admission_or_deadline_trigger');return False
            maximum=min(p['max_power_dbm'],cfg.get('nominal_tx_power_dbm',30),cfg['power_cap_dbm'] if cfg.get('power_cap_dbm') is not None else math.inf)
            new=clip(cfg['tx_power_dbm']+(p['power_step_db'] if raise_power else -p['power_step_db']),min(p['min_power_dbm'],maximum),maximum)
            self._evidence(cid,active,sum(r>=1 for r in ratios),ratios)
            if new==cfg['tx_power_dbm']:return False
            a.tx_power_dbm=new;return True
        if k in ('sum_service_caps','proportional_utility_caps','starvation_guard_caps'):
            if not active:return False
            caps={u.ue_id:0 for u in ues};demand={u.ue_id:min(n,max(1,math.ceil(u.queue_bytes/1500))) for u in active}
            qualities={u.ue_id:self.quality(u) for u in active} if k!='starvation_guard_caps' else {}
            if qualities and any(v is None for v in qualities.values()):return False
            starve={};last=s.setdefault('last_resource',{})
            if k=='starvation_guard_caps':
                for u in active:
                    total=u.executed_prb_slots_total;epoch=u.resource_counter_epoch
                    produced=u.kpi_metadata['executed_prb_slots_total']['produced_at_slot'];old=last.get(u.ue_id)
                    since=produced if old is None or old[0]!=epoch or total!=old[1] else old[2]
                    last[u.ue_id]=(epoch,total,since);starve[u.ue_id]=(produced-since)*self.dt
                for uid in list(last):
                    if uid not in demand:last.pop(uid)
                ids=sorted(demand,key=lambda uid:(-starve[uid],uid));rotation=s.get('rotation',0)%len(ids)
                waiting=[uid for uid in ids if starve[uid]>=p['starvation_ms']]
                # First serve the overdue cohort up to its conservative demand.
                # Requested caps never reset the starvation clock.
                for _ in range(n):
                    overdue=[uid for uid in waiting if caps[uid]<demand[uid]]
                    if not overdue:break
                    uid=min(overdue,key=lambda uid:(caps[uid],waiting.index(uid)));caps[uid]+=1
                remaining=n-sum(caps.values());ids=ids[rotation:]+ids[:rotation];s['rotation']=rotation+1
                for _ in range(remaining):
                    eligible=[uid for uid in ids if caps[uid]<demand[uid]]
                    if not eligible:break
                    uid=min(eligible,key=lambda uid:(caps[uid],ids.index(uid)));caps[uid]+=1
            else:
                rates={u.ue_id:max(p['service_floor_bps'],u.throughput_mbps*1e6) for u in active}
                for _ in range(n):
                    eligible=[uid for uid in demand if caps[uid]<demand[uid]]
                    if not eligible:break
                    def score(uid):
                        return qualities[uid] if k=='sum_service_caps' else qualities[uid]/rates[uid]/(caps[uid]+1)
                    uid=sorted(eligible,key=lambda uid:(-score(uid),uid))[0];caps[uid]+=1
                self._evidence(cid,active,value=self.objective_value(active,c),proxy=qualities)
            if k=='starvation_guard_caps':self._evidence(cid,active,value=max(starve.values()),proxy=starve)
            a.ue_prb_limits=caps;return True
        if k in ('sla_count_weights','worst_slice_weights','deadline_loss_weights'):
            weights={sid:1. for sid in sorted({u.slice_id for u in ues})};scores={}
            for sid in weights:
                members=[u for u in ues if u.slice_id==sid and u.queue_bytes>0]
                if not members:continue
                if k=='deadline_loss_weights':
                    losses=sum(u.deadline_misses_window for u in members);packets=sum(u.deadline_packets_window for u in members)
                    scores[sid]=(losses/max(packets,1),max(u.hol_delay_ms/max(u.latency_budget_ms,1e-6) for u in members))
                else:
                    if any(not finite(u.requested_dl_rate_bps) or u.requested_dl_rate_bps<=0 for u in members):return False
                    demand=sum(u.requested_dl_rate_bps for u in members);rate=sum(u.throughput_mbps*1e6 for u in members)
                    gap=max(0,1-rate/demand)
                    if gap>0:scores[sid]=gap
            if not scores:return False
            best=sorted(scores,key=lambda sid:(scores[sid] if k=='sla_count_weights' else tuple(-v for v in scores[sid]) if k=='deadline_loss_weights' else -scores[sid],sid))[0]
            weights[best]=p['slice_max_weight']
            for sid in weights:
                if sid!=best:weights[sid]=p['slice_min_weight']
            self.family(a,'slice_aware');a.slice_weight_adjustment=weights
            value=(len(weights)-len(scores))/len(weights) if k=='sla_count_weights' else max(scores.values()) if k=='worst_slice_weights' else sum(u.deadline_misses_window for u in ues)
            self._evidence(cid,ues,value=value,proxy=scores);return True
        if k in ('coverage_margin_steering','continuity_first_steering'):
            for u in ues:
                bad=u.rsrp_dbm<p['good_rsrp_dbm'] or u.reference_sinr_db<p['good_sinr_db']
                produced=u.kpi_metadata['reference_sinr_db']['produced_at_slot']
                previous=s.get(u.ue_id)
                if not bad:s.pop(u.ue_id,None);continue
                if previous is None or (produced-previous[1])*self.dt>self.dt*5:previous=(produced,produced)
                s[u.ue_id]=(previous[0],produced)
                duration=(produced-previous[0])*self.dt
                if k=='continuity_first_steering' and duration<p['continuity_bad_ms']:continue
                targets=self._candidate(u,cells,cfg);rs=rsrp_map(u)
                if not targets or not rs:continue
                def margin(t):return min(rs[t]-p['good_rsrp_dbm'],u.candidate_sinr_db[t]-p['good_sinr_db'])
                target=sorted(targets,key=lambda t:(-margin(t),t))[0]
                current=min(u.rsrp_dbm-p['good_rsrp_dbm'],u.reference_sinr_db-p['good_sinr_db'])
                if margin(target)>current and self._command(o,u,target,commands):s.pop(u.ue_id,None)
            return False
        if k in ('pingpong_guard','failure_guard'):
            events=sorted(self.completed_events.values(),key=lambda e:(e['slot_id'],e['ue_id'],e['kind']))
            ended=[e for e in events if e['kind'] in ('handover_success','handover_failure') and e.get('from_cell')==cid and (o.slot_id-e['slot_id'])*self.dt<=p['mro_window_ms']]
            if len(ended)<p['mro_min_attempts']:self._reason(cid,'insufficient_completed_events');return False
            token=tuple((e['slot_id'],e['ue_id'],e['kind']) for e in ended)
            if s.get('consumed_events')==token:return False
            last={};ping=0
            for e in events:
                if e['kind']!='handover_success' or (o.slot_id-e['slot_id'])*self.dt>p['mro_window_ms']:continue
                old=last.get(e['ue_id'])
                if old and old.get('to_cell')==e.get('from_cell') and old.get('from_cell')==e.get('to_cell') and (e['slot_id']-old['slot_id'])*self.dt<=p['pingpong_ms'] and e.get('from_cell')==cid:ping+=1
                last[e['ue_id']]=e
            failed=sum(e['kind']=='handover_failure' for e in ended)+sum(e['kind']=='connection_drop' and e.get('cell_id',e.get('from_cell'))==cid for e in events)
            if k=='pingpong_guard':
                if failed:self._reason(cid,'failure_protection');return False
                direction=1 if ping else -1
            else:
                weak=any(cell==cid and (o.slot_id-slot)*self.dt<=p['mro_window_ms'] for (cell,uid),slot in self.weak_windows.items())
                if not failed or not weak:self._reason(cid,'no_joint_failure_and_weak_link');return False
                direction=-1
            s['consumed_events']=token
            a.time_to_trigger_ms=clip(cfg['time_to_trigger_ms']+direction*20,20,480)
            a.steering_hysteresis_db=clip(cfg['steering_hysteresis_db']+direction*.5,0.,12)
            self._evidence(cid,ues,failed if k=='failure_guard' else ping)
            return a.time_to_trigger_ms!=cfg['time_to_trigger_ms'] or a.steering_hysteresis_db!=cfg['steering_hysteresis_db']
        raise AssertionError(k)

    def objective_search(self,o,c,cfg,ues,a):
        cid=c.cell_id;k=self.algorithm;p=self.parameters;s=self.state.setdefault(cid,{})
        if k=='rf_efficiency_search':fields=('tx_power_dbm',)
        elif k.endswith('_grid'):fields=('ret_enabled','ret_codebook_id','ret_tilt_deg')
        else:fields=('codebook_id','beam_profile_id')
        if k.endswith('_beam') and cfg['ret_enabled']:self._reason(cid,'ret_owns_analog_subspace');return False
        if k=='rf_efficiency_search' and (not self._valid(c,'rf_energy_total_j') or not finite(getattr(c,'rf_energy_total_j',None))):
            self._reason(cid,'missing_rf_energy');return False
        configs=deepcopy(getattr(o,'executed_configs',{}))
        for field in fields:configs.get(cid,{}).pop(field,None)
        signature=(configs,tuple(sorted((u.ue_id,u.slice_id,getattr(u,'requested_dl_rate_bps',None)) for u in ues)),p['objective_window_ms'])
        if s.get('objective_context')!=signature:
            if s:self._reason(cid,'search_context_changed')
            s.clear();s['objective_context']=signature;s['scores']=[];s['index']=0
            current={f:cfg[f] for f in fields};s['initial']=current
            if k=='rf_efficiency_search':
                hi=min(p['max_power_dbm'],cfg.get('nominal_tx_power_dbm',30),cfg['power_cap_dbm'] if cfg.get('power_cap_dbm') is not None else math.inf)
                targets=[current]+[{'tx_power_dbm':clip(cfg['tx_power_dbm']+d,min(p['min_power_dbm'],hi),hi)} for d in (-p['power_step_db'],p['power_step_db'])]
            elif k.endswith('_grid'):targets=[current]+[dict(ret_enabled=True,ret_codebook_id=0,ret_tilt_deg=x) for x in (0.,3.,6.,9.,12.,15.)]
            else:
                reports={}
                for u in ues:
                    for r in u.beam_reference_reports:
                        if r['valid'] and r['cell_id']==cid:reports.setdefault((r['codebook_id'],r['beam_profile_id']),set()).add(u.ue_id)
                targets=[current]+[dict(codebook_id=cb,beam_profile_id=b) for (cb,b),members in sorted(reports.items()) if len(members)==len(ues)]
            s['candidates']=[]
            for target in targets:
                if target not in s['candidates']:s['candidates'].append(target)
            if len(s['candidates'])<2:self._reason(cid,'insufficient_legal_candidates');return False
        revisions={cell.cell_id:getattr(cell,'config_revision',None) for cell in o.cell_observations}
        if any(value is None for value in revisions.values()):
            self._reason(cid,'missing_configuration_revision');return False
        ownership=getattr(o,'control_ownership',None)
        pending=s.get('pending')
        if pending:
            if pending.get('started') is not None and pending.get('revisions')!=revisions:
                s['scores']=[];s['index']=0;s.pop('pending');self._reason(cid,'search_configuration_interrupted');return False
            if ownership is not None and any(ownership.get(cid,{}).get(f)!=self.name for f in fields):
                if pending.get('started') is not None:
                    s['scores']=[];s['index']=0;s.pop('pending');self._reason(cid,'search_window_overridden');return False
                self._reason(cid,'search_waiting_for_ownership');return False
            if any(cfg[f]!=v for f,v in pending['target'].items()):
                if pending.get('started') is not None:
                    s['scores']=[];s['index']=0;s.pop('pending');self._reason(cid,'search_window_interrupted')
                else:self._reason(cid,'search_waiting_for_execution')
                return False
            # Sample baseline strictly after an executed configuration report.
            samples=[x for x in self.samples if x['cell']==cid and x['slot']>pending['issued']]
            if not samples:return False
            if pending.get('started') is None:
                pending['started']=samples[-1]['slot'];pending['baseline']=samples[-1];pending['revisions']=deepcopy(revisions)
                return False
            begin=pending['started'];end=samples[-1]
            if (end['slot']-begin)*self.dt<p['objective_window_ms']:return False
            window=[x for x in samples if begin<x['slot']<=end['slot']]
            first=pending['baseline']
            if not window:return False
            if k=='coverage_fraction_grid':
                margins=[v for x in window for v in x['coverage']]
                score=(sum(v>=0 for v in margins)/len(margins),float(np.quantile(margins,.1)))
            else:
                seconds=(end['slot']-begin)*self.dt/1000;bits=end['bits']-first['bits']
                if bits<0:self._reason(cid,'counter_epoch_changed');s.clear();return False
                if k=='rf_efficiency_search':
                    energy=end['energy']-first['energy']
                    if energy<=0:self._reason(cid,'no_rf_energy_samples');return False
                    score=(bits/energy,)
                    sla=sum(v>=end['demands'][uid] for uid,v in end['rates'].items() if finite(end['demands'][uid]))
                    if s['scores'] and sla<s['scores'][0]['sla'] and pending['target']['tx_power_dbm']<s['initial']['tx_power_dbm']:
                        for f,v in s['initial'].items():setattr(a,f,v)
                        s['done']=True;s.pop('pending');self._reason(cid,'sla_guard_rollback');return True
                elif k=='sla_count_beam':
                    demand=end['demands']
                    if any(not finite(v) or v<=0 for v in demand.values()):self._reason(cid,'missing_service_requirement');return False
                    score=(sum(end['rates'][uid]>=v for uid,v in demand.items()),-sum(max(0,1-end['rates'][uid]/v) for uid,v in demand.items()))
                else:score=(bits/seconds,)
            s['scores'].append(dict(score=score,target=deepcopy(pending['target']),sla=sum(end['rates'][uid]>=v for uid,v in end['demands'].items() if finite(v))))
            self._reason(cid,'objective_window_evaluated',objective_value=score,valid_samples=len(window),window_start_slot=begin,window_end_slot=end['slot'])
            s.pop('pending');s['index']+=1
        if s.get('done'):return False
        if s['index']>=len(s['candidates']):
            # Stable sorting preserves the initial configuration on equal scores.
            best=max(s['scores'],key=lambda x:x['score']);s['done']=True
            if all(cfg[f]==v for f,v in best['target'].items()):return False
            for f,v in best['target'].items():setattr(a,f,v)
            self._reason(cid,'objective_search_complete',objective_value=best['score']);return True
        target=s['candidates'][s['index']]
        for f,v in target.items():setattr(a,f,v)
        s['pending']=dict(target=deepcopy(target),issued=o.slot_id,started=None)
        return True
