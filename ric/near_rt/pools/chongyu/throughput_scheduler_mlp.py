"""X-08-inspired MLP selecting an existing weighted scheduler adapter.

Latency-aware base scores and unchanged NR packing consume the UE weights.
The default policy maximizes predicted goodput; optional QoS guard scores
are heuristics. Matched OFF checks report throughput and deadline outcomes.
"""
from collections import deque
from copy import deepcopy
import math
import json
from pathlib import Path
import time

from models.action import ConfigAction,ControlDecision
from ric.near_rt.pools.base import PoolXApp
from .online_throughput_mlp import OnlineThroughputMLP,MIN_ROWS,MAX_ROWS

ROOT=Path(__file__).resolve().parents[4]
USERS=('ue_0','ue_1','ue_2','ue_4','ue_6','ue_8')
URLLC=('ue_2','ue_6')
ACTIONS=('native_rr','weighted_neutral','efficiency_1_5','efficiency_2')
FEATURES=['network_queue_kb','network_reported_goodput_mbps','network_offered_mbps',
          'minimum_reference_sinr_db','mean_reference_sinr_db','urllc_max_hol_budget_ratio',
          'backlogged_ue_fraction','urllc_cumulative_failure_events']+[f'action_{i}' for i in range(4)]
TARGETS=['network_goodput_mbps']+[uid+'_goodput_mbps' for uid in USERS]+[uid+'_failure_event_sum_per_s' for uid in URLLC]
FIELDS=('scheduler_mode','ue_scheduler_weights')
# Numerical allowance is 0.1% of one observed event, not an SLA risk margin.
GUARD_NUMERIC_EVENT_TOLERANCE=.001
PACKET_COUNTERS=('admitted_packets_total','deadline_missed_packets_total','overflow_packets_total','completed_packets_total','late_completed_packets_total')


def action_map(users,index):
    """Deterministic public-observation rule; caller snapshots the returned map."""
    if type(index) is not int or not 0<=index<len(ACTIONS):raise ValueError('Invalid throughput arm')
    result={}
    for u in users:
        result.setdefault(u.serving_cell_id,dict(scheduler_mode='round_robin' if index==0 else 'latency_aware',ue_scheduler_weights={}))['ue_scheduler_weights'][u.ue_id]=1.
    if index>=2:
        for cid,values in result.items():
            active=[u for u in users if u.serving_cell_id==cid and u.queue_bytes>0]
            if active:
                embb=[u for u in active if u.slice_id.lower()=='embb']
                nonurgent=[u for u in active if u.slice_id.lower()!='urllc']
                best=max(embb or nonurgent or active,key=lambda u:(u.reference_sinr_db,u.ue_id))
                values['ue_scheduler_weights'][best.ue_id]=1.5 if index==2 else 2.
    return result


class ThroughputSchedulerMLP(PoolXApp):
    implementation_id='throughput_scheduler_mlp'
    category_id='throughput_scheduler'
    description='X-08: measured network-goodput MLP selecting existing UE-priority weights'
    required_kpis=('queue_bytes','hol_delay_ms','reference_sinr_db','throughput_mbps','offered_rate_bps',
                   'delivered_bits_total','offered_bits_total','executed_config')+PACKET_COUNTERS
    writable_actions=FIELDS
    min_action_ms=100

    def validate_parameters(self,p):
        super().validate_parameters(p)
        if set(p)-{'mode','model_path','log_path','window_ms','seed','fixed_action','policy'}:raise ValueError('Unknown throughput MLP parameters')
        if p.get('mode','online') not in ('online','frozen','fixed'):raise ValueError('Invalid throughput mode')
        if p.get('policy','throughput') not in ('guarded','throughput'):raise ValueError('Invalid throughput policy')
        if type(p.get('window_ms',100)) not in (int,float) or p.get('window_ms',100) not in (100,200):raise ValueError('Throughput window must be 100 or 200 ms')
        if type(p.get('seed',173)) is not int or not 0<=p.get('seed',173)<=2**32-1:raise ValueError('Invalid throughput seed')
        if type(p.get('fixed_action',3)) is not int or not 0<=p.get('fixed_action',3)<4:raise ValueError('Invalid fixed throughput arm')
        for key in ('model_path','log_path'):
            if p.get(key) is not None and not Path(p[key]).resolve().is_relative_to(ROOT/'outputs'):raise ValueError('Throughput artifacts must stay under outputs')
        if p.get('mode')=='frozen' and not p.get('model_path'):raise ValueError('Frozen throughput MLP requires model')

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.mode=self.parameters.get('mode','online');self.window_ms=self.parameters.get('window_ms',100)
        self.policy=self.parameters.get('policy','throughput')
        self.model_path=Path(self.parameters['model_path']) if self.parameters.get('model_path') else None
        self.log_path=Path(self.parameters['log_path']) if self.parameters.get('log_path') else None
        self.learner=OnlineThroughputMLP(FEATURES,TARGETS,self.parameters.get('seed',173))
        if self.model_path and self.model_path.is_file():self.learner=OnlineThroughputMLP.load(self.model_path,FEATURES,TARGETS)
        if any(sum(row[-4:])!=1 or any(v not in (0.,1.) for v in row[-4:]) for row in self.learner._x):raise ValueError('Invalid persisted throughput action encoding')
        self.pending=None;self.events=deque(maxlen=256)
        self.invalidated=self.mlp_selected_actions=self.fixed_actions=self.rules_only_actions=0
        self.status='not_evaluable';self.status_reason='no_valid_public_report';self.last_guard_bad=False;self.last_prediction_audit=None
        self.last_measured_action=None
        self.bootstrap_cooldown_pending=False
        if self.mode=='online' and self.learner._x and not self._coverage():
            # Resume only from retained real rows. A bad RR baseline is not
            # evidence against an untried weighted arm.
            self.last_measured_action=self.learner._x[-1][-4:].index(1.)
            reference=self._reference()
            self.last_guard_bad=bool(reference and any(v>reference[7+i]+self._guard_rate_tolerance() for i,v in enumerate(self.learner._y[-1][7:])))
            self.bootstrap_cooldown_pending=self.policy=='guarded' and self.last_guard_bad and self.last_measured_action!=0
        if self.mode=='frozen' and (not self.learner.trained or not self.learner.optimizer_verified or not self._coverage(self.learner.trained_samples)):
            raise ValueError('Frozen MLP requires a verified mean-output optimizer, 24 fitted measured rows and six per action')
        if self.log_path:
            self.log_path.parent.mkdir(parents=True,exist_ok=True)
            if self.log_path.exists():raise ValueError('Choose fresh throughput event log')

    @property
    def constraints_active(self):
        return self.policy=='guarded' and self.mode!='fixed'

    def action_samples(self,limit=None):
        rows=self.learner._x if limit is None else self.learner._x[:limit]
        return {str(i):sum(r[-4+i]==1. for r in rows) for i in range(4)}

    def _coverage(self,limit=None):
        n=self.learner.sample_count if limit is None else limit
        return n>=MIN_ROWS and all(v>=6 for v in self.action_samples(limit).values())

    def _log(self,kind,**values):
        row=dict(kind=kind,mode=self.mode,**values)
        row.update(policy=self.policy,constraints_active=self.constraints_active);self.events.append(row)
        if self.log_path:
            with self.log_path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')

    def _invalidate(self,reason,produced=None):
        if self.pending is not None:
            self.pending=None;self.invalidated+=1;self._log('sample_invalidated',reason=reason,produced_slot=produced)

    def _unavailable(self,reason):
        self._invalidate(reason)
        if self.status!='not_evaluable' or self.status_reason!=reason:self._log('not_evaluable',reason=reason,persistence='No new request; empty decision is not rollback')
        self.status,self.status_reason='not_evaluable',reason

    @staticmethod
    def _deadline(u):
        values=tuple(getattr(u,key,None) for key in PACKET_COUNTERS)
        if any(type(v) is not int or v<0 for v in values):raise ValueError('invalid_public_packet_counter')
        return values

    def _features(self,users,index):
        urgent=[u for u in users if u.ue_id in URLLC]
        failures=[v[1]+v[2]+v[4] for v in (self._deadline(u) for u in urgent)]
        return [sum(u.queue_bytes for u in users)/1000,sum(u.throughput_mbps for u in users),
            sum(u.offered_rate_bps for u in users)/1e6,min(u.reference_sinr_db for u in users),
            sum(u.reference_sinr_db for u in users)/len(users),max(u.hol_delay_ms/u.latency_budget_ms for u in urgent),
            sum(u.queue_bytes>0 for u in users)/len(users),max(failures)]+[float(i==index) for i in range(4)]

    def _reference(self):
        # Actual training RR windows, not a measured counterfactual for deployment.
        limit=self.learner.trained_samples if self.learner.trained else self.learner.sample_count
        rows=[y for x,y in zip(self.learner._x[:limit],self.learner._y[:limit]) if x[-4]==1.]
        return [sum(y[i] for y in rows)/len(rows) for i in range(len(TARGETS))] if rows else None

    def _guard_rate_tolerance(self):
        return GUARD_NUMERIC_EVENT_TOLERANCE/(self.window_ms/1000)

    def _choose(self,users):
        self.last_prediction_audit=None
        if self.mode=='fixed':return self.parameters.get('fixed_action',3),None,'fixed','predeclared_fixed_policy'
        if self.mode=='online' and self.learner.sample_count<MAX_ROWS and not self._coverage():
            if self.policy=='guarded' and self.bootstrap_cooldown_pending:
                return 0,None,'rules_only','bootstrap_urllc_measured_rr_cooldown'
            counts=self.action_samples();i=min(range(4),key=lambda i:(counts[str(i)],i))
            return i,None,'rules_only','balanced_actual_network_exploration'
        if self.policy=='guarded' and self.last_guard_bad:return 0,None,'rules_only','observed_urllc_guard_abstention'
        if not self.learner.trained or not self.learner.optimizer_verified:return 0,None,'rules_only','untrained_or_unverified_native_rr_abstention'
        predictions=self.learner.predict([self._features(users,i) for i in range(4)])
        reference=self._reference()
        rejected={}
        tolerance=self._guard_rate_tolerance()
        for i,values in enumerate(predictions):
            reasons=[uid+':predicted_goodput_below_observed_RR_training_reference' for j,uid in enumerate(USERS) if values[1+j]<.95*reference[1+j]]
            reasons.extend(uid+':predicted_failure_event_rate_above_training_reference' for j,uid in enumerate(URLLC) if max(0.,values[7+j])>reference[7+j]+tolerance)
            if reasons:rejected[str(i)]=reasons
        feasible=[i for i in range(4) if str(i) not in rejected] if self.policy=='guarded' else list(range(4))
        self.last_prediction_audit=dict(raw_predictions=deepcopy(predictions),
            observed_RR_training_reference=deepcopy(reference),
            rejected_actions=rejected if self.policy=='guarded' else {},
            unapplied_guard_warnings=rejected if self.policy=='throughput' else {},
            policy=self.policy,constraints_active=self.constraints_active,
            numeric_tolerance_events_per_window=GUARD_NUMERIC_EVENT_TOLERANCE,
            numeric_tolerance_event_rate_per_s=tolerance,
            scope='numerical_proxy_guard_not_calibrated_SLA_probability' if self.policy=='guarded' else 'throughput_only_research_no_QoS_constraints')
        index=max(feasible,key=lambda i:(predictions[i][0],-i)) if feasible else 0
        if index==0 or predictions[index][0]<1.01*max(0.,predictions[0][0]):index=0
        self.mlp_selected_actions+=1
        return index,predictions[index],'mlp',('predicted_goodput_with_training_reference_guards' if self.policy=='guarded' else 'predicted_goodput_single_objective_QoS_unconstrained')

    @staticmethod
    def _membership(users):return {u.ue_id:u.serving_cell_id for u in users}

    def _observe(self,users,cells,produced,owners):
        current=self.pending
        if current is None:return
        if self._membership(users)!=current['membership']:
            self._invalidate('handover_or_membership_changed',produced);return
        if produced<current['requested_slot']:return
        matches=True
        for cid,expected in current['snapshot'].items():
            cfg=cells[cid].executed_config;owned=owners.get(cid,{})
            matches=matches and owned.get('scheduler_mode')==self.name
            matches=matches and cfg.get('scheduler_mode')==expected['scheduler_mode']
            # Empty UID maps acquire no weight lease: only the cell's mode
            # must be owned. Occupied cells still require actual weight ownership.
            if expected['ue_scheduler_weights']:
                matches=matches and owned.get('ue_scheduler_weights')==self.name
                matches=matches and all(cfg.get('ue_scheduler_weights',{}).get(uid)==value for uid,value in expected['ue_scheduler_weights'].items())
            if expected['scheduler_mode']=='latency_aware' and cfg.get('latency_bias') not in (None,1.):matches=False
        if not matches:
            if current.get('confirmed') or produced-current['requested_slot']>round(200/self.dt):self._invalidate('execution_or_ownership_changed',produced)
            return
        if not current.get('confirmed'):
            current.update(confirmed=True,confirmed_slot=produced)
            self._log('executed_policy_confirmed',action=current['action'],produced_slot=produced,
                executed_snapshot=deepcopy(current['snapshot']),owners=deepcopy(owners))
        if produced-current['confirmed_slot']<round(20/self.dt):return
        if 'start_slot' not in current:
            current.update(start_slot=produced,counters={u.ue_id:[u.delivered_bits_total,u.offered_bits_total] for u in users},
                deadlines={u.ue_id:self._deadline(u) for u in users if u.ue_id in URLLC},
                deadline_start_slots={u.ue_id:u.kpi_metadata.get('admitted_packets_total',{}).get('produced_at_slot',produced) for u in users if u.ue_id in URLLC},
                max_hol={u.ue_id:u.hol_delay_ms for u in users if u.ue_id in URLLC})
            return
        for u in users:
            if u.ue_id in URLLC:current['max_hol'][u.ue_id]=max(current['max_hol'][u.ue_id],u.hol_delay_ms)
        if (produced-current['start_slot'])*self.dt+1e-9<self.window_ms:return
        duration=(produced-current['start_slot'])*self.dt/1000
        delivered={u.ue_id:u.delivered_bits_total-current['counters'][u.ue_id][0] for u in users}
        offered={u.ue_id:u.offered_bits_total-current['counters'][u.ue_id][1] for u in users}
        if any(v<0 for v in list(delivered.values())+list(offered.values())):
            self._invalidate('counter_reset',produced);return
        deadline_deltas={u.ue_id:[a-b for a,b in zip(self._deadline(u),current['deadlines'][u.ue_id])] for u in users if u.ue_id in URLLC}
        if any(v<0 for values in deadline_deltas.values() for v in values):self._invalidate('deadline_counter_reset',produced);return
        goodput=[delivered[uid]/duration/1e6 for uid in USERS]
        rates=[(deadline_deltas[uid][1]+deadline_deltas[uid][2]+deadline_deltas[uid][4])/duration for uid in URLLC]
        targets=[sum(goodput)]+goodput+rates
        self._log('measured_sample',action=current['action'],features=current['features'],targets=targets,
            produced_start_slot=current['start_slot'],produced_end_slot=produced,duration_s=duration,
            delivered_bits_by_ue=delivered,offered_bits_by_ue=offered,
            deadline_misses_by_ue={u:v[1] for u,v in deadline_deltas.items()},
            admitted_packets_by_ue={u:v[0] for u,v in deadline_deltas.items()},
            overflow_packets_by_ue={u:v[2] for u,v in deadline_deltas.items()},
            completed_packets_by_ue={u:v[3] for u,v in deadline_deltas.items()},
            late_completed_packets_by_ue={u:v[4] for u,v in deadline_deltas.items()},
            guard_label_scope='conservative_failure_event_sum_per_s_not_unique_packet_probability',
            deadline_counter_start_slots=current['deadline_start_slots'],
            deadline_counter_end_slots={u.ue_id:u.kpi_metadata.get('admitted_packets_total',{}).get('produced_at_slot',produced) for u in users if u.ue_id in URLLC},
            urllc_original_latency_budgets_ms={u.ue_id:u.latency_budget_ms for u in users if u.ue_id in URLLC},
            urllc_max_observed_hol_ms=current['max_hol'],membership=current['membership'],
            execution_confirmation='all_cells_modes_and_all_served_uid_weights_owned_and_executed',snapshot=current['snapshot'])
        reference=self._reference()
        self.last_guard_bad=bool(reference and any(v>reference[7+i]+self._guard_rate_tolerance() for i,v in enumerate(rates)))
        self.last_measured_action=current['action']
        if self.mode=='online' and self.policy=='guarded' and not self._coverage():
            if current['action']==0 and self.bootstrap_cooldown_pending:
                self.bootstrap_cooldown_pending=False
                self._log('bootstrap_cooldown_released',measured_action=0,produced_slot=produced,
                    rr_window_guard_bad=self.last_guard_bad,
                    reason='One real RR cooldown measured; baseline failure does not establish causality for untried arms')
            elif current['action']!=0 and self.last_guard_bad:
                self.bootstrap_cooldown_pending=True
                self._log('bootstrap_cooldown_required',measured_action=current['action'],produced_slot=produced,
                    reason='Bad weighted window retained; require one measured native RR cooldown')
        if self.mode=='online' and self.learner.sample_count<MAX_ROWS:
            self.learner.add(current['features'],targets)
            if self._coverage() and (not self.learner.trained or not self.learner.optimizer_verified or self.learner.sample_count-self.learner.trained_samples>=4):
                if self.learner.fit():self._log('mlp_fit',samples=self.learner.trained_samples,fit_count=self.learner.fit_count,
                    fit_parameters=deepcopy(self.learner.fit_parameters),fit_diagnostics=deepcopy(self.learner.fit_diagnostics),
                    optimizer_metadata=deepcopy(self.learner.optimizer_metadata))
            if self.model_path:self.learner.save(self.model_path)
        self.pending=None

    def run(self,observation,policy_snapshot):
        started=time.perf_counter();self.last_decisions=[];actions=[]
        users=sorted(observation.ue_observations,key=lambda u:u.ue_id);cells={c.cell_id:c for c in observation.cell_observations}
        reason=None
        if tuple(u.ue_id for u in users)!=USERS or self.active_cells!=set(self.allowed_cells) or any(u.serving_cell_id not in self.active_cells for u in users):reason='whole_six_ue_network_scope_required'
        elif any(cid not in cells for cid in self.allowed_cells):reason='missing_managed_cell'
        try:
            if reason is None:
                stamps=[cells[cid].kpi_metadata.get('executed_config',{}).get('produced_at_slot') for cid in self.allowed_cells]
                produced=stamps[0]
                if any(cells[cid].kpi_metadata.get('executed_config',{}).get('valid') is False for cid in self.allowed_cells):raise ValueError('invalid_cell_config_report')
                if type(produced) is not int or any(s!=produced for s in stamps) or produced>observation.slot_id or (observation.slot_id-produced)*self.dt>100:raise ValueError('invalid_common_report_time')
                for u in users:
                    for name in ('queue_bytes','hol_delay_ms','reference_sinr_db','throughput_mbps','offered_rate_bps','delivered_bits_total','offered_bits_total'):
                        v=getattr(u,name,None)
                        if type(v) not in (int,float) or not math.isfinite(v):raise ValueError('missing_or_nonfinite_public_'+name)
                    for name in ('delivered_bits_total','offered_bits_total'):
                        meta=u.kpi_metadata.get(name,{})
                        if meta.get('produced_at_slot')!=produced or meta.get('valid') is False:raise ValueError('spliced_or_invalid_cumulative_report')
                    if u.ue_id in URLLC:
                        if u.slice_id.lower()!='urllc' or u.latency_budget_ms!=10.:raise ValueError('original_urllc_profile_mismatch')
                        self._deadline(u)
                        for key in PACKET_COUNTERS:
                            meta=u.kpi_metadata.get(key,{})
                            if meta.get('produced_at_slot')!=produced or meta.get('valid') is False:raise ValueError('invalid_or_spliced_original_packet_counter')
        except (ValueError,AttributeError,TypeError) as exc:reason=str(exc)
        if reason:self._unavailable(reason)
        else:
            owners=getattr(observation,'control_ownership',{})
            self._observe(users,cells,produced,owners)
            if self.pending is None:
                index,prediction,status,why=self._choose(users)
                snapshot=action_map(users,index)
                for cid in self.allowed_cells:snapshot.setdefault(cid,dict(scheduler_mode='round_robin' if index==0 else 'latency_aware',ue_scheduler_weights={}))
                self.pending=dict(action=index,prediction=prediction,features=self._features(users,index),snapshot=deepcopy(snapshot),
                    membership=self._membership(users),requested_slot=observation.slot_id,confirmed=False)
                self.status,self.status_reason=status,why
                self.fixed_actions+=status=='fixed';self.rules_only_actions+=status=='rules_only'
                for cid,values in snapshot.items():
                    action=ConfigAction(source_name=self.name,target_cell_id=cid,**deepcopy(values));action._library_provenance=self.identity(cid);actions.append(action)
                decision=dict(slot_id=observation.slot_id,action=index,action_name=ACTIONS[index],status=status,reason=why,
                    features=deepcopy(self.pending['features']),
                    snapshot=deepcopy(snapshot),membership=self._membership(users),prediction=prediction,
                    effective_neutral=index!=0 and all(w==1. for cfg in snapshot.values() for w in cfg['ue_scheduler_weights'].values()),
                    neutralized_requested_arm=index>=2 and all(w==1. for cfg in snapshot.values() for w in cfg['ue_scheduler_weights'].values()),
                    prediction_guard_audit=deepcopy(self.last_prediction_audit),
                    guard_numeric_tolerance_events_per_window=GUARD_NUMERIC_EVENT_TOLERANCE,
                    guard_numeric_tolerance_event_rate_per_s=self._guard_rate_tolerance(),
                    policy=self.policy,constraints_active=self.constraints_active,
                    instance_id=self.name,implementation_id=self.implementation_id)
                self.last_decisions.append(decision);self._log('decision',**decision)
        self.decision_seconds+=time.perf_counter()-started
        result=ControlDecision(config_actions=actions)
        if not actions:result.config_actions.clear()
        return result

    def reset(self):super().reset();self.pending=None

    def summary(self):
        return dict(mode=self.mode,policy=self.policy,constraints_active=self.constraints_active,
            status=self.status,status_reason=self.status_reason,algorithm='MLPRegressor',
            scheduler_mapping='latency_aware_base_scores_plus_UID_weights_and_existing_NR_packing',
            samples=self.learner.sample_count,trained_samples=self.learner.trained_samples,fit_count=self.learner.fit_count,
            action_samples=self.action_samples(),fitted_action_samples=self.action_samples(self.learner.trained_samples),
            minimum_measured_rows=MIN_ROWS,maximum_measured_rows=MAX_ROWS,minimum_rows_per_action=6,
            prediction_batches=self.learner.prediction_batches,mlp_selected_actions=self.mlp_selected_actions,
            rules_only_actions=self.rules_only_actions,fixed_actions=self.fixed_actions,invalidated_samples=self.invalidated,
            last_measured_action=self.last_measured_action,bootstrap_cooldown_pending=self.bootstrap_cooldown_pending,
            fit_parameters=deepcopy(self.learner.fit_parameters),fit_diagnostics=deepcopy(self.learner.fit_diagnostics),
            optimizer_metadata=deepcopy(self.learner.optimizer_metadata),optimizer_verified=self.learner.optimizer_verified,
            guard_numeric_tolerance_events_per_window=GUARD_NUMERIC_EVENT_TOLERANCE,
            guard_numeric_tolerance_event_rate_per_s=self._guard_rate_tolerance(),
            experimental_horizon='Cumulative failure input grows with run length; deployment beyond observed training horizons can be out of distribution.',
            guard_reference=self._reference(),guard_scope=('observed_RR_training_reference_heuristic_not_matched_OFF_or_calibrated_SLA_bound' if self.policy=='guarded' else 'throughput_only_research_QoS_recorded_not_constrained'),
            fit_s=self.learner.fit_s,predict_s=self.learner.predict_s,decision_s=self.decision_seconds,
            model_path=str(self.model_path) if self.model_path else None,model_sha256=self.learner.artifact_sha256)

    def descriptor(self):return dict(super().descriptor(),learning=self.summary(),target_ue_ids=list(USERS),
        original_urllc_ue_ids=list(URLLC),actions=list(ACTIONS))
