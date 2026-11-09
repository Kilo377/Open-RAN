"""X-13-inspired expected-workload guard; UE routing is not server rejection.

The known workload budget is analytic. A measured-data lower utility quantile
only ranks feasible probabilities; it is not a calibrated safety bound.
"""
from collections import deque
from copy import deepcopy
import json
import math
from pathlib import Path
import time

from models.action import ConfigAction, ControlDecision
from ric.near_rt.pools.base import PoolXApp
from .online_qgbdt import OnlineQGBDT, MAX_ROWS

ROOT = Path(__file__).resolve().parents[4]
TARGET = 'ue_4'
CANDIDATES = (0., .5, 1.)
FEATURES = ['task_arrival_rate_per_s', 'pending_offloaded_tasks', 'queue_kb',
            'hol_delay_ms', 'reference_sinr_db', 'recent_all_task_bad_rate',
            'recent_all_task_mean_ms_or_deadline', 'candidate_offload_fraction']


class MecOffloadBudgetQGBDT(PoolXApp):
    implementation_id = 'mec_offload_budget_qgbdt'
    category_id = 'mec_offload_budget'
    description = 'MEC expected-workload guard with measured lower-quantile utility'
    required_kpis = ('queue_bytes', 'hol_delay_ms', 'reference_sinr_db',
                     'ue_service_policy', 'ue_energy_total_j', 'executed_config')
    writable_actions = ('ue_offload_fraction',)
    min_action_ms = 100

    def validate_parameters(self, p):
        super().validate_parameters(p)
        if set(p) - {'mode','model_path','log_path','window_ms','seed','budget_gflops','task_profile'}:
            raise ValueError('Unknown MEC budget QGBDT parameters')
        if p.get('mode','online') not in ('online','frozen','rules'): raise ValueError('Invalid budget mode')
        if p.get('window_ms',100) != 100: raise ValueError('Budget task cohorts must be 100 ms')
        seed = p.get('seed',137)
        if type(seed) is not int or not 0 <= seed <= 2**32-1: raise ValueError('Invalid budget seed')
        budget = p.get('budget_gflops',.75)
        if type(budget) not in (int,float) or not math.isfinite(budget) or budget < 0: raise ValueError('Invalid expected-work budget')
        profile = p.get('task_profile')
        if profile is not None:
            expected = dict(interval_ms=50.,compute_gflop=.05,deadline_ms=100.)
            if not isinstance(profile,dict) or set(profile) != set(expected): raise ValueError('Invalid task profile schema')
            if any(type(profile[k]) not in (int,float) or not math.isfinite(profile[k]) or not math.isclose(profile[k],v,rel_tol=0.,abs_tol=1e-10) for k,v in expected.items()):
                raise ValueError('Budget case requires validated 20 tasks/s, .05 GFLOP/task, 100 ms deadline')
        for key in ('model_path','log_path'):
            if p.get(key) is not None and not Path(p[key]).resolve().is_relative_to(ROOT/'outputs'):
                raise ValueError('Budget artifacts must stay under project outputs')
        if p.get('mode') == 'frozen' and not p.get('model_path'): raise ValueError('Frozen budget app requires a trained model')

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.mode = self.parameters.get('mode','online')
        self.profile = deepcopy(self.parameters.get('task_profile'))
        self.budget = float(self.parameters.get('budget_gflops',.75))
        self.model_path = Path(self.parameters['model_path']) if self.parameters.get('model_path') else None
        self.log_path = Path(self.parameters['log_path']) if self.parameters.get('log_path') else None
        self.learner = OnlineQGBDT(FEATURES,self.parameters.get('seed',137))
        if self.model_path and self.model_path.is_file(): self.learner = OnlineQGBDT.load(self.model_path,FEATURES)
        self.pending = None
        self.events = deque(maxlen=256)
        self.invalidated = self.rules_only_actions = self.qgbdt_selected_actions = 0
        self.status = 'not_evaluable'
        self.status_reason = 'no_valid_delivered_observation'
        if self.mode == 'frozen' and (not self.learner.trained or not self._coverage()):
            raise ValueError('Frozen budget model requires 12 measured rows and six per feasible action')
        if self.log_path:
            self.log_path.parent.mkdir(parents=True,exist_ok=True)
            if self.log_path.exists(): raise ValueError('Choose a fresh budget QGBDT event log')

    def feasible(self):
        if self.profile is None: return ()
        rate = 1000/self.profile['interval_ms']
        return tuple(p for p in CANDIDATES if rate*self.profile['compute_gflop']*p <= self.budget+1e-12)

    def action_samples(self):
        return {format(p,'g'):sum(row[-1] == p for row in self.learner._x) for p in self.feasible()}

    def _coverage(self):
        counts = self.action_samples()
        return self.learner.sample_count >= 12 and bool(counts) and all(n >= 6 for n in counts.values())

    def _log(self,kind,**values):
        row = dict(kind=kind,mode=self.mode,**values); self.events.append(row)
        if self.log_path:
            with self.log_path.open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')

    def _invalidate(self,reason,slot=None):
        if self.pending is not None:
            self.pending = None; self.invalidated += 1
            self._log('sample_invalidated',reason=reason,produced_slot=slot)

    def _unavailable(self,reason):
        self._invalidate(reason)
        if self.status != 'not_evaluable' or self.status_reason != reason:
            self._log('not_evaluable',reason=reason,persistence='No new action; existing lease is not rolled back')
        self.status,self.status_reason = 'not_evaluable',reason

    @staticmethod
    def _valid_cohort(c):
        if not isinstance(c,dict) or type(c.get('index')) is not int or c['index'] < 0: return False
        counts=[c.get(k) for k in ('generated','completed','on_time_completed','failed','pending')]
        if any(type(v) is not int or v < 0 for v in counts): return False
        n,completed,on_time,failed,pending=counts
        if completed+failed+pending != n or on_time > completed: return False
        for key in ('completion_latency_ms_sum','mature_after_s'):
            value=c.get(key)
            if type(value) not in (int,float) or not math.isfinite(value) or value < 0: return False
        p=c.get('offload_fraction')
        return p is None or (type(p) in (int,float) and math.isfinite(p) and 0 <= p <= 1)

    def _features(self,u,produced,p):
        now = produced*self.dt/1000
        mature = [c for c in u.ue_task_cohorts if c.get('generated',0)>0 and c.get('mature_after_s',math.inf)<=now+1e-9]
        recent = max(mature,key=lambda c:c['index']) if mature else None
        bad = 1-recent['on_time_completed']/recent['generated'] if recent else 0.
        mean = recent['completion_latency_ms_sum']/recent['completed'] if recent and recent['completed'] else self.profile['deadline_ms']
        return [1000/self.profile['interval_ms'],u.ue_task_metrics['offloaded_pending'],u.queue_bytes/1000,
                u.hol_delay_ms,u.reference_sinr_db,bad,mean,p]

    def _choose(self,u,produced):
        feasible = self.feasible()
        if self.mode == 'online' and self.learner.sample_count < MAX_ROWS and not self._coverage():
            counts = self.action_samples()
            return min(feasible,key=lambda p:(counts[format(p,'g')],-p)),None,'rules_only','balanced_measured_exploration'
        if self.mode != 'rules' and self.learner.trained:
            predictions = self.learner.predict([self._features(u,produced,p) for p in feasible])
            best = max(predictions)
            p = max(p for p,value in zip(feasible,predictions) if value >= best-1e-9)
            return p,predictions[feasible.index(p)],'qgbdt','maximum_predicted_lower_utility_quantile'
        return max(feasible),None,'rules_only','largest_analytically_feasible_probability'

    def _observe(self,u,cell,produced,owner):
        current = self.pending
        if current is None: return
        if u.serving_cell_id != current['cell_id']:
            self._invalidate('handover_or_scope_changed',produced); return
        if produced < current['requested_slot']: return
        probabilities = cell.executed_config.get('ue_offload_fraction',{})
        actual = probabilities.get(TARGET) if isinstance(probabilities,dict) else None
        policy = u.ue_service_policy
        if actual != current['p'] or u.ue_service_policy.get('offload_fraction') != current['p'] or owner != self.name:
            if current.get('confirmed') or produced-current['requested_slot'] > round(200/self.dt):
                self._invalidate('executed_probability_or_uid_owner_changed',produced)
            return
        now = produced*self.dt/1000
        if not current.get('confirmed'):
            current.update(confirmed=True,cohort=math.floor(now/.1+1e-8)+1)
            self._log('executed_policy_confirmed',ue_id=TARGET,cell_id=current['cell_id'],
                produced_slot=produced,p=current['p'],cohort=current['cohort'],owner=owner,
                executed_policy=deepcopy(policy),ownership_scope='ue_offload_fraction_by_ue')
        cohort = next((c for c in u.ue_task_cohorts if c['index']==current['cohort']),None)
        if cohort is None:
            if now > current['cohort']*.1+.6: self._invalidate('missing_future_cohort',produced)
            return
        # Only our offload routing policy must be stable. DRX may be owned by X01;
        # never demand its ownership or reject a cohort solely for mixed DRX.
        if cohort.get('offload_fraction') != current['p']:
            self._invalidate('mixed_offload_probability_in_cohort',produced); return
        if now+1e-9 < cohort['mature_after_s']: return
        n = cohort['generated']
        if n == 0:
            self._invalidate('zero_generated_cohort',produced); return
        if (n != 2 or cohort['completed']+cohort['failed']+cohort['pending'] != n or
                not 0 <= cohort['on_time_completed'] <= cohort['completed']):
            self._unavailable('task_profile_or_outcome_mismatch'); return False
        deadline = self.profile['deadline_ms']
        # Maturity covers end-of-arrival-window plus the maximum task deadline.
        if cohort['mature_after_s']+1e-9 < (current['cohort']+1)*.1+deadline/1000:
            self._unavailable('incomplete_deadline_horizon'); return False
        penalty = (cohort['completion_latency_ms_sum']+deadline*(cohort['failed']+cohort['pending']))/n
        utility = cohort['on_time_completed']/n-.25*penalty/deadline
        self._log('measured_sample',ue_id=TARGET,cell_id=current['cell_id'],p=current['p'],
            features=current['features'],utility=utility,cohort=deepcopy(cohort),
            deadline_penalized_mean_ms=penalty,all_task_on_time_ratio=cohort['on_time_completed']/n,
            prediction_before_outcome=current['prediction'],produced_slot=produced,
            confirmation='executed_uid_offload_policy_and_ownership')
        if self.mode == 'online' and self.learner.sample_count < MAX_ROWS:
            self.learner.add(current['features'],utility)
            if self._coverage() and (not self.learner.trained or self.learner.sample_count-self.learner.trained_samples>=4):
                if self.learner.fit(): self._log('qgbdt_fit',samples=self.learner.trained_samples,
                    fit_count=self.learner.fit_count,fit_parameters=deepcopy(self.learner.fit_parameters))
            # Even insufficient, untrained rows survive a bounded continuation.
            if self.model_path: self.learner.save(self.model_path)
        self.pending = None

    def run(self,observation,policy_snapshot):
        started=time.perf_counter();self.last_decisions=[];actions=[]
        users=[u for u in observation.ue_observations if u.ue_id==TARGET and u.serving_cell_id in self.active_cells]
        cells={c.cell_id:c for c in observation.cell_observations}
        reason=None
        if self.profile is None: reason='validated_manifest_task_profile_missing'
        elif len(users)!=1 or users[0].serving_cell_id not in cells: reason='target_outside_active_scope'
        if reason is None:
            u=users[0];cell=cells[u.serving_cell_id]
            meta=cell.kpi_metadata.get('executed_config',{});produced=meta.get('produced_at_slot')
            if (type(produced) is not int or produced>observation.slot_id or meta.get('valid') is False or
                    observation.slot_id-produced>round(100/self.dt) or not isinstance(getattr(cell,'executed_config',None),dict)):
                reason='invalid_or_stale_executed_policy_report'
            elif (not isinstance(getattr(u,'ue_task_metrics',None),dict) or 'offloaded_pending' not in u.ue_task_metrics or
                    not isinstance(getattr(u,'ue_task_cohorts',None),list) or not isinstance(getattr(u,'ue_service_policy',None),dict)):
                reason='missing_public_task_telemetry'
            elif (type(u.ue_task_metrics['offloaded_pending']) is not int or u.ue_task_metrics['offloaded_pending'] < 0 or
                  any(not self._valid_cohort(c) for c in u.ue_task_cohorts)):
                reason='invalid_public_task_outcome'
            elif any(type(getattr(u,k,None)) not in (int,float) or not math.isfinite(getattr(u,k)) for k in ('queue_bytes','hol_delay_ms','reference_sinr_db')):
                reason='missing_or_nonfinite_feature'
            elif u.kpi_metadata.get('ue_energy_total_j',{}).get('produced_at_slot') != produced:
                reason='spliced_public_reports'
            else:
                taskmeta=u.kpi_metadata.get('ue_task_metrics',{})
                if taskmeta and (taskmeta.get('produced_at_slot')!=produced or taskmeta.get('valid') is False): reason='invalid_task_report'
        if reason:
            self._unavailable(reason)
        else:
            owners=getattr(observation,'control_ownership',{}).get(u.serving_cell_id,{})
            by_uid=owners.get('ue_offload_fraction_by_ue',{})
            owner=by_uid.get(TARGET,owners.get('ue_offload_fraction'))
            evaluable=self._observe(u,cell,produced,owner)
            if self.pending is None and evaluable is not False:
                p,prediction,status,why=self._choose(u,produced)
                self.status,self.status_reason=status,why
                self.rules_only_actions+=status=='rules_only';self.qgbdt_selected_actions+=status=='qgbdt'
                features=self._features(u,produced,p)
                self.pending=dict(p=p,prediction=prediction,features=features,cell_id=u.serving_cell_id,
                                  requested_slot=observation.slot_id,confirmed=False)
                action=ConfigAction(source_name=self.name,target_cell_id=u.serving_cell_id,ue_offload_fraction={TARGET:p})
                action._library_provenance=self.identity(u.serving_cell_id);actions.append(action)
                decision=dict(slot_id=observation.slot_id,ue_id=TARGET,p=p,status=status,reason=why,
                    expected_offloaded_compute_demand_gflops=1000/self.profile['interval_ms']*self.profile['compute_gflop']*p,
                    feasible_candidates=list(self.feasible()),prediction=prediction,**self.identity(u.serving_cell_id))
                self.last_decisions.append(decision);self._log('decision',**decision)
        self.decision_seconds+=time.perf_counter()-started
        result=ControlDecision(config_actions=actions)
        if not actions:result.config_actions.clear()
        return result

    def reset(self):
        super().reset();self.pending=None

    def summary(self):
        return dict(mode=self.mode,status=self.status,status_reason=self.status_reason,
            algorithm='GradientBoostingRegressor_quantile',target_ue_ids=[TARGET],
            budget_semantics='expected offered compute demand, not a hard realized admission quota',
            budget_gflops=self.budget,feasible_candidates=list(self.feasible()),
            samples=self.learner.sample_count,trained_samples=self.learner.trained_samples,
            action_samples=self.action_samples(),minimum_measured_rows=12,minimum_rows_per_action=6,maximum_measured_rows=MAX_ROWS,
            fit_count=self.learner.fit_count,fit_parameters=deepcopy(self.learner.fit_parameters),
            prediction_batches=self.learner.prediction_batches,qgbdt_selected_actions=self.qgbdt_selected_actions,
            rules_only_actions=self.rules_only_actions,invalidated_samples=self.invalidated,
            fit_s=self.learner.fit_s,predict_s=self.learner.predict_s,decision_s=self.decision_seconds,
            model_path=str(self.model_path) if self.model_path else None,model_sha256=self.learner.artifact_sha256)

    def descriptor(self):
        return dict(super().descriptor(),learning=self.summary(),target_ue_ids=[TARGET],
                    target_required_kpis=['ue_task_metrics','ue_task_cohorts'],
                    configuration_requirement='validated manifest UE4 task_profile')
