"""Read-only paired summaries; no training code consumes development/heldout."""
from datetime import datetime,timezone,timedelta
import hashlib,json,math
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]
KEY='kpi:mec_offloaded_latency_ms'

def records():return [json.loads(p.read_text()) for p in OUT.glob('workers/*/*/metrics.json')]

def task_keys(row,length):
 return sorted((uid,t['task_key'],t['created_s'],t['deadline_s'],t['offloaded'])
  for uid in ('ue_0','ue_4') for t in row['windows'][str(length)]['per_ue'][uid]['tasks'])

def followed_cohort(row,length):
 end=row['case'].get('slots',row['case']['origin_slot']+length)
 tasks=[t for uid in ('ue_0','ue_4') for t in row['windows'][str(length)]['per_ue'][uid]['tasks']]
 completed=[t['terminal']['latency_ms'] for t in tasks if t.get('terminal',{}).get('kind')=='task_completed' and t['terminal']['recorded_slot']<=end]
 failed=sum(t.get('terminal',{}).get('kind')in('task_queue_overflow','task_result_overflow') and t['terminal']['recorded_slot']<=end for t in tasks)
 pending=len(tasks)-len(completed)-failed
 return dict(value=sum(completed)/len(completed) if completed and not pending else None,sample_count=len(completed),
  generated=len(tasks),failed=failed,pending=pending,followup_until_slot=end,
  measurement_basis='fixed_birth_cohort_actual_terminal_followup',
  last_observed_terminal_slot=max((t.get('terminal',{}).get('recorded_slot',0) for t in tasks),default=0))

def interval(off,on,length):
 raw_a=off['windows'][str(length)]['aggregate']['states'][KEY];raw_b=on['windows'][str(length)]['aggregate']['states'][KEY]
 a,b=followed_cohort(off,length),followed_cohort(on,length)
 valid=isinstance(a.get('value'),(float,int)) and a['value']>0 and isinstance(b.get('value'),(float,int))
 def restricted(row):
  end=row['case']['origin_slot']+length
  tasks=[t for uid in ('ue_0','ue_4') for t in row['windows'][str(length)]['per_ue'][uid]['tasks']]
  if not tasks:return None
  # Every eligible task has reached its100ms deadline by the endpoint.
  # This is observed restricted time-to-success, not an imputed late latency.
  def cost(t):
   terminal=t.get('terminal',{})
   return min(100.,terminal['latency_ms']) if terminal.get('kind')=='task_completed' and terminal['recorded_slot']<=end else 100.
  return sum(cost(t) for t in tasks)/len(tasks)
 cost_off,cost_on=restricted(off),restricted(on)
 return dict(off_ms=a['value'],on_ms=b['value'],gain_percent=100.*(a['value']-b['value'])/a['value'] if valid else None,
  off_sample_count=a.get('sample_count',0),on_sample_count=b.get('sample_count',0),off_failed=a.get('failed',0),on_failed=b.get('failed',0),
  off_pending=a.get('pending',0),on_pending=b.get('pending',0),task_identities_equal=task_keys(off,length)==task_keys(on,length),
  unavailable_reason=None if valid else 'unresolved_or_empty_followed_cohort',
  measurement_basis='fixed_birth_cohort_actual_terminal_followup',
  off_followup_until_slot=a['followup_until_slot'],on_followup_until_slot=b['followup_until_slot'],
  off_window_ms=raw_a['value'],on_window_ms=raw_b['value'],
  off_window_pending=raw_a.get('pending',0),on_window_pending=raw_b.get('pending',0),
  off_window_sample_count=raw_a.get('sample_count',0),on_window_sample_count=raw_b.get('sample_count',0),
  restricted_completion_horizon_ms=100.,off_restricted_completion_ms=cost_off,on_restricted_completion_ms=cost_on,
  restricted_completion_reduction_percent=100.*(cost_off-cost_on)/cost_off if cost_off and cost_on is not None else None)

def comparisons(split=None):
 rows=records();refs={};byname={r['case']['name']:r for r in rows}
 for r in rows:
  c=r['case']
  if c['arm']=='OFF':refs[c['worker'],c['seed'],c['origin_slot'],c['split']]=r
 results=[]
 for r in rows:
  c=r['case'];key=c['worker'],c['seed'],c['origin_slot'],c['split']
  if c['arm']=='OFF' or split and c['split']!=split:continue
  off=byname.get(c.get('reference_off_case')) if c.get('reference_off_case') else refs.get(key)
  if off is None:continue
  if off['case']['arm']!='OFF' or any(off['case'][k]!=c[k] for k in ('worker','seed','origin_slot')):raise ValueError('Explicit OFF reference differs in seed/host/native origin')
  lengths=sorted(set(map(int,off['windows']))&set(map(int,r['windows'])))
  anchor=r['before']['states'][KEY]['value']
  results.append(dict(case=c['name'],seed=c['seed'],worker=c['worker'],origin_slot=c['origin_slot'],split=c['split'],
   actor=c.get('actor',c.get('policy',c['arm'])),off_case=off['case']['name'],
   prefix_equal=off['network_prefix_sha256']==r['network_prefix_sha256'],
   model_sha256=c.get('actual_policy_sha256'),execution=r['execution_audit'],
   before_intent_ms=anchor,
   windows={str(n):dict(interval(off,r,n),before_intent_ms=anchor,
    pre_intent_relative_gain_percent=100.*(anchor-r['windows'][str(n)]['aggregate']['states'][KEY]['value'])/anchor
      if anchor and isinstance(r['windows'][str(n)]['aggregate']['states'][KEY]['value'],(float,int)) else None)
    for n in lengths},
   simulation_s=r['simulation_s'],off_simulation_s=off['simulation_s']))
 return results

def main():
 rows=comparisons();path=OUT/'analysis/paired-comparisons.json';tmp=path.with_suffix('.tmp')
 tmp.write_text(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(),comparisons=rows),indent=2,allow_nan=False)+'\n');tmp.replace(path)
 for r in rows:
  if r['split'] in ('development','diagnostic','holdout','hot_start_validation'):
   print(json.dumps(dict(case=r['case'],windows={k:{m:v.get(m) for m in ('off_ms','on_ms','gain_percent','off_pending','on_pending')} for k,v in r['windows'].items()},prefix_equal=r['prefix_equal'])))
if __name__=='__main__':main()
