"""Read-only before/after intent assessment, anchored to actual application.

Only explicitly requested KPIs participate. Recorded observation points are
correlated reports, not independent trials or an estimate of causal benefit.
"""
from copy import deepcopy
import json
from pathlib import Path
from threading import RLock
from uuid import uuid4

from smo.intent_language import IntentError
from smo.intent_ledger import intersect_intents
from smo.runtime import APP_LOGS, number
from smo.intent_quality import assess, score, normalization, quality, quantify, VALID, enrich_comparison


def application_evidence(frames, baseline, through):
    """Proposals, enabled badges and configured values are never receipts."""
    found = {}
    for frame in frames:
        if frame.get('slot', -1) > through:
            continue
        for ident in APP_LOGS:
            receipt = frame.get('states', {}).get('xapp:'+ident, {}).get('last_applied')
            if not receipt or receipt.get('status') != 'applied':
                continue
            at = receipt.get('slot')
            if type(at) is int and baseline < at <= min(through, frame.get('slot', -1)):
                key = (ident, at)
                found[key] = dict(app_id='xapp:'+ident, **deepcopy(receipt))
    return sorted(found.values(), key=lambda r:(r['slot'],r['app_id']))


class IntentComparison:
    def __init__(self, context, path=None):
        self.context, self.path = context, Path(path) if path else None
        self.sessions, self.lock = {}, RLock()
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text())
            if data.get('schema') != 'oran.intent-comparisons.v1' or data.get('context') != context:
                raise ValueError('Comparison records belong to a different network context')
            self.sessions = data['sessions']

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name+'.tmp')
            tmp.write_text(json.dumps(dict(schema='oran.intent-comparisons.v1',
                context=self.context, sessions=self.sessions), ensure_ascii=False, allow_nan=False, indent=2)+'\n')
            tmp.replace(self.path)

    def start(self, records, graph, frame, observe_slots=500, source_id=None):
        if type(observe_slots) is not int or not 1 <= observe_slots <= 20000:
            raise IntentError('观察长度须为 1–20,000 个整数 slots。')
        if not records or type(frame.get('slot')) is not int:
            raise IntentError('请先解析意图，并选择真实网络快照。')
        ranges = intersect_intents(records, graph, frame)['ranges']
        keys = {'kpi:'+r['kpi'] for r in ranges}
        # The comparison archive contains only target KPI readings, not all KPIs.
        before = {k:deepcopy(frame.get(k)) for k in ('slot','time_ms')}
        before['states'] = {k:deepcopy(v) for k,v in frame.get('states', {}).items() if k in keys}
        session = dict(id=uuid4().hex, intent_ids=[r['id'] for r in records],
            records=deepcopy(records), ranges=ranges, before=before,
            observe_slots=observe_slots, evidence=[], source_id=source_id)
        with self.lock:
            self.sessions[session['id']] = session
            self.save()
        return session['id']

    def review(self, ident, graph, source, current, *, windows=None):
        with self.lock:
            if not isinstance(ident,str)or ident not in self.sessions:
                raise IntentError('部署前快照不可用，请重新记录。')
            session=self.sessions[ident]
            if session.get('source_id')!=(source.get('run_id')or source.get('source_sha256')):
                raise IntentError('网络运行已变化，请重新记录部署前快照。')
            at,baseline=current.get('slot'),session['before']['slot']
            if type(at)is not int:raise IntentError('当前没有真实观测。')
            length=session['observe_slots'];keys=[r['kpi']for r in session['ranges']]
            def interval(start,end):
                if windows is not None:return windows.window(keys,start,end,through=at)
                return dict(slot=end,time_ms=end*source.get('slot_duration_ms',.5),window_start_slot=start,window_slots=end-start,
                    states={'kpi:'+k:dict(value=None,reason='exact_window_source_unavailable',window='matched_window')for k in keys})
            # Original point baselines/anchors remain untouched for audit. V2
            # derives both readings over precisely equal intervals instead.
            before=interval(baseline-length,baseline)
            scales={r['kpi']:normalization(r,before['states'].get('kpi:'+r['kpi'],{}))for r in session['ranges']}
            candidates=application_evidence(source.get('frames',[]),baseline,at)
            if windows is not None:
                candidates += [deepcopy(r)for r in windows.receipts.values()if baseline<r['slot']<=at]
            known={(r['app_id'],r['slot']):r for r in session['evidence']}
            for r in candidates:known[r['app_id'],r['slot']]=r
            retained=sorted(known.values(),key=lambda r:(r['slot'],r['app_id']))
            if retained!=session['evidence']:
                session['evidence']=retained;self.save()
            evidence=[r for r in retained if r['slot']<=at]
            applied=evidence[0]['slot']if evidence else None
            # C_e-C_s counts slots s+1..e. The first affected slot is included.
            end=applied+length-1 if applied is not None else None
            after=interval(applied-1,end)if end is not None and end<=at else None
            data_wait=after is not None and any(m.get('reason')=='exact_window_data_unavailable'for m in after['states'].values())
            if data_wait:after=None
            status='snapshot_before_baseline'if at<baseline else'awaiting_deployment'if applied is None else'observing_data'if data_wait else'observing'if after is None else'complete'
            before_score=score(session['ranges'],before,scales)
            after_score=score(session['ranges'],after,scales)if after else None
            if after_score and (before_score['valid']<before_score['required']or after_score['valid']<after_score['required']):
                status='complete_with_missing_kpis'
            if any(r.get('empty')or r['kpi']=='ue_energy_j'for r in session['ranges']):status='invalid_constraints'
            rows=[]
            for row in session['ranges']:
                scale=scales[row['kpi']];old=assess(row,before,scale);new=assess(row,after,scale)if after else None
                delta=new['value']-old['value']if new and new['status']in VALID and old['status']in VALID else None
                rows.append(dict(**deepcopy(row),before=old,after=new,delta=delta,normalization=scale,
                    quantification=quantify(row,old,new)))
            intents=[]
            for record in session['records']:
                local=intersect_intents([record],graph,session['before'])['ranges']
                old=score(local,before,scales);new=score(local,after,scales)if after else None
                intents.append(dict(id=record['id'],text=record['text'],before=old,after=new,quality=quality(old,new)))
            return enrich_comparison(dict(schema='oran.intent-comparison.v2',assessment_version=2,quantification_version=1,id=ident,
                intent_ids=session['intent_ids'],status=status,observe_slots=length,
                baseline={k:session['before'].get(k)for k in('slot','time_ms')},
                baseline_window={k:before.get(k)for k in('window_start_slot','slot','window_slots')},
                after_window={k:after.get(k)for k in('window_start_slot','slot','window_slots')}if after else None,
                application_slot=applied,target_end_slot=end,observed_after_slot=end if after else None,
                actual_elapsed_slots=length if after else None,remaining_slots=max(0,end-at)if end is not None else None,
                before=before_score,after=after_score,rows=rows,intents=intents,quality=quality(before_score,after_score),
                application_evidence=[r for r in evidence if end is None or r['slot']<=end],
                comparison_basis='exact_equal_windows_from_scalar_counters_and_mature_task_events',
                interpretation='short_term_response_including_initial_transition',
                deployment_enabled=False,selection_performed=False))
