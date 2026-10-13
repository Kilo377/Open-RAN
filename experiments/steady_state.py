"""Warmup-separated offered/admitted/delivered load and queue growth diagnostics."""
import numpy as np


class SteadyStateMonitor:
    def __init__(self,users,warmup_ms,slot_duration_ms,window_ms=100):
        self.warmup=warmup_ms;self.dt=slot_duration_ms;self.window=window_ms
        self.baseline={u.ue_id:u.packet_ledger.summary() for u in users}
        self.last=self.baseline.copy();self.latencies={u.ue_id:[] for u in users}
        self.windows=[];self.window_start=warmup_ms;self.window_base=None;self.duration=0.
        self.scheduled={u.ue_id:0 for u in users};self.no_path={u.ue_id:0 for u in users}
        self.capacity_outage={u.ue_id:0 for u in users}
        self.target_unmet={u.ue_id:0 for u in users}

    def observe(self,time_ms,users,trace,events):
        current={u.ue_id:u.packet_ledger.summary() for u in users}
        if time_ms<=self.warmup:
            self.baseline=current;self.last=current;return
        if self.window_base is None:self.window_base=self.baseline
        self.duration+=self.dt
        for row in trace['ues']:
            uid=row['ue_id'];self.scheduled[uid]+=bool(row.get('scheduled',False))
            self.no_path[uid]+=row.get('ack_source')=='no_path'
            self.capacity_outage[uid]+=row.get('ack_source')=='capacity_outage_no_tb'
            self.target_unmet[uid]+=row.get('amc_selection_target_met') is False
        for e in events:
            if e['kind']=='delivery' and e.get('completed'):
                # Cohort arrivals at/after the warmup boundary only.
                if time_ms-e['latency_ms']>=self.warmup:self.latencies[e['ue_id']].append(e['latency_ms'])
        self.last=current
        if time_ms-self.window_start>=self.window-1e-9:
            self.windows.append(dict(start_ms=self.window_start,end_ms=time_ms,
                offered_bytes=sum(m['offered_bytes']-self.window_base[uid]['offered_bytes'] for uid,m in current.items()),
                delivered_bytes=sum(m['delivered_bytes']-self.window_base[uid]['delivered_bytes'] for uid,m in current.items()),
                pending_bytes=sum(m['pending_bytes'] for m in current.values())))
            self.window_start=time_ms;self.window_base=current

    def summary(self):
        result={}
        for uid,last in self.last.items():
            base=self.baseline[uid];duration=max(self.duration,1e-9)
            offered=last['offered_bytes']-base['offered_bytes'];lost=last['overflow_bytes']-base['overflow_bytes']
            delivered=last['delivered_bytes']-base['delivered_bytes'];growth=last['pending_bytes']-base['pending_bytes']
            assert offered-lost-delivered==growth,(uid,offered,lost,delivered,growth)
            lat=self.latencies[uid]
            result[uid]=dict(offered_bytes=offered,admitted_bytes=offered-lost,delivered_bytes=delivered,
                overflow_bytes=lost,pending_start=base['pending_bytes'],pending_end=last['pending_bytes'],
                queue_growth_bytes=growth,queue_growth_bytes_per_s=growth*1000/duration,
                offered_mbps=offered*.008/duration,delivered_mbps=delivered*.008/duration,
                post_warmup_completed_packets=len(lat),p95_completion_latency_ms=float(np.percentile(lat,95)) if lat else None,
                deadline_misses=last['deadline_missed_packets']-base['deadline_missed_packets'],
                scheduled_slots=self.scheduled[uid],no_path_slots=self.no_path[uid],capacity_outage_slots=self.capacity_outage[uid],amc_target_unmet=self.target_unmet[uid],
                diagnosis='weak_coverage' if self.no_path[uid] else 'equivalent_link_capacity_outage' if self.capacity_outage[uid] else 'never_scheduled' if offered and not self.scheduled[uid]
                    else 'queue_growing' if growth>0 else 'no_positive_queue_growth')
        return dict(warmup_ms=self.warmup,evaluated_ms=self.duration,per_ue=result,windows=self.windows,
            interpretation='Queue slope and cohort delays diagnose finite baseline; no long-term stability or QoS guarantee from one run')
