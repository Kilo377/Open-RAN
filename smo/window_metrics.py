"""Exact, read-only interval metrics from recorded scalar counters/task events.

Compact arrays are retained once per run. No PHY tensors, controller decisions,
or cumulative task means enter this comparison index.
"""
from array import array
from bisect import bisect_left
import json
import math
from pathlib import Path
import zlib

from smo.population import task_members, urllc_members


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


class CounterColumn:
    def __init__(self):
        self.values, self.epochs = array('d'), array('I')
        self.last, self.epoch = None, 0

    def append(self, components):
        valid = bool(components) and all(numeric(v) and v >= 0 for v in components.values())
        valid = valid and math.isfinite(sum(components.values()))
        if not valid or self.last is None or components.keys() != self.last.keys() or any(
                not numeric(self.last.get(k)) or v < self.last[k] for k,v in components.items()):
            self.epoch += 1
        self.values.append(sum(components.values()) if valid else math.nan)
        self.epochs.append(self.epoch)
        self.last = dict(components) if valid else None

    def delta(self, a, b):
        x, y = self.values[a], self.values[b]
        return y-x if math.isfinite(x) and math.isfinite(y) and self.epochs[a] == self.epochs[b] and y >= x else None


class InstantColumn:
    def __init__(self):
        self.values, self.integral, self.valid = array('d'), array('d'), array('I')

    def append(self, value):
        ok = numeric(value)
        self.values.append(value if ok else math.nan)
        self.integral.append((self.integral[-1] if self.integral else 0.)+(value if ok else 0.))
        self.valid.append((self.valid[-1] if self.valid else 0)+int(ok))

    def interval(self, a, b):
        if self.valid[b]-self.valid[a] != b-a:
            return None
        values = self.values[a+1:b+1]
        return dict(value=(self.integral[b]-self.integral[a])/(b-a), minimum=min(values), maximum=max(values))


class WindowIndex:
    def __init__(self, config, *, population=True):
        self.config, self.population = config, population
        self.dt_ms = config.get('manifest', {}).get('slot_duration_ms', .5)
        self.dt = self.dt_ms/1000
        specs = config.get('manifest', {}).get('ues') or config.get('configs', {}).get('ue', {}).get('ues', [])
        self.uids = tuple(u['ue_id'] for u in specs)
        self.cell_ids = tuple(c['cell_id'] for c in config.get('manifest', {}).get('cells', []))
        self.task_uids = tuple(task_members(config))
        self.urllc_uids = tuple(urllc_members(config))
        self.profiles = config.get('configs', {}).get('phy', {}).get('rt_sys', {}).get('ue_services', {}).get('mec', {}).get('tasks_by_ue', {})
        self.slots = array('q')
        self.counters = {k:CounterColumn() for k in (
            'energy','ap_energy','delivered','failure','urllc_offered','active','ap_active','urllc_completed','urllc_latency',
            'packet_completed','packet_latency','overflow','task_generated','task_completed',
            'task_failed','task_on_time','mec_generated','mec_completed','mec_failed','mec_gflop')}
        self.instant = {k:InstantColumn() for k in ('expected','queue','hol')}
        self.tasks, self.task_errors, self.receipts = {}, [], {}

    def sync(self):
        """Recorded replay indexes are filled by the existing single pass."""
        return

    def observe_receipt(self, row):
        from smo.runtime import APP_LOGS
        at = row.get('apply_slot', row.get('slot_id'))
        if row.get('status') != 'applied' or type(at) is not int:
            return
        for p in row.get('provenance', []):
            ident = p.get('implementation_id')
            if ident in APP_LOGS:
                self.receipts[ident,at] = dict(app_id='xapp:'+ident, slot=at, status='applied',
                    fields=[f.get('field') for f in row.get('fields', [])], target_cell_ids=row.get('target_cell_ids', []))

    def observe(self, slot, users, services, ap, events=()):
        if self.slots and slot <= self.slots[-1]:
            raise ValueError('Window index requires increasing recorded slots')
        users = {u['ue_id']:u for u in users}
        if not self.uids:
            self.uids = tuple(users)
        if not self.cell_ids and ap:
            self.cell_ids = tuple(ap)
        self.slots.append(slot)
        def packet(uid, name):return users.get(uid, {}).get('packet', {}).get(name)
        def task(uid, name):return (services.get(uid, {}).get('ue_task_metrics') or {}).get(name)
        def components(ids, fn):return {uid:fn(uid) for uid in ids}
        data = dict(
            energy=components(self.uids,lambda u:services.get(u, {}).get('ue_energy_total_j')),
            ap_energy=components(self.cell_ids,lambda c:ap.get(c, {}).get('ap_energy_total_j')),
            delivered=components(self.uids,lambda u:packet(u,'delivered_bytes')),
            failure={u+':'+k:packet(u,k) for u in self.urllc_uids for k in ('deadline_failed_packets','overflow_packets')},
            urllc_offered={u+':'+k:packet(u,k) for u in self.urllc_uids for k in ('admitted_packets','overflow_packets')},
            active=components(self.uids,lambda u:round(services[u]['ue_active_fraction']*slot) if numeric(services.get(u,{}).get('ue_active_fraction')) and 0<=services[u]['ue_active_fraction']<=1 else None),
            ap_active=components(self.cell_ids,lambda c:ap.get(c,{}).get('ap_available_slots_total')),
            urllc_completed=components(self.urllc_uids,lambda u:packet(u,'completed_packets')),
            urllc_latency=components(self.urllc_uids,lambda u:packet(u,'completion_latency_ms_sum')),
            packet_completed=components(self.uids,lambda u:packet(u,'completed_packets')),
            packet_latency=components(self.uids,lambda u:packet(u,'completion_latency_ms_sum')),
            overflow=components(self.uids,lambda u:packet(u,'overflow_bytes')))
        for field,key in [('generated','task_generated'),('completed','task_completed'),('failed','task_failed'),
                          ('on_time_completed','task_on_time'),('offloaded_generated','mec_generated'),
                          ('offloaded_completed','mec_completed'),('offloaded_failed','mec_failed')]:
            data[key] = components(self.task_uids,lambda u:task(u,field))
        data['mec_gflop'] = components(self.task_uids,lambda u:
            task(u,'offloaded_generated')*self.profiles[u]['compute_gflop']
            if numeric(task(u,'offloaded_generated')) and numeric(self.profiles.get(u, {}).get('compute_gflop')) else None)
        for key,column in self.counters.items():column.append(data[key])
        expected = []
        for uid in self.task_uids:
            p = self.profiles.get(uid, {})
            fraction = (services.get(uid, {}).get('ue_service_policy') or {}).get('offload_fraction')
            expected.append(1000/p['interval_ms']*p['compute_gflop']*fraction
                if numeric(p.get('interval_ms')) and p['interval_ms'] > 0 and numeric(p.get('compute_gflop')) and numeric(fraction) and 0 <= fraction <= 1 else None)
        self.instant['expected'].append(sum(expected) if expected and all(numeric(v) for v in expected) else None)
        queues = [users.get(u, {}).get('queue_bytes') for u in self.uids]
        hols = [packet(u,'hol_delay_ms') for u in self.uids]
        self.instant['queue'].append(sum(queues) if queues and all(numeric(v) for v in queues) else None)
        self.instant['hol'].append(max(hols) if hols and all(numeric(v) for v in hols) else None)
        for event in events:
            self.observe_task(event, slot)

    def observe_task(self, event, slot):
        if event.get('ue_id') not in self.task_uids or not event.get('kind','').startswith('task_'):
            return
        try:
            key = (event['ue_id'], event['task_id'])
            for field in ('created_s','deadline_s','time_s'):
                if not numeric(event.get(field)) or event[field] < 0:
                    raise ValueError('invalid_task_clock')
            if event['deadline_s'] <= 0 or event['time_s'] < event['created_s']-1e-9 or event['time_s'] > slot*self.dt+1e-9:
                raise ValueError('invalid_task_clock')
            if event.get('offloaded') is not None and type(event['offloaded']) is not bool:
                raise ValueError('invalid_task_route')
            if event['kind'] == 'task_generated':
                if key in self.tasks:raise ValueError('duplicate_task_birth')
                if not math.isclose(event['created_s'],event['time_s'],abs_tol=1e-9):raise ValueError('invalid_task_birth')
                self.tasks[key] = dict(ue_id=key[0], created_s=event['created_s'], deadline_s=event['deadline_s'],
                    offloaded=event.get('offloaded'), born_slot=slot)
            else:
                if event['kind'] not in ('task_completed','task_queue_overflow','task_result_overflow'):
                    raise ValueError('unknown_task_outcome')
                task = self.tasks.get(key)
                if task is None or 'terminal' in task:raise ValueError('missing_or_duplicate_task_birth')
                if any(task[k] != event.get(k) for k in ('created_s','deadline_s','offloaded')):
                    raise ValueError('task_identity_changed')
                if event['kind'] == 'task_completed' and (not numeric(event.get('latency_ms')) or event['latency_ms'] < 0 or not math.isclose(event['latency_ms'],(event['time_s']-event['created_s'])*1000,abs_tol=1e-7)):
                    raise ValueError('invalid_task_latency')
                task['terminal'] = dict(event, recorded_slot=slot)
        except (ValueError,KeyError,TypeError) as error:
            self.task_errors.append((slot,str(error)))

    def _task_metric(self, key, start, end, a, b):
        # Both arms use the same birth exposure and maximum configured deadline.
        deadlines = [self.profiles.get(u, {}).get('deadline_ms') for u in self.task_uids]
        if not deadlines or not all(numeric(d) and d > 0 for d in deadlines):
            return dict(value=None,reason='missing_task_profile')
        cutoff = end*self.dt-max(deadlines)/1000
        exposure = max(0.,cutoff-start*self.dt)
        tasks = [t for t in self.tasks.values() if start < t['born_slot'] <= end]
        # Audit the event stream against the independent simulator counters.
        checks = [('task_generated',len(tasks)),
                  ('mec_generated',sum(t['offloaded'] is True for t in tasks))]
        all_tasks = self.tasks.values()
        for column,kinds in [('task_completed',{'task_completed'}),('task_failed',{'task_queue_overflow','task_result_overflow'})]:
            count = sum(start < t.get('terminal',{}).get('recorded_slot',-1) <= end and t['terminal']['kind'] in kinds for t in all_tasks)
            checks.append((column,count))
        if any(self.counters[k].delta(a,b) != count for k,count in checks) or any(s <= end for s,_ in self.task_errors):
            return dict(value=None,reason='incomplete_task_event_log')
        eligible = [t for t in tasks if start*self.dt-1e-10 <= t['created_s'] < cutoff-1e-10]
        if key == 'mec_offloaded_latency_ms':eligible = [t for t in eligible if t['offloaded'] is True]
        completed,failed,pending,on_time = [],0,0,0
        for task in eligible:
            terminal = task.get('terminal')
            if not terminal or terminal['recorded_slot'] > end or terminal['time_s'] > end*self.dt+1e-9:
                pending += 1
            elif terminal['kind'] == 'task_completed':
                completed.append(terminal['latency_ms'])
                on_time += terminal['latency_ms'] <= task['deadline_s']*1000+1e-9
            else:failed += 1
        metadata = dict(sample_count=len(completed), generated=len(eligible), failed=failed,pending=pending,
            cohort_exposure_ms=exposure*1000,deadline_guard_ms=max(deadlines),
            cohort_start_ms=start*self.dt_ms,cohort_end_ms=max(start*self.dt_ms,cutoff*1000),
            aggregation='matched_mature_birth_cohort',all_generated=len(tasks))
        if pending:
            return dict(value=None,reason='unresolved_mature_tasks',**metadata)
        if key == 'task_success_ratio':
            value=on_time/len(eligible) if eligible else None
            metadata['resolution']=1/len(eligible) if eligible else None
        else:
            value=sum(completed)/len(completed) if completed else None
        return dict(value=value,reason=None if value is not None else 'no_mature_task_samples',**metadata)

    def window(self, keys, start, end, *, through):
        """Counters C_end - C_start describe slots start+1 ... end inclusive."""
        keys = list(keys)
        result = dict(slot=end,time_ms=end*self.dt_ms,window_start_slot=start,window_slots=end-start,states={})
        a,b = bisect_left(self.slots,start),bisect_left(self.slots,end)
        reason = None
        if type(start) is not int or type(end) is not int or end <= start or end > through:
            reason='invalid_or_future_window'
        elif a >= len(self.slots) or b >= len(self.slots) or self.slots[a] != start or self.slots[b] != end or b-a != end-start:
            reason='exact_window_data_unavailable'
        seconds = (end-start)*self.dt
        for key in keys:
            row = dict(value=None,reason=reason,window='matched_window',window_start_slot=start,
                sampled_at_slot=end,window_slots=end-start,source='recorded_scalar_counters_and_task_events')
            if reason:
                result['states']['kpi:'+key] = row;continue
            delta=lambda name:self.counters[name].delta(a,b)
            mappings={'ue_power_w':('energy',1/seconds),'ap_power_w':('ap_energy',1/seconds),
                'network_goodput_mbps':('delivered',8/1e6/seconds),
                'per_ue_goodput_mbps':('delivered',8/1e6/seconds/len(self.uids)),
                'urllc_failure_events_per_s':('failure',1/seconds),
                'ue_active_fraction':('active',1/(end-start)/len(self.uids)),
                'ap_available_fraction':('ap_active',1/(end-start)/max(1,len(self.cell_ids))),
                'queue_overflow_bytes':('overflow',1.),'mec_realized_gflops':('mec_gflop',1/seconds)}
            if key in mappings:
                name,factor=mappings[key];change=delta(name)
                row.update(value=change*factor if change is not None else None,aggregation='counter_delta_over_equal_window',reason=None if change is not None else 'missing_or_reset_counter')
                if key in ('network_goodput_mbps','per_ue_goodput_mbps'):row['resolution']=8/1e6/seconds/(len(self.uids) if key.startswith('per_ue') else 1)
                if key=='urllc_failure_events_per_s':row.update(sample_count=change,resolution=1/seconds)
                if key in ('ue_active_fraction','ap_available_fraction'):row['resolution']=factor
            elif key == 'urllc_failure_fraction':
                failures,offered=delta('failure'),delta('urllc_offered')
                row.update(value=failures/offered if failures is not None and offered is not None and offered>0 else None,
                    reason=None if failures is not None and offered is not None and offered>0 else 'no_valid_offered_packet_samples',
                    sample_count=offered,resolution=1/offered if offered else None,
                    failure_events=failures,offered_packets=offered,
                    canonical_unit='ratio',display_unit='%',display_multiplier=100.,
                    includes_carryover=True,mature_cohort_probability=False,
                    aggregation='interval_event_ratio_including_carryover_not_cohort_probability')
            elif key in ('mec_offloaded_latency_ms','task_latency_ms','task_success_ratio'):
                row.update(self._task_metric(key,start,end,a,b))
            elif key in ('urllc_packet_latency_ms','packet_latency_ms'):
                stem='urllc' if key.startswith('urllc') else 'packet'
                n,total=delta(stem+'_completed'),delta(stem+'_latency')
                row.update(value=total/n if n is not None and n>0 and total is not None else None,
                    reason=None if n is not None and n>0 and total is not None else 'no_valid_completed_packet_samples',
                    sample_count=n,resolution=self.dt_ms,aggregation='completion_weighted_interval_mean')
            elif key in ('mec_expected_gflops','queue_bytes','hol_delay_ms'):
                stat=self.instant[{'mec_expected_gflops':'expected','queue_bytes':'queue','hol_delay_ms':'hol'}[key]].interval(a,b)
                row.update(stat or dict(value=None,reason='missing_interval_configuration'))
                if stat:row.update(aggregation='time_mean_and_full_interval_bounds',constraint_uses_extrema=True)
            else:
                row['reason']='unsupported_window_metric'
            result['states']['kpi:'+key] = row
        return result


class JsonTail:
    """Incrementally decode the existing gzip/plain JSONL, including live flushes."""
    def __init__(self, folder, name, consume):
        self.paths=[Path(folder)/(name+'.jsonl.gz'),Path(folder)/(name+'.jsonl')]
        self.consume,self.offset,self.pending,self.identity=consume,0,b'',None
        self.decoder=None

    def sync(self):
        path=next((p for p in self.paths if p.exists()),None)
        if path is None:return
        stat=path.stat();identity=(str(path),stat.st_dev,stat.st_ino)
        if self.identity is not None and (identity != self.identity or stat.st_size < self.offset):
            raise ValueError('Recorded run changed; reload its window index')
        self.identity=identity
        if path.suffix=='.gz' and self.decoder is None:self.decoder=zlib.decompressobj(31)
        with path.open('rb') as stream:
            stream.seek(self.offset)
            while chunk:=stream.read(1024*1024):
                self.offset+=len(chunk)
                raw=b''
                if self.decoder:
                    while chunk:
                        if self.decoder.eof:self.decoder=zlib.decompressobj(31)
                        raw+=self.decoder.decompress(chunk)
                        chunk=self.decoder.unused_data
                else:raw=chunk
                self.pending+=raw
                lines=self.pending.split(b'\n');self.pending=lines.pop()
                for line in lines:
                    if line.strip():self.consume(json.loads(line))


class RecordedWindows(WindowIndex):
    def __init__(self, folder, config, *, population=True):
        super().__init__(config,population=population)
        def consume(row):
            from smo.runtime import trace_users
            self.observe(row['slot_id'],trace_users(row),row.get('ue_services',{}),
                row.get('ap_services',{}).get('cells',{}),row.get('ue_service_events',[]))
        self.trace=JsonTail(folder,'rt_sys_trace',consume)
        self.controls=JsonTail(folder,'control_receipts',self.observe_receipt)

    def sync(self):
        self.trace.sync();self.controls.sync()
