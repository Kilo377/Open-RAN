"""Offline exact task cohorts from routed birth/terminal events, not telemetry retention.

No event, KPI or controller state is changed by this collector. Each recorded
event is consumed once. Late completions remain attributable after the public
eight-cohort and internal sixteen-cohort buffers have evicted their birth.
"""
from collections import defaultdict
import math


def _number(value, name, minimum=0.):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError('Invalid task event ' + name)
    return float(value)


def _counts(tasks, duration_s):
    result = dict(generated=0, completed=0, on_time_completed=0, failed=0, pending=0,
        completion_latency_ms_sum=0., max_completion_latency_ms=0.,
        offloaded=0, local=0, unassigned=0, offloaded_generated=0,
        offloaded_completed=0, offloaded_on_time_completed=0, offloaded_failed=0,
        offloaded_pending=0, offloaded_completion_latency_ms_sum=0.)
    for task in tasks:
        result['generated'] += 1
        route = task['offloaded']
        result['offloaded' if route is True else 'local' if route is False else 'unassigned'] += 1
        if route is True: result['offloaded_generated'] += 1
        terminal = task.get('terminal')
        outcome = 'pending' if terminal is None else 'completed' if terminal['kind'] == 'task_completed' else 'failed'
        result[outcome] += 1
        if route is True: result['offloaded_' + outcome] += 1
        if outcome == 'completed':
            latency = terminal['latency_ms']
            on_time = latency <= task['deadline_s'] * 1000 + 1e-9
            result['on_time_completed'] += on_time
            result['completion_latency_ms_sum'] += latency
            result['max_completion_latency_ms'] = max(result['max_completion_latency_ms'], latency)
            if route is True:
                result['offloaded_on_time_completed'] += on_time
                result['offloaded_completion_latency_ms_sum'] += latency
    assert result['generated'] == result['completed'] + result['failed'] + result['pending']
    assert result['offloaded_generated'] == result['offloaded_completed'] + result['offloaded_failed'] + result['offloaded_pending']
    return result


class TaskEventCollector:
    """Collect every target task; summarize matched mature creation cohorts.

    ``per_ue[uid]['cohorts']`` contains legacy-compatible count records, ordered
    by birth cohort. Empty/immature populations remain empty, with no fabricated
    zero delay. ``whole_run`` includes startup and right-censored tasks.
    """
    def __init__(self, duration_s, cohort_ms=100, start_s=.1,
                 target_uids=('ue_0', 'ue_4')):
        self.duration_s = _number(duration_s, 'duration_s', 1e-12)
        self.period_s = _number(cohort_ms, 'cohort_ms', 1e-12) / 1000
        self.start_s = _number(start_s, 'start_s')
        self.uids = tuple(target_uids)
        if len(set(self.uids)) != len(self.uids): raise ValueError('Repeated task target UID')
        self.tasks = {}
        self.observed_terminal = 0

    def observe(self, event):
        kind = event.get('kind', '')
        if not kind.startswith('task_') or event.get('ue_id') not in self.uids: return
        if kind not in ('task_generated', 'task_completed', 'task_queue_overflow', 'task_result_overflow'):
            raise ValueError('Unknown terminal task event kind')
        required = ('task_id', 'created_s', 'deadline_s', 'cohort_index', 'offloaded', 'time_s')
        if any(k not in event for k in required):
            raise ValueError('Task event lacks routed birth provenance; legacy telemetry is insufficient')
        uid = event['ue_id']; key = (uid, event['task_id'])
        created = _number(event['created_s'], 'created_s')
        deadline = _number(event['deadline_s'], 'deadline_s', 1e-12)
        timestamp = _number(event['time_s'], 'time_s')
        if timestamp < created - 1e-9 or timestamp > self.duration_s + 1e-9:
            raise ValueError('Task event outside its visible simulation horizon')
        route = event['offloaded']
        if route is not None and type(route) is not bool: raise ValueError('Invalid recorded task route')
        index = event['cohort_index']
        if type(index) is not int or index < 0: raise ValueError('Task birth cohort unavailable')
        # Generation uses now=(slot-1)*dt and index=(slot-1)//cohort_slots.
        # An exact endpoint starts the NEW bucket; normalize floating point
        # values close to an integer before flooring the recorded birth time.
        ratio = created / self.period_s
        nearest = round(ratio)
        expected_index = nearest if math.isclose(ratio, nearest, abs_tol=1e-8, rel_tol=0) else math.floor(ratio)
        if index != expected_index:
            raise ValueError('Task birth time/cohort mismatch')
        if kind == 'task_generated':
            if key in self.tasks: raise ValueError('Repeated task generation event')
            if not math.isclose(timestamp, created, abs_tol=1e-9, rel_tol=0):
                raise ValueError('Task generation time differs from birth')
            self.tasks[key] = dict(created_s=created, deadline_s=deadline,
                cohort_index=index, offloaded=route)
            return
        task = self.tasks.get(key)
        if task is None: raise ValueError('Terminal task event has no recorded birth')
        if 'terminal' in task: raise ValueError('Repeated task terminal event')
        if any(task[k] != event[k] for k in ('created_s', 'deadline_s', 'cohort_index', 'offloaded')):
            raise ValueError('Terminal task event changed immutable birth/route provenance')
        if kind == 'task_completed':
            latency = _number(event.get('latency_ms'), 'latency_ms')
            if not math.isclose(latency, (timestamp - created) * 1000, abs_tol=1e-7, rel_tol=1e-10):
                raise ValueError('Task completion latency differs from the recorded clock')
            if route is None: raise ValueError('Unassigned task cannot complete')
        task['terminal'] = dict(event)
        self.observed_terminal += 1

    def summary(self):
        result = {}
        for uid in self.uids:
            tasks = [v for (u, _), v in self.tasks.items() if u == uid]
            groups = defaultdict(list)
            for task in tasks: groups[task['cohort_index']].append(task)
            cohorts = []
            for index, members in sorted(groups.items()):
                start, end = index * self.period_s, (index + 1) * self.period_s
                mature = end + max(t['deadline_s'] for t in members)
                if start < self.start_s - 1e-9 or mature > self.duration_s + 1e-9: continue
                cohorts.append(dict(index=index, start_s=start, end_s=end,
                    mature_after_s=mature, **_counts(members, self.duration_s)))
            result[uid] = dict(cohorts=cohorts, whole_run=_counts(tasks, self.duration_s),
                eligible_cohorts=len(cohorts), eligible_generated=sum(c['generated'] for c in cohorts))
        return dict(schema='routed_task_event_cohorts_v1', per_ue=result,
            observed_generated=len(self.tasks), observed_terminal=self.observed_terminal,
            duration_s=self.duration_s, start_s=self.start_s, cohort_ms=self.period_s * 1000,
            scope='All recorded terminal outcomes through run end for matched mature birth cohorts; late completions survive telemetry eviction. Completion means exclude pending/failed outcomes; those remain explicit in task success denominators.')
