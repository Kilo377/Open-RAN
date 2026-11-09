"""Opt-in, independently seeded offload and atomic scheduler arbitration.

This chooses a request, never a blended parameter. Requests remain held until
withdrawal or source expiry; a choice is a snapshot valid for one decision window.
"""
from copy import deepcopy
import math
import random


SCHEDULER_BUNDLE_FIELDS = frozenset({'scheduler_mode', 'ue_scheduler_weights'})
SCHEDULER_RESERVED_FIELDS = SCHEDULER_BUNDLE_FIELDS | frozenset({
    'scheduler_policy_hint', 'latency_bias', 'slice_weight_adjustment'})
SUPPORTED_FIELDS = frozenset({'ue_offload_fraction'}) | SCHEDULER_BUNDLE_FIELDS


def overlaps_unequally(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return any(left[uid] != right[uid] for uid in left.keys() & right.keys())
    return left != right


class RandomOffloadArbiter:
    def __init__(self, seed, window_ms, slot_ms):
        if type(seed) is not int:
            raise ValueError('mitigation_seed must be an integer')
        for name, value in [('mitigation_window_ms', window_ms), ('slot_duration_ms', slot_ms)]:
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + ' must be positive and finite')
        slots = window_ms / slot_ms
        if not math.isclose(slots, round(slots), abs_tol=1e-8) or slots < 1:
            raise ValueError('mitigation window must align to simulation slots')
        self.seed = seed
        self.window_slots = round(slots)
        # The stream follows stable UID + RCP, including across handover.
        # Another UE's contention cannot consume this UE's random draws.
        self.rngs = {}
        self.draw_indices = {}
        self.current = {}
        self.generations = {}

    @staticmethod
    def request_valid(request, slot):
        start, end = request.get('effective_from_slot'), request.get('effective_until_slot')
        return (start is None or slot >= start) and (end is None or slot <= end)

    def choose(self, uid, cid, slot, candidates, baseline):
        candidates = sorted((deepcopy(c) for c in candidates if self.request_valid(c, slot)), key=lambda c: c['source'])
        if len({c['source'] for c in candidates}) != len(candidates):
            raise ValueError('one live offload request per xApp and UID is required')
        window = slot // self.window_slots
        old = self.current.get(uid)
        source_set = tuple(c['source'] for c in candidates)
        old_source = next((c for c in candidates if old and c['source'] == old['winner']), None)
        still_valid = old and old['window_index'] == window and old['cell_id'] == cid and old['sources'] == source_set
        if still_valid and old['winner'] != 'rule_withdrawal':
            still_valid = old_source is not None and self.request_valid(old, slot) and old_source.get('provenance') == old.get('provenance')
        if still_valid:
            return deepcopy(old), None
        if not candidates and baseline is None:
            self.current.pop(uid, None)
            self.generations[uid] = self.generations.get(uid, 0) + 1
            return None, None
        conflict = len({c['value'] for c in candidates}) > 1
        stream_key='ue_offload_fraction:'+uid
        if conflict:
            rng=self.rngs.setdefault(uid,random.Random(f'{self.seed}:direct-conflict-mitigation:{stream_key}'))
            draw=rng.random()
            self.draw_indices[uid]=self.draw_indices.get(uid,0)+1
        else:draw=None
        selected = (candidates[min(int(draw * len(candidates)), len(candidates) - 1)] if conflict else
                    max(candidates, key=lambda c: (c['priority'], c['source'])) if candidates else
                    dict(source='rule_withdrawal', value=baseline, request_id=None, priority=0))
        generation = self.generations.get(uid, 0) + 1
        self.generations[uid] = generation
        result = dict(field='ue_offload_fraction', ue_id=uid, cell_id=cid,
            window_index=window, window_start_slot=window * self.window_slots,
            window_end_slot_exclusive=(window + 1) * self.window_slots,
            mitigation_seed=self.seed, generation=generation, winner=selected['source'],
            selected_value=selected['value'], request_id=selected.get('request_id'),
            effective_from_slot=selected.get('effective_from_slot'), effective_until_slot=selected.get('effective_until_slot'),
            provenance=deepcopy(selected.get('provenance')), priority=selected['priority'],
            sources=source_set, candidates=[dict(source=c['source'], requested_value=c['value'],
                request_id=c.get('request_id'),effective_from_slot=c.get('effective_from_slot'),
                effective_until_slot=c.get('effective_until_slot'),provenance=deepcopy(c.get('provenance'))) for c in candidates], draw=draw,
            stream_key=stream_key,draw_index=self.draw_indices.get(uid) if conflict else None,
            selection_cause='initial' if old is None else 'window_advance' if old['window_index']!=window else 'candidate_set_or_validity_change',
            probability=1 / len(candidates) if conflict else 1., conflict=conflict)
        self.current[uid] = deepcopy(result)
        event = dict(reason='direct_conflict_random_selection' if conflict else 'direct_conflict_request_selection',
                     slot_id=slot, **deepcopy(result))
        return result, event

    def valid_selection(self, record, slot, cid):
        current = self.current.get(record['ue_id'])
        return bool(current and current['generation'] == record['generation'] and
                    current['cell_id'] == cid and slot < record['window_end_slot_exclusive'] and
                    self.request_valid(record, slot))


class RandomSchedulerArbiter(RandomOffloadArbiter):
    """One unchanged mode/complete-UID-map request per cell and time window.

    Its cell streams are separate from every offload stream. Re-emissions and
    same-source replacements are deferred to the next window, unless the held
    request expires, its source changes selection generation, or membership
    changes. A new membership invalidates the whole old action, never a subset.
    """

    def choose(self, cid, slot, candidates, baseline, serving_ue_ids, baseline_tuning=None):
        members=tuple(sorted(serving_ue_ids))
        candidates=sorted((deepcopy(c) for c in candidates if self.request_valid(c,slot)),key=lambda c:c['source'])
        if len({c['source'] for c in candidates})!=len(candidates):
            raise ValueError('one live scheduler bundle per xApp and cell is required')
        window=slot//self.window_slots;old=self.current.get(cid)
        sources=tuple(c['source'] for c in candidates)
        source=next((c for c in candidates if old and c['source']==old['winner']),None)
        held=bool(old and old['window_index']==window and old['sources']==sources and old['serving_ue_ids']==members)
        if held and old['winner']!='rule_withdrawal':
            held=bool(source and self.request_valid(old,slot) and source.get('provenance')==old.get('provenance'))
        if held:return deepcopy(old),None
        if not candidates and baseline is None:
            self.invalidate(cid)
            return None,None
        import json
        signature=lambda c:json.dumps(c['value'],sort_keys=True,separators=(',',':'),allow_nan=False)
        conflict=len({signature(c) for c in candidates})>1
        stream_key='scheduler_bundle:'+cid
        if conflict:
            rng=self.rngs.setdefault(cid,random.Random(f'{self.seed}:direct-conflict-mitigation:{stream_key}'))
            draw=rng.random();self.draw_indices[cid]=self.draw_indices.get(cid,0)+1
            selected=candidates[min(int(draw*len(candidates)),len(candidates)-1)]
        else:
            draw=None
            selected=max(candidates,key=lambda c:(c['priority'],c['source'])) if candidates else dict(
                source='rule_withdrawal',value=baseline,request_id=None,priority=0,framework_tuning=baseline_tuning or {})
        generation=self.generations.get(cid,0)+1;self.generations[cid]=generation
        result=dict(field='scheduler_bundle',fields=sorted(SCHEDULER_BUNDLE_FIELDS),cell_id=cid,
            serving_ue_ids=members,window_index=window,window_start_slot=window*self.window_slots,
            window_end_slot_exclusive=(window+1)*self.window_slots,mitigation_seed=self.seed,
            generation=generation,winner=selected['source'],selected_value=deepcopy(selected['value']),
            framework_tuning=deepcopy(selected.get('framework_tuning',{})),request_id=selected.get('request_id'),
            effective_from_slot=selected.get('effective_from_slot'),effective_until_slot=selected.get('effective_until_slot'),
            provenance=deepcopy(selected.get('provenance')),priority=selected['priority'],sources=sources,
            candidates=[dict(source=c['source'],requested_value=deepcopy(c['value']),request_id=c.get('request_id'),
                effective_from_slot=c.get('effective_from_slot'),effective_until_slot=c.get('effective_until_slot'),
                provenance=deepcopy(c.get('provenance'))) for c in candidates],draw=draw,
            stream_key=stream_key,draw_index=self.draw_indices.get(cid) if conflict else None,
            probability=1/len(candidates) if conflict else 1.,conflict=conflict,
            selection_cause='initial' if old is None else 'window_advance' if old['window_index']!=window else 'source_validity_or_membership_change')
        self.current[cid]=deepcopy(result)
        return result,dict(reason='direct_conflict_random_selection' if conflict else 'direct_conflict_request_selection',slot_id=slot,**deepcopy(result))

    def invalidate(self,cid):
        self.current.pop(cid,None)
        self.generations[cid]=self.generations.get(cid,0)+1

    def valid_selection(self,record,slot,cid):
        current=self.current.get(cid)
        return bool(current and current['generation']==record['generation'] and current['cell_id']==cid and
                    slot<record['window_end_slot_exclusive'] and self.request_valid(record,slot))
