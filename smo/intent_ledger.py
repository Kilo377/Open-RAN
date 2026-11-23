"""Immutable intent anchors; active constraints are selected by record ID.

Stored ranges contain resolved numeric bounds. Re-observing the network changes
their assessment, never the snapshot that defined an earlier requirement.
"""
from copy import deepcopy
import json
from pathlib import Path
from threading import RLock
from uuid import uuid4

from smo.intent_language import IntentError, resolve_ranges


FIELDS = ('kpi', 'lower', 'upper', 'lower_inclusive', 'upper_inclusive')


class IntentLedger:
    def __init__(self, context, path=None):
        self.context = context
        self.path = Path(path) if path else None
        self.lock = RLock()
        self.records = {}
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text())
            if data.get('schema') != 'oran.frozen-intents.v1' or data.get('context') != context:
                raise ValueError('The intent ledger belongs to a different network context.')
            self.records = {r['id']: r for r in data['records']}

    def select(self, ids):
        if not isinstance(ids, list) or len(ids) > 32 or any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
            raise IntentError('Select up to 32 distinct saved intents.')
        with self.lock:
            if any(i not in self.records for i in ids):
                raise IntentError('A saved intent is unavailable in this network context. Clear it or restore its ledger.')
            return deepcopy([self.records[i] for i in ids])

    def add(self, text, frame, interpreted, language):
        record = dict(id=uuid4().hex, text=text,
            snapshot={k: frame.get(k) for k in ('slot', 'time_ms')},
            ranges=deepcopy(interpreted['ranges']), language=deepcopy(language))
        if frame.get('evaluation_origin'):
            record['evaluation_origin']=deepcopy(frame['evaluation_origin'])
        with self.lock:
            # Commit to disk before exposing the ID; keep all records, including
            # subsequently inactive or contradictory constraints, for audit.
            updated = {**self.records, record['id']: record}
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_name(self.path.name+'.tmp')
                tmp.write_text(json.dumps(dict(schema='oran.frozen-intents.v1',
                    context=self.context, records=list(updated.values())), ensure_ascii=False,
                    allow_nan=False, indent=2)+'\n')
                tmp.replace(self.path)
            self.records = updated
        return deepcopy(record)


def intersect_intents(records, graph, frame):
    """Intersect frozen numeric bounds, assessing them at the new observation."""
    merged, anchors = {}, {}
    for record in records:
        for frozen in record['ranges']:
            key = frozen['kpi']
            fixed = {k: frozen[k] for k in FIELDS}
            rows = ([{k: merged[key][k] for k in FIELDS}] if key in merged else []) + [fixed]
            merged[key] = resolve_ranges(dict(ranges=rows, questions=[]), graph, frame)['ranges'][0]
            anchors.setdefault(key, []).append(dict(intent_id=record['id'],
                snapshot=deepcopy(record['snapshot']), current=frozen['current'],
                measurement=deepcopy(frozen.get('measurement')),
                requested=deepcopy(frozen.get('requested', [])), **fixed))
    if len(merged) > 8:
        raise IntentError('Active intents support up to eight distinct KPIs.')
    for key, row in merged.items():
        row['anchors'] = anchors[key]
    return dict(ranges=list(merged.values()), questions=[])
