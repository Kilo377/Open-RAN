"""Read-only actual-decode statistics; never an input to scheduling or PHY.

First transmissions and retransmissions have different denominators. A pooled
NACK ratio must not be presented as a first-transmission BLER target certificate.
"""
from collections import Counter, defaultdict


LATENCY_SCOPE = ('Slot-index queue completion delay of completed arrival units; '
                 'same-slot completion is zero. Excludes within-slot airtime, '
                 'processing/transport/core delay and unfinished/dropped units.')


def summarize(counts):
    result = {name: counts[name] for name in
              ('first_tx_attempts', 'first_tx_nacks', 'retx_attempts', 'retx_nacks')}
    first, retx = result['first_tx_attempts'], result['retx_attempts']
    result.update(first_tx_bler=result['first_tx_nacks'] / first if first else None,
                  retx_error_rate=result['retx_nacks'] / retx if retx else None,
                  actual_decode_attempts=first + retx,
                  actual_nacks=result['first_tx_nacks'] + result['retx_nacks'])
    return result


class LinkStatistics:
    def __init__(self):
        self.last_slot = None
        self.total = Counter()
        self.users = defaultdict(Counter)
        self.modes = defaultdict(lambda: defaultdict(Counter))

    def observe(self, slot, rows):
        if slot == self.last_slot:
            return
        if self.last_slot is not None and slot < self.last_slot:
            raise ValueError('Link statistics require increasing simulation slots')
        self.last_slot = slot
        for row in rows:
            if not row.get('actual_decode'):
                continue
            phase = 'retx' if row.get('is_retx', False) else 'first_tx'
            uid = row['ue_id']
            # Old artifacts lack resource-mode evidence; never infer SU merely
            # from a missing field or the number of UEs scheduled in a cell.
            mode = row.get('resource_mode', 'unknown')
            for counts in (self.total, self.users[uid], self.modes[uid][mode]):
                counts[phase + '_attempts'] += 1
                counts[phase + '_nacks'] += not row['ack']

    def user(self, uid):
        return dict(**summarize(self.users.get(uid, Counter())),
                    by_resource_mode={mode: summarize(c)
                                      for mode, c in self.modes.get(uid, {}).items()})

    def snapshot(self):
        return dict(network=summarize(self.total),
                    per_ue={uid: self.user(uid) for uid in self.users},
                    scope='Cumulative actual NR decode outcomes, including warmup',
                    interpretation='Descriptive ratios; no BLER convergence or QoS guarantee')
