"""FIFO arrival units with explicit byte conservation and completion latency.

Byte arrival models are packetized into units no larger than packet_size_bytes.
The last unit of an arrival event can be shorter. Deadlines count overdue units;
they do not silently remove data still owned by MAC/HARQ.
"""
from collections import Counter, deque


class PacketLedger:
    def __init__(self, ue, slot_duration_ms):
        self.ue = ue
        self.dt = float(slot_duration_ms)
        self.queues = {lc.logical_channel_id: deque() for lc in ue.logical_channels}
        self.stats = Counter(dict.fromkeys(('initial_bytes', 'offered_bytes', 'overflow_bytes',
            'overflow_packets', 'admitted_packets', 'delivered_bytes', 'completed_packets',
            'completion_latency_ms_sum', 'max_completion_latency_ms', 'late_completed_packets',
            'on_time_completed_packets', 'deadline_missed_packets', 'pending_bytes', 'pending_packets'), 0))
        self.events = []
        self.sequence = 0
        self.slot = 0
        for lc in ue.logical_channels:
            self.queues[lc.logical_channel_id] = deque()
            self.add(lc, int(lc.queue_bytes), 0, initial=True)
        self.refresh(0)

    def add(self, lc, amount, slot, *, initial=False, packet_size=200):
        queue = self.queues.setdefault(lc.logical_channel_id, deque())
        occupancy = sum(p['remaining_bytes'] for p in queue)
        cap = lc.max_queue_bytes
        admitted = 0
        self.stats['initial_bytes' if initial else 'offered_bytes'] += amount
        # Initial queues are an explicit initial condition, not a capped arrival.
        while amount:
            size = min(max(1, int(packet_size)), amount)
            amount -= size
            self.sequence += 1
            pid = f'{self.ue.ue_id}:{lc.logical_channel_id}:{self.sequence}'
            accepted = initial or cap is None or occupancy + size <= cap
            event = dict(ue_id=self.ue.ue_id, logical_channel_id=lc.logical_channel_id,
                         packet_id=pid, slot_id=slot, bytes=size,
                         kind='initial' if initial else 'arrival', admitted=accepted)
            self.events.append(event)
            if not accepted:
                self.stats['overflow_bytes'] += size
                self.stats['overflow_packets'] += 1
                continue
            queue.append(dict(packet_id=pid, arrival_slot=slot, size_bytes=size,
                              remaining_bytes=size, deadline_ms=float(lc.packet_delay_budget_ms), overdue=False))
            occupancy += size
            admitted += size
            self.stats['admitted_packets'] += 1
        lc.queue_bytes = occupancy
        self.refresh(slot)
        return admitted

    def deliver(self, lc, amount, slot):
        queue = self.queues[lc.logical_channel_id]
        if amount > sum(p['remaining_bytes'] for p in queue):
            raise RuntimeError('Delivery exceeds packet ledger backlog')
        while amount:
            packet = queue[0]
            taken = min(amount, packet['remaining_bytes'])
            amount -= taken
            packet['remaining_bytes'] -= taken
            self.stats['delivered_bytes'] += taken
            delay = (slot - packet['arrival_slot']) * self.dt
            self.events.append(dict(kind='delivery', ue_id=self.ue.ue_id,
                logical_channel_id=lc.logical_channel_id, packet_id=packet['packet_id'],
                slot_id=slot, bytes=taken, latency_ms=delay, completed=packet['remaining_bytes'] == 0))
            if not packet['remaining_bytes']:
                queue.popleft()
                self.stats['completed_packets'] += 1
                self.stats['completion_latency_ms_sum'] += delay
                self.stats['max_completion_latency_ms'] = max(self.stats['max_completion_latency_ms'], delay)
                self.stats['late_completed_packets' if delay > packet['deadline_ms'] else 'on_time_completed_packets'] += 1
        self.refresh(slot)

    def refresh(self, slot):
        self.slot = slot
        heads = []
        pending = 0
        for lc in self.ue.logical_channels:
            queue = self.queues[lc.logical_channel_id]
            lc.hol_delay_ms = (slot - queue[0]['arrival_slot']) * self.dt if queue else 0.
            heads.append(lc.hol_delay_ms)
            for p in queue:
                pending += p['remaining_bytes']
                if not p['overdue'] and (slot - p['arrival_slot']) * self.dt > p['deadline_ms']:
                    p['overdue'] = True
                    self.stats['deadline_missed_packets'] += 1
                    self.events.append(dict(kind='deadline_miss', ue_id=self.ue.ue_id,
                        logical_channel_id=lc.logical_channel_id, packet_id=p['packet_id'],
                        slot_id=slot, remaining_bytes=p['remaining_bytes']))
        self.ue.hol_delay_ms = max(heads, default=0.)
        self.stats['pending_bytes'] = pending
        self.stats['pending_packets'] = sum(len(q) for q in self.queues.values())
        if self.stats['initial_bytes'] + self.stats['offered_bytes'] != self.stats['delivered_bytes'] + self.stats['overflow_bytes'] + pending:
            raise RuntimeError('Packet ledger byte conservation failed')

    def summary(self):
        result = dict(self.stats)
        result['mean_completion_latency_ms'] = self.stats['completion_latency_ms_sum'] / max(1, self.stats['completed_packets'])
        result['hol_delay_ms'] = self.ue.hol_delay_ms
        result['packetization'] = 'arrival_event_chunks_up_to_configured_packet_size'
        return result

    def drain_events(self):
        result, self.events = self.events, []
        return result
