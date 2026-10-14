"""Fixed DU SINR-offset OLLA, driven only by causal first-TB CRC feedback.

Algorithm reference: OCUDU b78d09a77ab060cda752f32483e551f1e9132f66,
lib/scheduler/support/outer_loop_link_adaptation.h (BSD-3-Clause-Open-MPI).
This independently written adapter uses measured SINR dB, not OCUDU DL CQI.
"""
from dataclasses import dataclass, field
from collections import Counter


@dataclass(frozen=True)
class HarqFeedbackEvent:
    tb_id: str
    cell_id: str
    table: str
    mcs: int
    ack: bool
    is_retx: bool
    available_slot: int
    min_mcs: int = 0
    max_mcs: int = 27


@dataclass
class OllaState:
    cell_id: str | None = None
    offsets: dict = field(default_factory=dict)
    pending: list = field(default_factory=list)
    seen: set = field(default_factory=set)
    stats: Counter = field(default_factory=Counter)
    target: float = .1
    ack_step_db: float = .01
    limit_db: float = 10.

    def consume(self, slot, cell_id):
        cell_id=str(cell_id)
        if self.cell_id != cell_id:
            self.offsets.clear()
            self.cell_id=cell_id
        remaining=[]
        for e in self.pending:
            if e.available_slot>slot:
                remaining.append(e);continue
            key=(e.tb_id,e.is_retx,e.available_slot)
            if key in self.seen:continue
            self.seen.add(key)
            if e.cell_id != cell_id:
                self.stats['stale_cell_feedback']+=1;continue
            if e.is_retx:
                self.stats['retx_samples']+=1
                self.stats['retx_errors']+=not e.ack
                continue
            # A first-TB feedback can only update once even if delivered twice.
            first_key=('first',e.tb_id)
            if first_key in self.seen:continue
            self.seen.add(first_key)
            self.stats['first_samples']+=1;self.stats['first_errors']+=not e.ack
            offset=self.offsets.get(e.table,0.)
            delta=self.ack_step_db if e.ack else -self.ack_step_db*(1-self.target)/self.target
            if (e.ack and e.mcs>=e.max_mcs) or (not e.ack and e.mcs<=e.min_mcs):
                self.stats['boundary_protected']+=1;delta=0.
            self.offsets[e.table]=max(-self.limit_db,min(self.limit_db,offset+delta))
        self.pending=remaining

    def summary(self):
        n=self.stats['first_samples'];r=self.stats['retx_samples']
        from phy.rt_sys.reference import probability_interval
        return dict(self.stats,offsets_db=dict(self.offsets),target_bler=self.target,
            first_tx_bler=self.stats['first_errors']/n if n else None,
            first_tx_wilson_95=probability_interval(self.stats['first_errors'],n) if n else None,
            retx_error_rate=self.stats['retx_errors']/r if r else None,
            target_status='insufficient_samples' if n<2000 else 'requires_post_warmup_reachability_validation')
