"""traffic/arrival_model.py

Defines minimal per-UE traffic arrival processes.

The simulator models downlink/uplink backlog at the base-station side as
`ue.queue_bytes`. These arrival models add bytes into that queue every slot.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


def _poisson_sample(rng: random.Random, lam: float) -> int:
    lam_value = float(lam)
    if lam_value <= 0.0:
        return 0

    # Knuth algorithm for small lambda.
    if lam_value < 30.0:
        l_value = math.exp(-lam_value)
        k = 0
        p = 1.0
        while p > l_value:
            k += 1
            p *= rng.random()
        return max(0, k - 1)

    # Normal approximation for large lambda.
    sample = rng.gauss(lam_value, math.sqrt(lam_value))
    return max(0, int(round(sample)))


@dataclass
class TrafficProfile:
    ue_id: str
    arrival_model: str = "none"  # none | cbr | poisson | on_off
    arrival_rate_bps: float = 0.0
    packet_size_bytes: int = 200
    burst_size_bytes: int = 0
    on_duration_slots: int = 50
    off_duration_slots: int = 20
    start_offset_slots: int = 0
    max_queue_bytes: int | None = None


class ArrivalModel:
    def sample_bytes(self, *, slot_id: int, slot_duration_s: float, rng: random.Random) -> int:
        raise NotImplementedError


class NoneArrivalModel(ArrivalModel):
    def sample_bytes(self, *, slot_id: int, slot_duration_s: float, rng: random.Random) -> int:
        _ = slot_id
        _ = slot_duration_s
        _ = rng
        return 0


class CbrArrivalModel(ArrivalModel):
    def __init__(self, rate_bps: float) -> None:
        self._rate_bps = float(rate_bps)

    def sample_bytes(self, *, slot_id: int, slot_duration_s: float, rng: random.Random) -> int:
        _ = slot_id
        _ = rng
        bits = max(0.0, self._rate_bps) * max(float(slot_duration_s), 0.0)
        return int(bits / 8.0)


class PoissonArrivalModel(ArrivalModel):
    def __init__(self, *, rate_bps: float, packet_size_bytes: int) -> None:
        self._rate_bps = float(rate_bps)
        self._packet_size_bytes = int(max(1, packet_size_bytes))

    def sample_bytes(self, *, slot_id: int, slot_duration_s: float, rng: random.Random) -> int:
        _ = slot_id
        mean_bytes = max(0.0, self._rate_bps) * max(float(slot_duration_s), 0.0) / 8.0
        lam_packets = mean_bytes / float(self._packet_size_bytes)
        packets = _poisson_sample(rng, lam_packets)
        return int(packets) * int(self._packet_size_bytes)


class OnOffArrivalModel(ArrivalModel):
    def __init__(
        self,
        *,
        rate_bps: float,
        on_duration_slots: int,
        off_duration_slots: int,
        burst_size_bytes: int,
        start_offset_slots: int,
    ) -> None:
        self._rate_bps = float(rate_bps)
        self._on = max(1, int(on_duration_slots))
        self._off = max(0, int(off_duration_slots))
        self._burst = int(max(0, burst_size_bytes))
        self._offset = int(max(0, start_offset_slots))

    def _is_on(self, slot_id: int) -> tuple[bool, bool]:
        cycle = self._on + self._off
        if cycle <= 0:
            return True, False
        phase = (int(slot_id) + int(self._offset)) % int(cycle)
        is_on = phase < self._on
        on_start = (phase == 0)
        return is_on, on_start

    def sample_bytes(self, *, slot_id: int, slot_duration_s: float, rng: random.Random) -> int:
        _ = rng
        is_on, on_start = self._is_on(int(slot_id))
        if not is_on:
            return 0

        bits = max(0.0, self._rate_bps) * max(float(slot_duration_s), 0.0)
        base = int(bits / 8.0)
        burst = int(self._burst) if on_start else 0
        return int(base + burst)


def build_arrival_model(profile: TrafficProfile) -> ArrivalModel:
    model = str(profile.arrival_model or "none").lower()
    if model in ("none", "disabled", "off"):
        return NoneArrivalModel()
    if model in ("cbr", "constant"):
        return CbrArrivalModel(rate_bps=float(profile.arrival_rate_bps))
    if model in ("poisson", "burst", "bursty"):
        return PoissonArrivalModel(
            rate_bps=float(profile.arrival_rate_bps),
            packet_size_bytes=int(profile.packet_size_bytes),
        )
    if model in ("on_off", "onoff", "on-off"):
        return OnOffArrivalModel(
            rate_bps=float(profile.arrival_rate_bps),
            on_duration_slots=int(profile.on_duration_slots),
            off_duration_slots=int(profile.off_duration_slots),
            burst_size_bytes=int(profile.burst_size_bytes),
            start_offset_slots=int(profile.start_offset_slots),
        )
    raise ValueError(f"Unsupported arrival_model: {profile.arrival_model}")
