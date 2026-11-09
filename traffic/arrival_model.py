"""traffic/arrival_model.py

Defines minimal per-UE traffic arrival processes.

The simulator models downlink/uplink backlog at the base-station side as
`ue.queue_bytes`. These arrival models add bytes into that queue every slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
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
    arrival_model: str = "none"  # none | cbr | poisson | on_off | phased
    arrival_rate_bps: float = 0.0
    packet_size_bytes: int = 200
    burst_size_bytes: int = 0
    on_duration_slots: int = 50
    off_duration_slots: int = 20
    start_offset_slots: int = 0
    max_queue_bytes: int | None = None
    arrival_phases: list[dict] | None = None


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


def validate_arrival_phases(phases: list[dict] | None) -> list[dict]:
    """Validate explicit, complete phases without modifying the caller's data.

    Phases vary arrival timing/rate only. Packet size remains the enclosing
    profile's packet size. Whole ON/OFF periods prevent cut-off burst cycles.
    """
    if not isinstance(phases, list) or not phases:
        raise ValueError('Phased arrivals require a nonempty arrival_phases list')
    common = {'duration_slots', 'arrival_model', 'arrival_rate_bps'}
    on_off = {'on_duration_slots', 'off_duration_slots', 'burst_size_bytes', 'start_offset_slots'}
    for phase in phases:
        if not isinstance(phase, dict) or not common <= set(phase):
            raise ValueError('Each arrival phase requires duration, model and rate')
        model = phase['arrival_model']
        allowed = common | on_off if model == 'on_off' else common
        if model not in ('cbr', 'on_off') or set(phase) - allowed:
            raise ValueError('Arrival phases support only explicit CBR or ON/OFF parameters')
        duration = phase['duration_slots']
        rate = phase['arrival_rate_bps']
        if type(duration) is not int or duration <= 0:
            raise ValueError('Arrival phase duration must be a positive integer')
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate < 0:
            raise ValueError('Invalid arrival phase rate')
        if model == 'on_off':
            if not {'on_duration_slots', 'off_duration_slots'} <= set(phase):
                raise ValueError('ON/OFF arrival phases require explicit ON and OFF durations')
            for key in on_off & set(phase):
                minimum = 1 if key == 'on_duration_slots' else 0
                if type(phase[key]) is not int or phase[key] < minimum:
                    raise ValueError('Invalid integer arrival phase parameter')
            period = phase['on_duration_slots'] + phase['off_duration_slots']
            if duration % period:
                raise ValueError('ON/OFF phase duration must contain whole traffic periods')
    return deepcopy(phases)


class PhasedArrivalModel(ArrivalModel):
    """Repeat phases in one run, preserving queues/channel/controller state.

    The simulator's first sampled slot is 1. Each phase's child sampler receives
    a zero-based local clock, so an ON/OFF phase begins with its full ON segment.
    Legacy standalone ON/OFF sampling keeps its original clock semantics.
    """
    def __init__(self, profile: TrafficProfile) -> None:
        self._phases = []
        self._period = 0
        for phase in validate_arrival_phases(profile.arrival_phases):
            duration = phase.pop('duration_slots')
            child = build_arrival_model(TrafficProfile(
                ue_id=profile.ue_id, packet_size_bytes=profile.packet_size_bytes, **phase))
            self._phases.append((self._period, self._period + duration, child))
            self._period += duration

    def sample_bytes(self, *, slot_id: int, slot_duration_s: float, rng: random.Random) -> int:
        if type(slot_id) is not int or slot_id < 1:
            raise ValueError('Phased arrivals use simulator slots starting at 1')
        position = (slot_id - 1) % self._period
        for start, end, child in self._phases:
            if position < end:
                return child.sample_bytes(slot_id=position-start, slot_duration_s=slot_duration_s, rng=rng)
        raise AssertionError('Unreachable phased arrival position')


def build_arrival_model(profile: TrafficProfile) -> ArrivalModel:
    model = str(profile.arrival_model or "none").lower()
    if model == 'phased':
        return PhasedArrivalModel(profile)
    if profile.arrival_phases is not None:
        raise ValueError('arrival_phases requires arrival_model=phased')
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
