"""traffic/traffic_engine.py

Applies per-slot traffic arrivals into UE queues.
"""

from __future__ import annotations

import random

from traffic.arrival_model import TrafficProfile, ArrivalModel, build_arrival_model


class TrafficEngine:
    def __init__(
        self,
        *,
        ue_config: dict,
        slot_duration_ms: float,
        rng_seed: int,
    ) -> None:
        self._slot_duration_s = max(float(slot_duration_ms), 1e-6) / 1000.0
        self._rng_seed = int(rng_seed)

        self._profiles: dict[str, TrafficProfile] = {}
        self._models: dict[str, ArrivalModel] = {}
        self._channel_keys_by_ue: dict[str, list[str]] = {}

        for item in ue_config.get("ues", []):
            ue_id = str(item.get("ue_id", ""))
            if not ue_id:
                continue

            raw_logical_channels = list(item.get("logical_channels", []) or [])
            if len(raw_logical_channels) == 0:
                raw_logical_channels = [
                    {
                        "logical_channel_id": str(item.get("default_logical_channel_id", "lc_0")),
                        "arrival_model": item.get("arrival_model", "none"),
                        "arrival_rate_bps": item.get("arrival_rate_bps", 0.0),
                        "packet_size_bytes": item.get("packet_size_bytes", 200),
                        "burst_size_bytes": item.get("burst_size_bytes", 0),
                        "on_duration_slots": item.get("on_duration_slots", 50),
                        "off_duration_slots": item.get("off_duration_slots", 20),
                        "start_offset_slots": item.get("start_offset_slots", 0),
                        "max_queue_bytes": item.get("max_queue_bytes", None),
                    }
                ]

            for logical_channel_index, logical_channel_item in enumerate(raw_logical_channels):
                logical_channel_id = str(
                    logical_channel_item.get("logical_channel_id", f"lc_{logical_channel_index}")
                )
                channel_key = f"{ue_id}:{logical_channel_id}"
                profile = TrafficProfile(
                    ue_id=ue_id,
                    arrival_model=str(
                        logical_channel_item.get("arrival_model", item.get("arrival_model", "none"))
                    ),
                    arrival_rate_bps=float(
                        logical_channel_item.get("arrival_rate_bps", item.get("arrival_rate_bps", 0.0))
                    ),
                    packet_size_bytes=int(
                        logical_channel_item.get("packet_size_bytes", item.get("packet_size_bytes", 200))
                    ),
                    burst_size_bytes=int(
                        logical_channel_item.get("burst_size_bytes", item.get("burst_size_bytes", 0))
                    ),
                    on_duration_slots=int(
                        logical_channel_item.get("on_duration_slots", item.get("on_duration_slots", 50))
                    ),
                    off_duration_slots=int(
                        logical_channel_item.get("off_duration_slots", item.get("off_duration_slots", 20))
                    ),
                    start_offset_slots=int(
                        logical_channel_item.get("start_offset_slots", item.get("start_offset_slots", 0))
                    ),
                    max_queue_bytes=(
                        int(logical_channel_item["max_queue_bytes"])
                        if (
                            "max_queue_bytes" in logical_channel_item
                            and logical_channel_item["max_queue_bytes"] is not None
                        )
                        else (
                            int(item["max_queue_bytes"])
                            if ("max_queue_bytes" in item and item["max_queue_bytes"] is not None)
                            else None
                        )
                    ),
                )

                self._profiles[channel_key] = profile
                self._models[channel_key] = build_arrival_model(profile)
                self._channel_keys_by_ue.setdefault(ue_id, []).append(channel_key)

    def _per_ue_slot_rng(self, ue_id: str, slot_id: int) -> random.Random:
        ue_hash = sum(ord(ch) for ch in str(ue_id))
        seed = 700001 * int(self._rng_seed) + 37 * int(slot_id) + 19 * int(ue_hash)
        return random.Random(seed)

    def enable_packet_tracking(self, state):
        from traffic.packet_ledger import PacketLedger
        for ue in state.ues:
            ue.packet_ledger = PacketLedger(ue, self._slot_duration_s * 1000)

    def apply_arrivals(self, state) -> tuple[dict[str, int], dict[str, int]]:
        slot_id = int(getattr(state, "slot_id", 0))

        arrivals_by_ue: dict[str, int] = {}
        arrivals_by_logical_channel: dict[str, int] = {}
        for ue in list(getattr(state, "ues", [])):
            ue_id = str(getattr(ue, "ue_id", ""))
            if not ue_id:
                continue
            logical_channels = list(getattr(ue, "logical_channels", []) or [])
            logical_channel_map = {
                str(getattr(logical_channel, "logical_channel_id", "")): logical_channel
                for logical_channel in logical_channels
                if str(getattr(logical_channel, "logical_channel_id", "")) != ""
            }
            channel_keys = list(self._channel_keys_by_ue.get(ue_id, []))
            if logical_channels and channel_keys:
                ue_arrival_total = 0
                for channel_key in channel_keys:
                    profile = self._profiles.get(channel_key)
                    model = self._models.get(channel_key)
                    logical_channel_id = str(channel_key.split(":", 1)[1])
                    logical_channel = logical_channel_map.get(logical_channel_id)
                    if profile is None or model is None or logical_channel is None:
                        continue

                    rng = self._per_ue_slot_rng(channel_key, slot_id)
                    arrival_bytes = int(
                        max(
                            0,
                            model.sample_bytes(
                                slot_id=slot_id,
                                slot_duration_s=self._slot_duration_s,
                                rng=rng,
                            ),
                        )
                    )
                    current_queue = int(getattr(logical_channel, "queue_bytes", 0))
                    new_queue = current_queue + arrival_bytes
                    max_queue_bytes = getattr(logical_channel, "max_queue_bytes", None)
                    if max_queue_bytes is None:
                        max_queue_bytes = profile.max_queue_bytes
                    if max_queue_bytes is not None:
                        new_queue = min(int(max_queue_bytes), int(new_queue))
                    new_queue = max(0, int(new_queue))
                    ledger = getattr(ue, 'packet_ledger', None)
                    if ledger is not None:
                        logical_channel.max_queue_bytes = max_queue_bytes
                        ledger.add(logical_channel, arrival_bytes, slot_id, packet_size=profile.packet_size_bytes)
                        new_queue = logical_channel.queue_bytes

                    try:
                        setattr(logical_channel, "queue_bytes", int(new_queue))
                    except Exception:
                        pass

                    ue_arrival_total += int(arrival_bytes)
                    arrivals_by_logical_channel[channel_key] = int(arrival_bytes)

                try:
                    setattr(
                        ue,
                        "queue_bytes",
                        int(
                            sum(int(getattr(logical_channel, "queue_bytes", 0)) for logical_channel in logical_channels)
                        ),
                    )
                except Exception:
                    pass
                arrivals_by_ue[ue_id] = int(ue_arrival_total)
                if getattr(ue, 'packet_ledger', None) is not None:
                    ue.packet_ledger.refresh(slot_id)
                continue

            profile = self._profiles.get(ue_id)
            model = self._models.get(ue_id)
            if profile is None or model is None:
                arrivals_by_ue[ue_id] = 0
                continue

            rng = self._per_ue_slot_rng(ue_id, slot_id)
            arrival_bytes = int(
                max(
                    0,
                    model.sample_bytes(
                        slot_id=slot_id,
                        slot_duration_s=self._slot_duration_s,
                        rng=rng,
                    ),
                )
            )

            current_queue = int(getattr(ue, "queue_bytes", 0))
            new_queue = current_queue + arrival_bytes
            if profile.max_queue_bytes is not None:
                new_queue = min(int(profile.max_queue_bytes), int(new_queue))
            new_queue = max(0, int(new_queue))

            try:
                setattr(ue, "queue_bytes", int(new_queue))
            except Exception:
                pass

            arrivals_by_ue[ue_id] = int(arrival_bytes)

        return arrivals_by_ue, arrivals_by_logical_channel
