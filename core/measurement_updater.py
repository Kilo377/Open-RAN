"""core/measurement_updater.py

Maintains a controller-visible measurement view that can differ from the
internal truth state.

This module implements a minimal approximation of KPM-like measurement
effects:
- queue reporting delay (in slots)
- SINR window smoothing
- BLER estimated from a window of recent TX ACK/NACK samples
- throughput aggregated over a granularity period (in slots)
"""

from __future__ import annotations

import math
import random
from collections import deque

from core.runtime_views import get_cluster_scheduled_ue_ids, get_managed_cells
from state.measurement_state import MeasurementState, UeMeasurement


class MeasurementUpdater:
    def __init__(
        self,
        *,
        rng_seed: int,
        slot_duration_ms: float,
        queue_delay_slots: int,
        sinr_window_slots: int,
        sinr_noise_std_db: float,
        bler_window_tbs: int,
        throughput_granularity_slots: int,
        noise_floor_dbm: float = -94.0,
    ) -> None:
        self._rng_seed = int(rng_seed)
        self._slot_duration_s = max(float(slot_duration_ms), 1e-6) / 1000.0

        self._queue_delay_slots = max(0, int(queue_delay_slots))
        self._sinr_window_slots = max(1, int(sinr_window_slots))
        self._sinr_noise_std_db = max(0.0, float(sinr_noise_std_db))
        self._bler_window_tbs = max(1, int(bler_window_tbs))
        self._throughput_granularity_slots = max(1, int(throughput_granularity_slots))
        self._noise_floor_dbm = float(noise_floor_dbm)

        self._queue_hist: dict[str, deque[int]] = {}
        self._sinr_ewma: dict[str, float] = {}
        self._sinr_sample_count: dict[str, int] = {}
        self._tx_ack_hist: dict[str, deque[bool]] = {}
        self._acked_bytes_hist: dict[str, deque[int]] = {}
        self._bler_estimate: dict[str, float] = {}

    def _per_ue_slot_rng(self, ue_id: str, slot_id: int) -> random.Random:
        ue_hash = sum(ord(ch) for ch in str(ue_id))
        seed = 1000003 * int(self._rng_seed) + 37 * int(slot_id) + 19 * int(ue_hash)
        return random.Random(seed)

    def _ewma_alpha(self) -> float:
        window = max(1, int(self._sinr_window_slots))
        # A common EWMA mapping from window length to alpha.
        return 2.0 / float(window + 1)

    def update(self, state) -> None:
        measurement: MeasurementState | None = getattr(state, "measurement", None)
        if measurement is None:
            measurement = MeasurementState()
            setattr(state, "measurement", measurement)

        slot_id = int(getattr(state, "slot_id", 0))
        measurement.produced_at_slot = slot_id
        measurement.delivered_at_slot = None
        measurement.age_slots = 0
        measurement.last_measurement_update_slot = slot_id
        measurement.queue_delay_slots = int(self._queue_delay_slots)
        measurement.sinr_window_slots = int(self._sinr_window_slots)
        measurement.sinr_noise_std_db_applied = float(self._sinr_noise_std_db)
        measurement.bler_window_tbs = int(self._bler_window_tbs)
        measurement.throughput_granularity_slots = int(self._throughput_granularity_slots)

        cell = getattr(state, "cell", None)
        channel = getattr(state, "channel", None)
        managed_cells = get_managed_cells(state)

        # Keep these in sync with the PHY gain defaults.
        codebook_gain_step_db = 1.0
        beam_profile_gain_step_db = 0.8

        scheduled_set = set(get_cluster_scheduled_ue_ids(state))

        for ue in list(getattr(state, "ues", [])):
            ue_id = str(getattr(ue, "ue_id", "unknown_ue"))
            connections = getattr(state, "connections", {})
            connection = connections.get(ue_id) if isinstance(connections, dict) else None
            serving_cell_id = str(
                getattr(connection, "serving_cell_id", getattr(cell, "cell_id", "cell_0"))
            )
            runtime_state = managed_cells.get(serving_cell_id)
            serving_cell = getattr(runtime_state, "cell", cell)
            serving_channel = getattr(runtime_state, "channel", channel)

            tx_power_dbm = (
                float(getattr(serving_cell, "tx_power_dbm", 30.0))
                if serving_cell is not None
                else 30.0
            )
            codebook_id = (
                int(getattr(serving_cell, "current_codebook_id", 0))
                if serving_cell is not None
                else 0
            )
            beam_profile_id = (
                int(getattr(serving_cell, "current_beam_profile_id", 0))
                if serving_cell is not None
                else 0
            )
            pathloss_db = (
                float(getattr(serving_channel, "pathloss_db", 0.0))
                if serving_channel is not None
                else 0.0
            )
            shadowing_db = (
                float(getattr(serving_channel, "shadowing_db", 0.0))
                if serving_channel is not None
                else 0.0
            )
            doppler_hz = (
                float(getattr(serving_channel, "doppler_hz", 0.0))
                if serving_channel is not None
                else 0.0
            )
            links = getattr(state, "links", {})
            link_state = links.get(ue_id) if isinstance(links, dict) else None
            large_scale = getattr(link_state, "large_scale", None) if link_state is not None else None
            small_scale = getattr(link_state, "small_scale", None) if link_state is not None else None
            ue_pathloss_db = float(
                getattr(large_scale, "pathloss_db", getattr(link_state, "pathloss_db", pathloss_db))
            )
            ue_shadowing_db = float(
                getattr(large_scale, "shadowing_db", getattr(link_state, "shadowing_db", shadowing_db))
            )
            ue_o2i_penetration_loss_db = float(
                getattr(
                    large_scale,
                    "o2i_penetration_loss_db",
                    getattr(link_state, "o2i_penetration_loss_db", 0.0),
                )
            )
            ue_doppler_hz = float(
                getattr(small_scale, "doppler_hz", getattr(link_state, "doppler_hz", doppler_hz))
            )
            ue_fast_fading_seed = int(
                getattr(small_scale, "fast_fading_seed", getattr(link_state, "fast_fading_seed", 0))
            )

            # Queue measurement with reporting delay.
            current_queue = int(getattr(ue, "queue_bytes", 0))
            q_hist = self._queue_hist.get(ue_id)
            if q_hist is None:
                q_hist = deque(maxlen=max(4, self._queue_delay_slots + 2))
                self._queue_hist[ue_id] = q_hist
            q_hist.append(current_queue)
            if self._queue_delay_slots > 0 and len(q_hist) > self._queue_delay_slots:
                measured_queue = int(q_hist[-(self._queue_delay_slots + 1)])
                queue_age_slots = int(self._queue_delay_slots)
            else:
                measured_queue = int(current_queue)
                queue_age_slots = int(max(0, len(q_hist) - 1))
            queue_produced_at_slot = int(max(0, slot_id - queue_age_slots))

            # SINR measurement independent of scheduling (avoids stale truth fields).
            rx_gain_db = float(getattr(ue, "radio_quality_db", 0.0))
            rng = self._per_ue_slot_rng(ue_id, slot_id)
            if ue_fast_fading_seed != 0:
                rng.seed(rng.randint(0, 2**31 - 1) + int(ue_fast_fading_seed) + int(ue_doppler_hz))
            small_scale_db = rng.gauss(0.0, self._sinr_noise_std_db) if self._sinr_noise_std_db > 0.0 else 0.0

            rx_power_dbm = (
                tx_power_dbm
                + rx_gain_db
                + codebook_gain_step_db * float(codebook_id)
                + beam_profile_gain_step_db * float(beam_profile_id)
                + small_scale_db
                - ue_pathloss_db
                - ue_shadowing_db
                - ue_o2i_penetration_loss_db
            )
            raw_sinr_db = (
                rx_power_dbm
                - float(self._noise_floor_dbm)
                - float(getattr(ue, "interference_penalty_db", 0.0) or 0.0)
            )

            alpha = self._ewma_alpha()
            prev = self._sinr_ewma.get(ue_id)
            if prev is None:
                smoothed_sinr = float(raw_sinr_db)
            else:
                smoothed_sinr = alpha * float(raw_sinr_db) + (1.0 - alpha) * float(prev)
            self._sinr_ewma[ue_id] = float(smoothed_sinr)

            self._sinr_sample_count[ue_id] = int(self._sinr_sample_count.get(ue_id, 0) + 1)
            sinr_effective_sample_count = int(
                min(self._sinr_sample_count[ue_id], int(self._sinr_window_slots))
            )

            # BLER estimate from recent scheduled TX samples.
            is_scheduled = ue_id in scheduled_set
            ack = bool(getattr(ue, "ack", False)) if is_scheduled else False

            tx_hist = self._tx_ack_hist.get(ue_id)
            if tx_hist is None:
                tx_hist = deque(maxlen=self._bler_window_tbs)
                self._tx_ack_hist[ue_id] = tx_hist
            if is_scheduled:
                tx_hist.append(bool(ack))

            if len(tx_hist) > 0:
                nack_count = sum(1 for ok in tx_hist if not ok)
                tx_count = len(tx_hist)
                # Jeffreys prior smoothing avoids exact 0/1.
                bler_est = (float(nack_count) + 0.5) / (float(tx_count) + 1.0)
                self._bler_estimate[ue_id] = float(bler_est)
            bler_value = float(self._bler_estimate.get(ue_id, float(getattr(ue, "bler", 0.1))))
            bler_window_sample_count = int(len(tx_hist))

            # Throughput aggregated over a granularity period.
            acked_bytes = 0
            if is_scheduled and ack:
                acked_bytes = int(getattr(ue, "transport_block_size_bytes", 0))
                if acked_bytes <= 0:
                    acked_bytes = int(getattr(ue, "tb_size_bytes", 0))

            bytes_hist = self._acked_bytes_hist.get(ue_id)
            if bytes_hist is None:
                bytes_hist = deque(maxlen=self._throughput_granularity_slots)
                self._acked_bytes_hist[ue_id] = bytes_hist
            bytes_hist.append(int(max(0, acked_bytes)))
            total_bytes = int(sum(bytes_hist))
            denom_s = float(len(bytes_hist)) * float(self._slot_duration_s)
            measured_throughput_mbps = (float(total_bytes) * 8.0) / 1e6 / max(denom_s, 1e-9)
            throughput_window_slots = int(len(bytes_hist))

            measurement.ue_measurements[ue_id] = UeMeasurement(
                ue_id=ue_id,
                queue_bytes=int(measured_queue),
                throughput_mbps=float(measured_throughput_mbps),
                sinr_db=float(smoothed_sinr),
                bler=float(bler_value),
                is_scheduled=bool(is_scheduled),
                ack=bool(ack),
                tb_size_bytes=int(acked_bytes),
                produced_at_slot=int(slot_id),
                delivered_at_slot=None,
                age_slots=0,
                queue_age_slots=int(queue_age_slots),
                queue_delay_slots_applied=int(queue_age_slots),
                queue_produced_at_slot=int(queue_produced_at_slot),
                sinr_window_len=int(self._sinr_window_slots),
                sinr_window_slots=int(self._sinr_window_slots),
                sinr_effective_sample_count=int(sinr_effective_sample_count),
                sinr_noise_std_db_applied=float(self._sinr_noise_std_db),
                bler_window_tbs=int(self._bler_window_tbs),
                bler_window_sample_count=int(bler_window_sample_count),
                bler_sample_count=int(bler_window_sample_count),
                throughput_window_slots=int(throughput_window_slots),
                throughput_granularity_slots=int(self._throughput_granularity_slots),
                last_measurement_update_slot=int(slot_id),
            )
