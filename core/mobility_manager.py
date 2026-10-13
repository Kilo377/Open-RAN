"""
Lightweight serving-cell, mobility, and traffic-steering manager.
"""

from __future__ import annotations

import math
import random
from typing import Any

from core.rf_interference import (
    build_frequency_interval_hz,
    compute_frequency_coupling_fraction,
    resolve_cell_rf_profile,
)
from state.candidate_cell_state import CandidateCellState


class MobilityManager:
    def __init__(self, *, slot_duration_ms: float, rng_seed: int = 1) -> None:
        self.slot_duration_s = max(0.0, float(slot_duration_ms)) / 1000.0
        self.rng_seed = int(rng_seed)

    def _dbm_to_mw(self, dbm_value: float) -> float:
        clamped_dbm = max(-200.0, min(120.0, float(dbm_value)))
        return 10.0 ** (clamped_dbm / 10.0)

    def _mw_to_dbm(self, mw_value: float) -> float:
        if float(mw_value) <= 0.0:
            return -120.0
        return 10.0 * math.log10(float(mw_value))

    def _resolve_shadowing_sigma_db(
        self,
        *,
        link: Any,
        scenario_type: str,
        los_state: str,
    ) -> float:
        large_scale = getattr(link, "large_scale", None)
        if large_scale is not None:
            raw_sigma = getattr(large_scale, "shadowing_sigma_db", None)
            if raw_sigma is not None:
                return max(0.1, min(20.0, abs(float(raw_sigma))))

        raw_sigma = getattr(link, "shadowing_sigma_db", None)
        if raw_sigma is not None:
            return max(0.1, min(20.0, abs(float(raw_sigma))))

        normalized = str(scenario_type).strip().lower()
        normalized_los = str(los_state).strip().lower()
        if normalized.startswith("uma"):
            default_sigma = 5.0 if normalized_los == "los" else 9.0
        elif normalized.startswith("umi"):
            default_sigma = 4.0 if normalized_los == "los" else 7.5
        elif normalized.startswith("inh"):
            default_sigma = 3.0 if normalized_los == "los" else 5.0
        elif normalized.startswith("rma"):
            default_sigma = 4.0 if normalized_los == "los" else 8.0
        else:
            default_sigma = 4.0 if normalized_los == "los" else 7.0
        return float(default_sigma)

    def _iter_ues(self, state: Any) -> list[Any]:
        ues = getattr(state, "ues", [])
        if isinstance(ues, dict):
            return list(ues.values())
        if isinstance(ues, list):
            return ues
        return []

    def _get_connection(self, state: Any, ue_id: str) -> Any | None:
        connections = getattr(state, "connections", {})
        if isinstance(connections, dict):
            return connections.get(str(ue_id))
        return None

    def _get_link(self, state: Any, ue_id: str) -> Any | None:
        links = getattr(state, "links", {})
        if isinstance(links, dict):
            return links.get(str(ue_id))
        return None

    def _managed_cells(self, state: Any) -> dict[str, dict[str, float]]:
        cell = getattr(state, "cell", None)
        cell_id = str(getattr(cell, "cell_id", "cell_0"))
        positions = dict(getattr(cell, "neighbor_cell_positions", {}) or {})
        if cell_id not in positions:
            positions[cell_id] = {"x_m": 0.0, "y_m": 0.0, "z_m": 25.0}
        return positions

    def _cell_radio_profiles(self, state: Any) -> dict[str, dict[str, Any]]:
        cell = getattr(state, "cell", None)
        antenna = getattr(state, "antenna", None)
        raw_profiles = dict(getattr(cell, "cell_radio_profiles", {}) or {})
        default_profile = {
            "cell_type": "macro",
            "carrier_frequency_hz": float(getattr(antenna, "carrier_frequency_hz", 3.5e9)),
            "bandwidth_hz": float(getattr(antenna, "bandwidth_hz", 20e6)),
            "subcarrier_spacing_hz": float(getattr(antenna, "subcarrier_spacing_hz", 30e3)),
            "tx_power_dbm": float(getattr(cell, "tx_power_dbm", 30.0)),
            "scenario_type": "umi_street_canyon",
        }
        merged_profiles: dict[str, dict[str, Any]] = {}
        for cell_id in self._managed_cells(state).keys():
            profile = dict(default_profile)
            profile.update(dict(raw_profiles.get(str(cell_id), {}) or {}))
            merged_profiles[str(cell_id)] = resolve_cell_rf_profile(profile)
        return merged_profiles

    def _cell_sleep_modes(self, state: Any) -> dict[str, str]:
        managed_cells = getattr(state, "managed_cells", {})
        if not isinstance(managed_cells, dict):
            managed_cells = {}
        sleep_modes: dict[str, str] = {}
        for cell_id, runtime_state in managed_cells.items():
            runtime_cell = getattr(runtime_state, "cell", None)
            sleep_modes[str(cell_id)] = str(
                getattr(runtime_cell, "cell_sleep_mode", "active")
            ).strip().lower()
        primary_cell = getattr(state, "cell", None)
        primary_cell_id = str(getattr(primary_cell, "cell_id", "cell_0"))
        sleep_modes.setdefault(
            primary_cell_id,
            str(getattr(primary_cell, "cell_sleep_mode", "active")).strip().lower(),
        )
        return sleep_modes

    def _build_cell_load_view(self, state: Any, cell_radio_profiles: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
        scheduler = getattr(state, "scheduler", None)
        scheduled_ue_ids = {str(ue_id) for ue_id in list(getattr(scheduler, "scheduled_ue_ids", []) or [])}
        load_view: dict[str, dict[str, float]] = {}
        for cell_id, profile in cell_radio_profiles.items():
            bandwidth_mhz = max(1.0, float(profile.get("bandwidth_hz", 20e6)) / 1e6)
            num_tx_ant = max(1.0, float(profile.get("gnb_num_tx_ant", 4)))
            capacity_units = max(1.0, bandwidth_mhz * math.sqrt(num_tx_ant))
            load_view[str(cell_id)] = {
                "connected_ue_count": 0.0,
                "scheduled_ue_count": 0.0,
                "queued_bytes": 0.0,
                "capacity_units": float(capacity_units),
                "estimated_load": 0.0,
                "load_penalty_db": 0.0,
                "activity_factor": 0.0,
            }

        for ue in self._iter_ues(state):
            ue_id = str(getattr(ue, "ue_id", ""))
            if ue_id == "":
                continue
            connection = self._get_connection(state, ue_id)
            serving_cell_id = str(
                getattr(connection, "serving_cell_id", getattr(self._get_link(state, ue_id), "serving_cell_id", ""))
            )
            if serving_cell_id not in load_view:
                continue
            load_entry = load_view[serving_cell_id]
            load_entry["connected_ue_count"] += 1.0
            load_entry["queued_bytes"] += float(max(0, int(getattr(ue, "queue_bytes", 0))))
            if ue_id in scheduled_ue_ids:
                load_entry["scheduled_ue_count"] += 1.0

        for load_entry in load_view.values():
            capacity_units = max(1.0, float(load_entry["capacity_units"]))
            connected_pressure = float(load_entry["connected_ue_count"]) / max(1.0, capacity_units / 18.0)
            scheduled_pressure = float(load_entry["scheduled_ue_count"]) / max(1.0, capacity_units / 24.0)
            queue_pressure = float(load_entry["queued_bytes"]) / max(1.0, capacity_units * 900.0)
            estimated_load = min(2.0, 0.35 * connected_pressure + 0.25 * scheduled_pressure + 0.40 * queue_pressure)
            load_entry["estimated_load"] = float(estimated_load)
            load_entry["load_penalty_db"] = float(12.0 * estimated_load)
            load_entry["activity_factor"] = float(
                min(1.0, 0.10 + 0.60 * estimated_load + 0.30 * min(1.0, scheduled_pressure))
            )
        return load_view

    def _compute_pathloss_db(
        self,
        *,
        scenario_type: str,
        los_state: str,
        distance_3d_m: float,
        ue_height_m: float,
        carrier_frequency_hz: float,
    ) -> float:
        d3d = max(1.0, float(distance_3d_m))
        fc_ghz = max(0.5, float(carrier_frequency_hz) / 1e9)
        normalized = str(scenario_type).strip().lower()

        if normalized.startswith("uma"):
            los_db = 28.0 + 22.0 * math.log10(d3d) + 20.0 * math.log10(fc_ghz)
            nlos_db = max(los_db, 13.54 + 39.08 * math.log10(d3d) + 20.0 * math.log10(fc_ghz) - 0.6 * (ue_height_m - 1.5))
            return nlos_db if los_state == "nlos" else los_db
        if normalized.startswith("inh"):
            los_db = 32.4 + 17.3 * math.log10(d3d) + 20.0 * math.log10(fc_ghz)
            nlos_db = max(los_db, 17.3 + 38.3 * math.log10(d3d) + 24.9 * math.log10(fc_ghz))
            return nlos_db if los_state == "nlos" else los_db
        if normalized.startswith("rma"):
            los_db = 32.4 + 20.0 * math.log10(fc_ghz) + 23.1 * math.log10(d3d)
            nlos_db = los_db + 8.0
            return nlos_db if los_state == "nlos" else los_db

        los_db = 32.4 + 21.0 * math.log10(d3d) + 20.0 * math.log10(fc_ghz)
        nlos_db = max(los_db, 35.3 + 22.4 * math.log10(d3d) + 21.3 * math.log10(fc_ghz) - 0.3 * (ue_height_m - 1.5))
        return nlos_db if los_state == "nlos" else los_db

    def _compute_candidate_cells(
        self,
        state: Any,
        ue: Any,
        link: Any,
        connection: Any | None,
        cell_load_view: dict[str, dict[str, float]],
    ) -> list[CandidateCellState]:
        positions = self._managed_cells(state)
        cell_radio_profiles = self._cell_radio_profiles(state)
        cell_sleep_modes = self._cell_sleep_modes(state)
        serving_cell_id = str(
            getattr(connection, "serving_cell_id", getattr(link, "serving_cell_id", getattr(getattr(state, "cell", None), "cell_id", "cell_0")))
        )
        mobility_bias_db = float(getattr(getattr(state, "cell", None), "mobility_bias_db", 0.0))
        steering_preference = str(getattr(getattr(state, "cell", None), "traffic_steering_preference", "balanced")).strip().lower()
        ue_id = str(getattr(ue, "ue_id", "ue"))
        candidate_inputs: list[dict[str, float | str | int | bool]] = []
        for cell_rank, (cell_id, position) in enumerate(sorted(positions.items(), key=lambda item: str(item[0]))):
            cell_profile = dict(cell_radio_profiles.get(str(cell_id), {}) or {})
            sleep_mode = str(cell_sleep_modes.get(str(cell_id), "active")).strip().lower()
            carrier_frequency_hz = float(cell_profile.get("carrier_frequency_hz", 3.5e9))
            tx_power_dbm = float(
                cell_profile.get("tx_power_dbm", getattr(getattr(state, "cell", None), "tx_power_dbm", 30.0))
            )
            if sleep_mode == "micro_sleep":
                tx_power_dbm -= 3.0
            elif sleep_mode == "light_sleep":
                tx_power_dbm -= 8.0
            elif sleep_mode == "deep_sleep":
                tx_power_dbm -= 25.0
            scenario_type = str(cell_profile.get("scenario_type", getattr(link, "scenario_type", "umi_street_canyon")))
            dx_m = float(getattr(link, "x_m", 0.0)) - float(position.get("x_m", 0.0))
            dy_m = float(getattr(link, "y_m", 0.0)) - float(position.get("y_m", 0.0))
            dz_m = float(position.get("z_m", 25.0)) - float(getattr(link, "z_m", 1.5))
            distance_2d_m = math.sqrt(dx_m * dx_m + dy_m * dy_m)
            distance_3d_m = math.sqrt(distance_2d_m * distance_2d_m + dz_m * dz_m)
            los_state = str(getattr(link, "los_state", "los"))
            pathloss_db = self._compute_pathloss_db(
                scenario_type=scenario_type,
                los_state=los_state,
                distance_3d_m=distance_3d_m,
                ue_height_m=float(getattr(link, "z_m", 1.5)),
                carrier_frequency_hz=carrier_frequency_hz,
            )
            shadowing_rng = random.Random(
                self.rng_seed
                + 97 * int("".join(ch for ch in ue_id if ch.isdigit()) or 0)
                + 7919 * cell_rank
                + int(getattr(link, "spatial_corr_seed", 0))
            )
            shadowing_sigma_db = self._resolve_shadowing_sigma_db(
                link=link,
                scenario_type=scenario_type,
                los_state=los_state,
            )
            shadowing_db = float(shadowing_rng.gauss(0.0, shadowing_sigma_db))
            o2i_penetration_loss_db = float(getattr(link, "o2i_penetration_loss_db", 0.0))
            rsrp_dbm = float(tx_power_dbm - pathloss_db - shadowing_db - o2i_penetration_loss_db)
            quality_score_db = float(rsrp_dbm)
            if cell_id != serving_cell_id:
                if steering_preference == "edge_offload":
                    quality_score_db += mobility_bias_db
                elif steering_preference == "stay_put":
                    quality_score_db -= abs(mobility_bias_db)
            candidate_beam_id = int(
                round(((math.degrees(math.atan2(dy_m, dx_m)) + 360.0) % 360.0) / 45.0)
            ) % 8
            candidate_inputs.append(
                {
                    "cell_id": str(cell_id),
                    "cell_type": str(cell_profile.get("cell_type", "macro")),
                    "x_m": float(position.get("x_m", 0.0)),
                    "y_m": float(position.get("y_m", 0.0)),
                    "z_m": float(position.get("z_m", 25.0)),
                    "distance_2d_m": float(distance_2d_m),
                    "distance_3d_m": float(distance_3d_m),
                    "carrier_frequency_hz": float(carrier_frequency_hz),
                    "bandwidth_hz": float(cell_profile.get("bandwidth_hz", 20e6)),
                    "subcarrier_spacing_hz": float(cell_profile.get("subcarrier_spacing_hz", 30e3)),
                    "rf_band": str(cell_profile.get("rf_band", "sub6")),
                    "adjacent_channel_leakage_ratio_db": float(
                        cell_profile.get("adjacent_channel_leakage_ratio_db", 45.0)
                    ),
                    "adjacent_channel_selectivity_db": float(
                        cell_profile.get("adjacent_channel_selectivity_db", 33.0)
                    ),
                    "out_of_band_isolation_db": float(
                        cell_profile.get("out_of_band_isolation_db", 42.0)
                    ),
                    "tx_power_dbm": float(tx_power_dbm),
                    "pathloss_db": float(pathloss_db),
                    "shadowing_db": float(shadowing_db),
                    "rsrp_dbm": float(rsrp_dbm),
                    "candidate_beam_id": int(candidate_beam_id),
                    "is_serving": str(cell_id) == serving_cell_id,
                    "is_target": str(cell_id) == str(getattr(connection, "handover_pending_target_cell_id", None)),
                }
            )

        candidates: list[CandidateCellState] = []
        for candidate_input in candidate_inputs:
            interference_mw = 0.0
            for interferer in candidate_inputs:
                if str(interferer["cell_id"]) == str(candidate_input["cell_id"]):
                    continue
                frequency_coupling_fraction, _, _, _ = compute_frequency_coupling_fraction(
                    target_interval_hz=build_frequency_interval_hz(
                        carrier_frequency_hz=float(candidate_input["carrier_frequency_hz"]),
                        bandwidth_hz=float(candidate_input["bandwidth_hz"]),
                    ),
                    interferer_interval_hz=build_frequency_interval_hz(
                        carrier_frequency_hz=float(interferer["carrier_frequency_hz"]),
                        bandwidth_hz=float(interferer["bandwidth_hz"]),
                    ),
                    target_rf_profile={
                        "cell_type": candidate_input["cell_type"],
                        "carrier_frequency_hz": candidate_input["carrier_frequency_hz"],
                        "rf_band": candidate_input["rf_band"],
                        "adjacent_channel_leakage_ratio_db": candidate_input[
                            "adjacent_channel_leakage_ratio_db"
                        ],
                        "adjacent_channel_selectivity_db": candidate_input[
                            "adjacent_channel_selectivity_db"
                        ],
                        "out_of_band_isolation_db": candidate_input[
                            "out_of_band_isolation_db"
                        ],
                    },
                    interferer_rf_profile={
                        "cell_type": interferer["cell_type"],
                        "carrier_frequency_hz": interferer["carrier_frequency_hz"],
                        "rf_band": interferer["rf_band"],
                        "adjacent_channel_leakage_ratio_db": interferer[
                            "adjacent_channel_leakage_ratio_db"
                        ],
                        "adjacent_channel_selectivity_db": interferer[
                            "adjacent_channel_selectivity_db"
                        ],
                        "out_of_band_isolation_db": interferer[
                            "out_of_band_isolation_db"
                        ],
                    },
                    enable_cross_frequency_interference=True,
                    far_frequency_leakage_floor=1e-7,
                    guard_decay_scale=1.0,
                )
                if frequency_coupling_fraction <= 0.0:
                    continue
                interferer_load = dict(cell_load_view.get(str(interferer["cell_id"]), {}) or {})
                activity_factor = float(interferer_load.get("activity_factor", 0.0))
                interference_mw += (
                    self._dbm_to_mw(float(interferer["rsrp_dbm"]))
                    * float(frequency_coupling_fraction)
                    * activity_factor
                )

            bandwidth_hz = float(candidate_input["bandwidth_hz"])
            thermal_noise_dbm = -174.0 + 10.0 * math.log10(max(1.0, bandwidth_hz)) + 7.0
            denominator_dbm = self._mw_to_dbm(self._dbm_to_mw(thermal_noise_dbm) + interference_mw)
            interference_db = self._mw_to_dbm(interference_mw)
            effective_margin_db = float(candidate_input["rsrp_dbm"]) - denominator_dbm
            load_entry = dict(cell_load_view.get(str(candidate_input["cell_id"]), {}) or {})
            load_penalty_db = float(load_entry.get("load_penalty_db", 0.0))
            steering_adjustment_db = 0.0
            if str(candidate_input["cell_id"]) != serving_cell_id:
                if steering_preference == "edge_offload":
                    steering_adjustment_db += mobility_bias_db
                elif steering_preference == "stay_put":
                    steering_adjustment_db -= abs(mobility_bias_db)
            sleep_mode = str(cell_sleep_modes.get(str(candidate_input["cell_id"]), "active")).strip().lower()
            sleep_penalty_db = 0.0
            if sleep_mode == "micro_sleep":
                sleep_penalty_db = 2.0
            elif sleep_mode == "light_sleep":
                sleep_penalty_db = 6.0
            elif sleep_mode == "deep_sleep":
                sleep_penalty_db = 18.0
            steering_score_db = effective_margin_db - load_penalty_db - sleep_penalty_db + steering_adjustment_db
            candidates.append(
                CandidateCellState(
                    cell_id=str(candidate_input["cell_id"]),
                    cell_type=str(candidate_input["cell_type"]),
                    x_m=float(candidate_input["x_m"]),
                    y_m=float(candidate_input["y_m"]),
                    z_m=float(candidate_input["z_m"]),
                    distance_2d_m=float(candidate_input["distance_2d_m"]),
                    distance_3d_m=float(candidate_input["distance_3d_m"]),
                    carrier_frequency_hz=float(candidate_input["carrier_frequency_hz"]),
                    bandwidth_hz=float(candidate_input["bandwidth_hz"]),
                    subcarrier_spacing_hz=float(candidate_input["subcarrier_spacing_hz"]),
                    rf_band=str(candidate_input["rf_band"]),
                    adjacent_channel_leakage_ratio_db=float(
                        candidate_input["adjacent_channel_leakage_ratio_db"]
                    ),
                    adjacent_channel_selectivity_db=float(
                        candidate_input["adjacent_channel_selectivity_db"]
                    ),
                    out_of_band_isolation_db=float(
                        candidate_input["out_of_band_isolation_db"]
                    ),
                    tx_power_dbm=float(candidate_input["tx_power_dbm"]),
                    connected_ue_count=int(load_entry.get("connected_ue_count", 0.0)),
                    scheduled_ue_count=int(load_entry.get("scheduled_ue_count", 0.0)),
                    queued_bytes=int(load_entry.get("queued_bytes", 0.0)),
                    estimated_load=float(load_entry.get("estimated_load", 0.0)),
                    load_penalty_db=float(load_penalty_db),
                    estimated_interference_db=float(interference_db),
                    interference_penalty_db=float(max(0.0, interference_db - thermal_noise_dbm)),
                    steering_score_db=float(steering_score_db),
                    pathloss_db=float(candidate_input["pathloss_db"]),
                    shadowing_db=float(candidate_input["shadowing_db"]),
                    rsrp_dbm=float(candidate_input["rsrp_dbm"]),
                    quality_score_db=float(steering_score_db),
                    candidate_beam_id=int(candidate_input["candidate_beam_id"]),
                    is_serving=bool(candidate_input["is_serving"]),
                    is_target=bool(candidate_input["is_target"]),
                )
            )
        candidates.sort(key=lambda item: float(item.quality_score_db), reverse=True)
        return candidates

    def _advance_geometry(self, link: Any) -> None:
        velocity_mps = float(getattr(link, "velocity_mps", 0.0))
        if velocity_mps == 0.0 or self.slot_duration_s <= 0.0:
            return
        heading_rad = math.radians(float(getattr(link, "heading_deg", 0.0)))
        link.x_m = float(getattr(link, "x_m", 0.0)) + velocity_mps * self.slot_duration_s * math.cos(heading_rad)
        link.y_m = float(getattr(link, "y_m", 0.0)) + velocity_mps * self.slot_duration_s * math.sin(heading_rad)

    def _apply_serving_candidate(self, ue: Any, link: Any, connection: Any | None, candidate: CandidateCellState) -> None:
        link.serving_cell_id = str(candidate.cell_id)
        link.distance_2d_m = float(candidate.distance_2d_m)
        link.distance_3d_m = float(candidate.distance_3d_m)
        link.pathloss_db = float(candidate.pathloss_db)
        link.shadowing_db = float(candidate.shadowing_db)
        link.refresh_propagation_views()
        setattr(ue, "serving_beam_id", int(candidate.candidate_beam_id))
        if connection is not None:
            connection.serving_cell_id = str(candidate.cell_id)
            connection.serving_beam_id = int(candidate.candidate_beam_id)
            connection.traffic_steering_bias_db = float(
                getattr(connection, "traffic_steering_bias_db", 0.0)
            )

    def _compute_measurement_event_met(
        self,
        *,
        event_trigger: str,
        trigger_margin_db: float,
        event_offset_db: float,
        best_candidate: CandidateCellState,
        current_candidate: CandidateCellState,
    ) -> bool:
        normalized_trigger = str(event_trigger).strip().lower()
        if normalized_trigger in {"", "disabled", "none"}:
            return False
        if normalized_trigger == "strongest_cell":
            return str(best_candidate.cell_id) != str(current_candidate.cell_id)
        if normalized_trigger == "a5_quality_floor":
            return (
                float(current_candidate.signal_margin_db) <= float(event_offset_db)
                and float(best_candidate.signal_margin_db) > float(current_candidate.signal_margin_db)
            )
        return float(trigger_margin_db) >= float(event_offset_db)

    def update(self, state: Any, *, scoped_cell_policy: bool = False) -> None:
        handover_margin_db = float(getattr(getattr(state, "cell", None), "handover_margin_db", 3.0))
        handover_delay_slots = max(1, int(getattr(getattr(state, "cell", None), "handover_delay_slots", 2)))
        handover_cost_db = max(0.0, float(getattr(getattr(state, "cell", None), "handover_cost_db", 0.0)))
        handover_interruption_slots = max(
            0,
            int(getattr(getattr(state, "cell", None), "handover_interruption_slots", 0)),
        )
        steering_hysteresis_db = max(
            0.0,
            float(getattr(getattr(state, "cell", None), "steering_hysteresis_db", 0.0)),
        )
        measurement_event_trigger = str(
            getattr(getattr(state, "cell", None), "measurement_event_trigger", "a3_offset")
        ).strip().lower()
        measurement_event_offset_db = float(
            getattr(getattr(state, "cell", None), "measurement_event_offset_db", 0.0)
        )
        managed_cell_ids = list(getattr(getattr(state, "cell", None), "managed_cell_ids", []) or [])
        if len(managed_cell_ids) == 0:
            state.cell.managed_cell_ids = sorted(self._managed_cells(state).keys())
        cell_load_view = self._build_cell_load_view(state, self._cell_radio_profiles(state))

        for ue in self._iter_ues(state):
            ue_id = str(getattr(ue, "ue_id", ""))
            if ue_id == "":
                continue
            connection = self._get_connection(state, ue_id)
            link = self._get_link(state, ue_id)
            if link is None:
                continue

            if scoped_cell_policy:
                # RT association controls belong to the UE's current serving
                # cell, including while a handover is pending.
                policy = state.managed_cells[str(connection.serving_cell_id)].cell
                handover_margin_db = float(policy.handover_margin_db)
                handover_delay_slots = max(1, int(policy.handover_delay_slots))
                handover_cost_db = max(0., float(policy.handover_cost_db))
                handover_interruption_slots = max(0, int(policy.handover_interruption_slots))
                steering_hysteresis_db = max(0., float(policy.steering_hysteresis_db))
                measurement_event_trigger = str(policy.measurement_event_trigger).strip().lower()
                measurement_event_offset_db = float(policy.measurement_event_offset_db)

            self._advance_geometry(link)
            candidates = self._compute_candidate_cells(state, ue, link, connection, cell_load_view)
            if connection is not None:
                connection.candidate_cells = list(candidates)
                connection.handover_cost_db = float(handover_cost_db)
                connection.handover_interruption_slots = int(handover_interruption_slots)
                connection.steering_hysteresis_db = float(steering_hysteresis_db)
                connection.measurement_event_trigger = str(measurement_event_trigger)
                connection.measurement_event_offset_db = float(measurement_event_offset_db)
            if len(candidates) == 0:
                continue

            best_candidate = candidates[0]
            current_candidate = next(
                (
                    candidate
                    for candidate in candidates
                    if str(candidate.cell_id)
                    == str(getattr(connection, "serving_cell_id", getattr(link, "serving_cell_id", "")))
                ),
                best_candidate,
            )

            current_candidate.steering_hysteresis_db = float(steering_hysteresis_db)
            current_candidate.handover_cost_db = 0.0
            current_candidate.hysteresis_adjusted_score_db = float(current_candidate.steering_score_db)
            current_candidate.trigger_margin_db = 0.0
            current_candidate.measurement_event_trigger = str(measurement_event_trigger)
            current_candidate.measurement_event_offset_db = float(measurement_event_offset_db)
            current_candidate.measurement_event_met = False

            for candidate in candidates:
                is_serving_candidate = str(candidate.cell_id) == str(current_candidate.cell_id)
                candidate.handover_cost_db = 0.0 if is_serving_candidate else float(handover_cost_db)
                candidate.steering_hysteresis_db = 0.0 if is_serving_candidate else float(steering_hysteresis_db)
                candidate.signal_margin_db = float(candidate.rsrp_dbm) - float(candidate.estimated_interference_db)
                candidate.hysteresis_adjusted_score_db = float(
                    candidate.steering_score_db
                    - (0.0 if is_serving_candidate else (handover_cost_db + steering_hysteresis_db))
                )
                candidate.trigger_margin_db = float(candidate.steering_score_db) - float(current_candidate.steering_score_db)
                candidate.measurement_event_trigger = str(measurement_event_trigger)
                candidate.measurement_event_offset_db = float(measurement_event_offset_db)
                candidate.measurement_event_met = self._compute_measurement_event_met(
                    event_trigger=measurement_event_trigger,
                    trigger_margin_db=float(candidate.trigger_margin_db),
                    event_offset_db=float(measurement_event_offset_db),
                    best_candidate=candidate,
                    current_candidate=current_candidate,
                ) if not is_serving_candidate else False

            self._apply_serving_candidate(ue, link, connection, current_candidate)

            if connection is None:
                continue

            current_state = str(getattr(connection, "connection_state", "connected"))
            pending_target = getattr(connection, "handover_pending_target_cell_id", None)
            interruption_remaining_slots = int(
                max(0, int(getattr(connection, "interruption_remaining_slots", 0) or 0))
            )
            if current_state == "handover_interruption":
                if interruption_remaining_slots > 0:
                    connection.interruption_remaining_slots = int(interruption_remaining_slots - 1)
                if int(getattr(connection, "interruption_remaining_slots", 0)) <= 0:
                    connection.connection_state = "connected"
                    connection.interruption_remaining_slots = 0
                continue
            if current_state == "handover_pending" and pending_target is not None:
                pending_candidate = next(
                    (
                        candidate
                        for candidate in candidates
                        if str(candidate.cell_id) == str(pending_target)
                    ),
                    None,
                )
                if pending_candidate is None or not bool(getattr(pending_candidate, "measurement_event_met", False)):
                    connection.connection_state = "connected"
                    connection.handover_pending_target_cell_id = None
                    connection.handover_started_slot = None
                    connection.last_trigger_reason = "handover_cancelled_trigger_lost"
                elif int(getattr(state, "slot_id", 0)) - int(getattr(connection, "handover_started_slot", 0) or 0) >= handover_delay_slots:
                    self._apply_serving_candidate(ue, link, connection, pending_candidate)
                    if handover_interruption_slots > 0:
                        connection.connection_state = "handover_interruption"
                        connection.interruption_remaining_slots = int(handover_interruption_slots)
                    else:
                        connection.connection_state = "connected"
                    connection.last_handover_slot = int(getattr(state, "slot_id", 0))
                    connection.handover_count = int(getattr(connection, "handover_count", 0)) + 1
                    connection.last_handover_cost_db = float(handover_cost_db)
                    connection.handover_pending_target_cell_id = None
                    connection.handover_started_slot = None
                    connection.last_trigger_reason = "handover_completed"
                continue

            if str(best_candidate.cell_id) != str(current_candidate.cell_id):
                gain_db = float(best_candidate.steering_score_db) - float(current_candidate.steering_score_db)
                trigger_threshold_db = float(
                    handover_margin_db + steering_hysteresis_db + handover_cost_db
                )
                measurement_event_met = bool(getattr(best_candidate, "measurement_event_met", False))
                connection.last_measurement_event_slot = int(getattr(state, "slot_id", 0)) if measurement_event_met else getattr(connection, "last_measurement_event_slot", None)
                connection.last_measurement_event_type = (
                    str(measurement_event_trigger) if measurement_event_met else getattr(connection, "last_measurement_event_type", None)
                )
                connection.last_measurement_event_margin_db = float(gain_db)
                if measurement_event_met and gain_db >= trigger_threshold_db:
                    connection.connection_state = "handover_pending"
                    connection.handover_pending_target_cell_id = str(best_candidate.cell_id)
                    connection.handover_started_slot = int(getattr(state, "slot_id", 0))
                    connection.preferred_target_cell_id = str(best_candidate.cell_id)
                    connection.last_trigger_reason = f"{measurement_event_trigger}_met"
                else:
                    connection.connection_state = "connected"
                    connection.handover_pending_target_cell_id = None
                    connection.handover_started_slot = None
                    if measurement_event_met:
                        connection.last_trigger_reason = "trigger_met_but_below_handover_threshold"
                    else:
                        connection.last_trigger_reason = "measurement_event_not_met"
            elif current_state != "out_of_sync":
                connection.connection_state = "connected"
                connection.handover_pending_target_cell_id = None
                connection.handover_started_slot = None
                connection.last_trigger_reason = "serving_cell_retained"
