"""
Scene generator that expands a template into UE geometry and link defaults.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
import random
from typing import Any

from core.rf_interference import resolve_cell_rf_profile
from models.scenario import GeneratedScenarioBundle, ScenarioTemplate, UeGeometry


class ScenarioGenerator:
    def __init__(self) -> None:
        pass

    def _build_cell_layout(
        self,
        *,
        template: ScenarioTemplate,
        cell_config: dict[str, Any],
    ) -> dict[str, dict[str, float]]:
        topology = template.topology
        num_cells = max(1, int(getattr(topology, "num_cells", 1)))
        inter_site_distance_m = max(
            float(getattr(topology, "inter_site_distance_m", 0.0)),
            float(getattr(topology, "cell_radius_m", 250.0)) * 1.8,
        )
        cell_height_m = float(getattr(topology, "cell_height_m", 25.0))
        explicit_positions = dict(template.metadata.get("cell_positions", {}) or {})
        cell_layout: dict[str, dict[str, float]] = {}
        if len(explicit_positions) > 0:
            for raw_cell_id, raw_position in explicit_positions.items():
                position = dict(raw_position or {})
                cell_layout[str(raw_cell_id)] = {
                    "x_m": float(position.get("x_m", 0.0)),
                    "y_m": float(position.get("y_m", 0.0)),
                    "z_m": float(position.get("z_m", cell_height_m)),
                }
        if len(cell_layout) == 0:
            cell_layout = {
                str(cell_config.get("cell_id", "cell_0")): {
                    "x_m": 0.0,
                    "y_m": 0.0,
                    "z_m": cell_height_m,
                }
            }
        if num_cells <= 1 and len(cell_layout) > 0:
            return cell_layout

        for cell_index in range(1, num_cells):
            candidate_cell_id = f"cell_{cell_index}"
            if candidate_cell_id in cell_layout:
                continue
            angle_rad = 2.0 * math.pi * float(cell_index - 1) / float(max(num_cells - 1, 1))
            cell_layout[candidate_cell_id] = {
                "x_m": inter_site_distance_m * math.cos(angle_rad),
                "y_m": inter_site_distance_m * math.sin(angle_rad),
                "z_m": cell_height_m,
            }
        return cell_layout

    def _build_cell_radio_profiles(
        self,
        *,
        template: ScenarioTemplate,
        cell_config: dict[str, Any],
        cell_layout: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, Any]]:
        raw_profiles = dict(template.metadata.get("cell_profiles", {}) or {})
        primary_cell_id = str(cell_config.get("cell_id", "cell_0"))
        default_profile = {
            "cell_type": "macro",
            "carrier_frequency_hz": float(template.radio.carrier_frequency_hz),
            "bandwidth_hz": float(template.radio.bandwidth_hz),
            "subcarrier_spacing_hz": float(template.radio.subcarrier_spacing_hz),
            "cyclic_prefix_type": str(template.radio.cyclic_prefix_type),
            "tx_power_dbm": float(cell_config.get("tx_power_dbm", 30.0)),
            "gnb_num_tx_ant": int(template.radio.gnb_num_tx_ant),
            "gnb_panel_layout": str(template.radio.gnb_panel_layout),
            "polarization": str(template.radio.polarization),
            "scenario_type": str(template.channel.scenario_type),
        }
        profiles: dict[str, dict[str, Any]] = {}
        for cell_id, position in sorted(cell_layout.items(), key=lambda item: str(item[0])):
            profile = dict(default_profile)
            if str(cell_id) != primary_cell_id:
                profile["cell_type"] = "small_cell"
            profile.update(dict(raw_profiles.get(str(cell_id), {}) or {}))
            profile = resolve_cell_rf_profile(profile)
            profile["x_m"] = float(position.get("x_m", 0.0))
            profile["y_m"] = float(position.get("y_m", 0.0))
            profile["z_m"] = float(position.get("z_m", template.topology.cell_height_m))
            profiles[str(cell_id)] = profile
        return profiles

    def _compute_geometry(
        self,
        *,
        ue_id: str,
        ue_item: dict[str, Any],
        ue_index: int,
        ue_count: int,
        template: ScenarioTemplate,
        rng: random.Random,
    ) -> UeGeometry:
        topology = template.topology
        radius_m = float(topology.cell_radius_m)
        layout = str(topology.geometry_layout).strip().lower()
        if "x_m" in ue_item and "y_m" in ue_item:
            x_m = float(ue_item.get("x_m", 0.0))
            y_m = float(ue_item.get("y_m", 0.0))
        else:
            normalized_index = ue_index + 1
            if layout == "arc":
                angle_deg = 20.0 + 140.0 * float(normalized_index) / float(max(ue_count, 1) + 1)
                radial_fraction = 0.35 + 0.45 * float(normalized_index) / float(max(ue_count, 1) + 1)
            else:
                angle_deg = 360.0 * float(ue_index) / float(max(1, ue_count))
                radial_fraction = 0.2 + 0.6 * float(normalized_index) / float(max(ue_count, 1) + 1)
            if "radius_fraction" in ue_item:
                radial_fraction = float(ue_item.get("radius_fraction", radial_fraction))
            radius_value_m = max(1.0, float(ue_item.get("distance_2d_m", radial_fraction * radius_m)))
            theta_rad = math.radians(float(ue_item.get("azimuth_deg", angle_deg)))
            x_m = radius_value_m * math.cos(theta_rad)
            y_m = radius_value_m * math.sin(theta_rad)

        z_m = float(ue_item.get("z_m", topology.default_ue_height_m))
        distance_2d_m = math.sqrt(x_m * x_m + y_m * y_m)
        distance_3d_m = math.sqrt(
            distance_2d_m * distance_2d_m + (float(topology.cell_height_m) - z_m) ** 2
        )
        velocity_mps = float(ue_item.get("velocity_mps", topology.default_velocity_mps))
        heading_deg = float(ue_item.get("heading_deg", topology.default_heading_deg))
        indoor_ratio = min(1.0, max(0.0, float(topology.indoor_ratio)))
        is_indoor = bool(ue_item.get("is_indoor", rng.random() < indoor_ratio))
        return UeGeometry(
            ue_id=str(ue_id),
            x_m=float(x_m),
            y_m=float(y_m),
            z_m=float(z_m),
            distance_2d_m=float(distance_2d_m),
            distance_3d_m=float(distance_3d_m),
            velocity_mps=float(velocity_mps),
            heading_deg=float(heading_deg),
            is_indoor=is_indoor,
        )

    def _compute_los_probability(self, *, scenario_type: str, distance_2d_m: float) -> float:
        d2d = max(1.0, float(distance_2d_m))
        normalized = str(scenario_type).strip().lower()
        if normalized.startswith("umi"):
            return min(1.0, 18.0 / d2d) * (1.0 - math.exp(-d2d / 36.0)) + math.exp(-d2d / 36.0)
        if normalized.startswith("uma"):
            return min(1.0, 18.0 / d2d) * (1.0 - math.exp(-d2d / 63.0)) + math.exp(-d2d / 63.0)
        if normalized.startswith("inh"):
            return max(0.2, min(1.0, math.exp(-d2d / 25.0)))
        if normalized.startswith("rma"):
            return max(0.3, min(1.0, math.exp(-d2d / 200.0)))
        return max(0.2, min(1.0, math.exp(-d2d / 50.0)))

    def _resolve_los_state(
        self,
        *,
        ue_item: dict[str, Any],
        distance_2d_m: float,
        template: ScenarioTemplate,
        rng: random.Random,
    ) -> str:
        explicit = ue_item.get("los_state", None)
        if explicit is not None:
            return str(explicit).strip().lower()

        los_mode = str(template.channel.los_mode).strip().lower()
        if los_mode in {"los", "nlos"}:
            return los_mode

        los_probability = self._compute_los_probability(
            scenario_type=template.channel.scenario_type,
            distance_2d_m=distance_2d_m,
        )
        return "los" if rng.random() <= los_probability else "nlos"

    def _select_channel_model_type(self, *, template: ScenarioTemplate, los_state: str) -> str:
        normalized_type = str(template.channel.default_channel_model_type).strip().lower()
        if normalized_type != "":
            return normalized_type
        if str(template.channel.scenario_type).strip().lower().startswith("inh"):
            return "tdl_a" if los_state == "los" else "tdl_b"
        return "tdl_c" if los_state == "los" else "tdl_d"

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
        *,
        ue_item: dict[str, Any],
        ue_index: int,
        geometry: UeGeometry,
        template: ScenarioTemplate,
        cell_radio_profiles: dict[str, dict[str, Any]],
        cell_layout: dict[str, dict[str, float]],
        beamforming_config: dict[str, Any],
        rng_seed: int,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        sorted_cells = sorted(cell_layout.items(), key=lambda item: str(item[0]))
        for cell_rank, (cell_id, cell_position) in enumerate(sorted_cells):
            cell_profile = dict(cell_radio_profiles.get(str(cell_id), {}) or {})
            carrier_frequency_hz = float(
                cell_profile.get("carrier_frequency_hz", template.radio.carrier_frequency_hz)
            )
            tx_power_dbm = float(cell_profile.get("tx_power_dbm", 30.0))
            scenario_type = str(cell_profile.get("scenario_type", template.channel.scenario_type))
            dx_m = float(geometry.x_m) - float(cell_position.get("x_m", 0.0))
            dy_m = float(geometry.y_m) - float(cell_position.get("y_m", 0.0))
            dz_m = float(float(cell_position.get("z_m", template.topology.cell_height_m)) - float(geometry.z_m))
            distance_2d_m = math.sqrt(dx_m * dx_m + dy_m * dy_m)
            distance_3d_m = math.sqrt(distance_2d_m * distance_2d_m + dz_m * dz_m)
            candidate_rng = random.Random(rng_seed + 97 * ue_index + 7919 * cell_rank)
            los_state = self._resolve_los_state(
                ue_item=ue_item,
                distance_2d_m=distance_2d_m,
                template=template,
                rng=candidate_rng,
            )
            pathloss_db = self._compute_pathloss_db(
                scenario_type=scenario_type,
                los_state=los_state,
                distance_3d_m=distance_3d_m,
                ue_height_m=geometry.z_m,
                carrier_frequency_hz=carrier_frequency_hz,
            )
            shadow_sigma_db = (
                float(template.channel.shadowing_sigma_db_los)
                if los_state == "los"
                else float(template.channel.shadowing_sigma_db_nlos)
            )
            shadowing_db = float(candidate_rng.gauss(0.0, shadow_sigma_db))
            o2i_penetration_loss_db = float(
                template.channel.o2i_penetration_loss_db if geometry.is_indoor else 0.0
            )
            rsrp_dbm = float(tx_power_dbm - pathloss_db - shadowing_db - o2i_penetration_loss_db)
            candidate_beam_id = self._compute_serving_beam_id(
                ue_item=ue_item,
                geometry=geometry,
                beamforming_config=beamforming_config,
                cell_position=cell_position,
            )
            candidates.append(
                {
                    "cell_id": str(cell_id),
                    "cell_type": str(cell_profile.get("cell_type", "macro")),
                    "x_m": float(cell_position.get("x_m", 0.0)),
                    "y_m": float(cell_position.get("y_m", 0.0)),
                    "z_m": float(cell_position.get("z_m", template.topology.cell_height_m)),
                    "distance_2d_m": float(distance_2d_m),
                    "distance_3d_m": float(distance_3d_m),
                    "carrier_frequency_hz": float(carrier_frequency_hz),
                    "bandwidth_hz": float(cell_profile.get("bandwidth_hz", template.radio.bandwidth_hz)),
                    "subcarrier_spacing_hz": float(
                        cell_profile.get("subcarrier_spacing_hz", template.radio.subcarrier_spacing_hz)
                    ),
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
                    "los_state": str(los_state),
                    "pathloss_db": float(pathloss_db),
                    "shadowing_db": float(shadowing_db),
                    "o2i_penetration_loss_db": float(o2i_penetration_loss_db),
                    "rsrp_dbm": float(rsrp_dbm),
                    "quality_score_db": float(rsrp_dbm),
                    "candidate_beam_id": int(candidate_beam_id),
                }
            )
        candidates.sort(key=lambda item: float(item.get("quality_score_db", -1e9)), reverse=True)
        candidate_limit = max(1, int(getattr(template.topology, "candidate_cell_limit", 3)))
        return candidates[:candidate_limit]

    def _compute_serving_beam_id(
        self,
        *,
        ue_item: dict[str, Any],
        geometry: UeGeometry,
        beamforming_config: dict[str, Any],
        cell_position: dict[str, float] | None = None,
    ) -> int:
        if "serving_beam_id" in ue_item and ue_item["serving_beam_id"] is not None:
            return int(ue_item["serving_beam_id"])

        raw_codebooks = dict(beamforming_config.get("codebooks", {}) or {})
        if len(raw_codebooks) == 0:
            return 0
        first_codebook = raw_codebooks.get(0, next(iter(raw_codebooks.values())))
        beam_ids = sorted(int(item) for item in dict(first_codebook.get("beams", {}) or {}).keys())
        if len(beam_ids) == 0:
            return 0

        cell_x_m = float((cell_position or {}).get("x_m", 0.0))
        cell_y_m = float((cell_position or {}).get("y_m", 0.0))
        azimuth_deg = (
            math.degrees(math.atan2(geometry.y_m - cell_y_m, geometry.x_m - cell_x_m)) + 360.0
        ) % 360.0
        candidate_angles = {
            beam_id: (360.0 * index / float(len(beam_ids)))
            for index, beam_id in enumerate(beam_ids)
        }
        return min(
            beam_ids,
            key=lambda beam_id: min(
                abs(candidate_angles[beam_id] - azimuth_deg),
                360.0 - abs(candidate_angles[beam_id] - azimuth_deg),
            ),
        )

    def _apply_override_paths(self, bundle_dicts: dict[str, dict[str, Any]], overrides: dict[str, Any]) -> None:
        for dotted_path, value in dict(overrides or {}).items():
            if "." not in str(dotted_path):
                continue
            root_name, *path_parts = str(dotted_path).split(".")
            root_dict = bundle_dicts.get(root_name)
            if root_dict is None:
                continue
            current = root_dict
            for path_part in path_parts[:-1]:
                current = current.setdefault(path_part, {})
            current[path_parts[-1]] = deepcopy(value)

    def generate(
        self,
        *,
        template: ScenarioTemplate,
        scenario_config: dict[str, Any] | None,
        sim_config: dict[str, Any],
        cell_config: dict[str, Any],
        ue_config: dict[str, Any],
        beamforming_config: dict[str, Any],
    ) -> GeneratedScenarioBundle:
        config = dict(scenario_config or {})
        generated_sim_config = deepcopy(sim_config)
        generated_cell_config = deepcopy(cell_config)
        generated_ue_config = deepcopy(ue_config)
        generated_beamforming_config = deepcopy(beamforming_config)
        cell_layout = self._build_cell_layout(template=template, cell_config=generated_cell_config)
        cell_radio_profiles = self._build_cell_radio_profiles(
            template=template,
            cell_config=generated_cell_config,
            cell_layout=cell_layout,
        )

        antenna_config = generated_beamforming_config.setdefault("antenna", {})
        antenna_config["carrier_frequency_hz"] = float(template.radio.carrier_frequency_hz)
        antenna_config["bandwidth_hz"] = float(template.radio.bandwidth_hz)
        antenna_config["subcarrier_spacing_hz"] = float(template.radio.subcarrier_spacing_hz)
        antenna_config["cyclic_prefix_type"] = str(template.radio.cyclic_prefix_type)
        antenna_config["gnb_num_tx_ant"] = int(template.radio.gnb_num_tx_ant)
        antenna_config["default_ue_num_rx_ant"] = int(template.radio.default_ue_num_rx_ant)
        antenna_config["gnb_panel_layout"] = str(template.radio.gnb_panel_layout)
        antenna_config["default_ue_panel_layout"] = str(template.radio.default_ue_panel_layout)
        antenna_config["polarization"] = str(template.radio.polarization)
        generated_cell_config["managed_cell_ids"] = sorted(cell_layout.keys())
        generated_cell_config["neighbor_cell_positions"] = deepcopy(cell_layout)
        generated_cell_config["cell_radio_profiles"] = deepcopy(cell_radio_profiles)
        generated_cell_config["handover_margin_db"] = float(template.topology.handover_margin_db)
        generated_cell_config["handover_delay_slots"] = int(template.topology.handover_delay_slots)

        generated_sim_config["obs_delay_slots"] = int(template.control.obs_delay_slots)
        generated_sim_config["config_actuation_delay_slots"] = int(template.control.config_actuation_delay_slots)
        generated_sim_config["command_actuation_delay_slots"] = int(template.control.command_actuation_delay_slots)
        generated_sim_config["measurement_queue_delay_slots"] = int(template.control.measurement_queue_delay_slots)
        generated_sim_config["measurement_sinr_window_slots"] = int(template.control.measurement_sinr_window_slots)
        generated_sim_config["measurement_sinr_noise_std_db"] = float(template.control.measurement_sinr_noise_std_db)
        generated_sim_config["measurement_bler_window_tbs"] = int(template.control.measurement_bler_window_tbs)
        generated_sim_config["measurement_throughput_granularity_slots"] = int(
            template.control.measurement_throughput_granularity_slots
        )

        ue_items = list(generated_ue_config.get("ues", []) or [])
        rng_seed = int(generated_sim_config.get("rng_seed", 1))
        ue_geometries: dict[str, UeGeometry] = {}

        for index, ue_item in enumerate(ue_items):
            ue_id = str(ue_item.get("ue_id", f"ue_{index}"))
            rng = random.Random(rng_seed + 97 * index)
            geometry = self._compute_geometry(
                ue_id=ue_id,
                ue_item=ue_item,
                ue_index=index,
                ue_count=len(ue_items),
                template=template,
                rng=rng,
            )
            candidate_cells = self._compute_candidate_cells(
                ue_item=ue_item,
                ue_index=index,
                geometry=geometry,
                template=template,
                cell_radio_profiles=cell_radio_profiles,
                cell_layout=cell_layout,
                beamforming_config=generated_beamforming_config,
                rng_seed=rng_seed,
            )
            explicit_serving_cell_id = ue_item.get("serving_cell_id", None)
            serving_candidate = None
            if explicit_serving_cell_id is not None:
                for candidate_cell in candidate_cells:
                    if str(candidate_cell.get("cell_id", "")) == str(explicit_serving_cell_id):
                        serving_candidate = candidate_cell
                        break
            if serving_candidate is None and len(candidate_cells) > 0:
                serving_candidate = candidate_cells[0]
            if serving_candidate is None:
                serving_candidate = {
                    "cell_id": str(generated_cell_config.get("cell_id", "cell_0")),
                    "distance_2d_m": float(geometry.distance_2d_m),
                    "distance_3d_m": float(geometry.distance_3d_m),
                    "los_state": "los",
                    "pathloss_db": 130.0,
                    "shadowing_db": 0.0,
                    "o2i_penetration_loss_db": 0.0,
                    "candidate_beam_id": 0,
                    "quality_score_db": -100.0,
                    "rsrp_dbm": -100.0,
                }
            los_state = str(serving_candidate.get("los_state", "los"))
            serving_shadow_sigma_db = (
                float(template.channel.shadowing_sigma_db_los)
                if los_state == "los"
                else float(template.channel.shadowing_sigma_db_nlos)
            )
            pathloss_db = float(serving_candidate.get("pathloss_db", 130.0))
            shadowing_db = float(ue_item.get("shadowing_db", serving_candidate.get("shadowing_db", 0.0)))
            delay_spread_ns = float(
                ue_item.get(
                    "delay_spread_ns",
                    template.channel.delay_spread_ns_los if los_state == "los" else template.channel.delay_spread_ns_nlos,
                )
            )
            k_factor_db = float(
                ue_item.get(
                    "k_factor_db",
                    template.channel.k_factor_db_los if los_state == "los" else template.channel.k_factor_db_nlos,
                )
            )
            serving_carrier_frequency_hz = float(
                serving_candidate.get("carrier_frequency_hz", template.radio.carrier_frequency_hz)
            )
            angle_spread_deg = float(
                ue_item.get(
                    "angle_spread_deg",
                    template.channel.angle_spread_deg_los if los_state == "los" else template.channel.angle_spread_deg_nlos,
                )
            )
            spatial_correlation_level = float(
                ue_item.get(
                    "spatial_correlation_level",
                    template.channel.spatial_correlation_level_los if los_state == "los" else template.channel.spatial_correlation_level_nlos,
                )
            )
            velocity_mps = float(ue_item.get("velocity_mps", geometry.velocity_mps))
            doppler_hz = float(
                ue_item.get(
                    "doppler_hz",
                    abs(velocity_mps) * serving_carrier_frequency_hz / 3.0e8,
                )
            )
            o2i_penetration_loss_db = float(
                ue_item.get(
                    "o2i_penetration_loss_db",
                    serving_candidate.get("o2i_penetration_loss_db", template.channel.o2i_penetration_loss_db if geometry.is_indoor else 0.0),
                )
            )
            serving_beam_id = int(
                ue_item.get("serving_beam_id", serving_candidate.get("candidate_beam_id", 0))
            )

            ue_item["ue_id"] = ue_id
            ue_item["x_m"] = float(geometry.x_m)
            ue_item["y_m"] = float(geometry.y_m)
            ue_item["z_m"] = float(geometry.z_m)
            ue_item["distance_2d_m"] = float(geometry.distance_2d_m)
            ue_item["distance_3d_m"] = float(geometry.distance_3d_m)
            ue_item["velocity_mps"] = float(geometry.velocity_mps)
            ue_item["heading_deg"] = float(geometry.heading_deg)
            ue_item["is_indoor"] = bool(geometry.is_indoor)
            ue_item["scenario_type"] = str(template.channel.scenario_type)
            ue_item["serving_cell_id"] = str(serving_candidate.get("cell_id", generated_cell_config.get("cell_id", "cell_0")))
            ue_item["serving_cell_type"] = str(serving_candidate.get("cell_type", "macro"))
            ue_item["serving_carrier_frequency_hz"] = float(serving_carrier_frequency_hz)
            ue_item["los_state"] = str(los_state)
            ue_item["pathloss_db"] = float(ue_item.get("pathloss_db", pathloss_db))
            ue_item["shadowing_db"] = float(shadowing_db)
            ue_item["shadowing_sigma_db"] = float(
                ue_item.get("shadowing_sigma_db", serving_shadow_sigma_db)
            )
            ue_item["doppler_hz"] = float(doppler_hz)
            ue_item["delay_spread_ns"] = float(delay_spread_ns)
            ue_item["channel_model_type"] = str(
                ue_item.get(
                    "channel_model_type",
                    self._select_channel_model_type(template=template, los_state=los_state),
                )
            )
            ue_item["k_factor_db"] = float(k_factor_db)
            ue_item["angle_spread_deg"] = float(angle_spread_deg)
            ue_item["spatial_correlation_level"] = float(spatial_correlation_level)
            ue_item["o2i_penetration_loss_db"] = float(o2i_penetration_loss_db)
            ue_item["fast_fading_seed"] = int(
                ue_item.get("fast_fading_seed", int(template.channel.fast_fading_seed_offset) + index)
            )
            ue_item["spatial_corr_seed"] = int(
                ue_item.get("spatial_corr_seed", int(template.channel.spatial_corr_seed_offset) + index)
            )
            ue_item["candidate_cells"] = deepcopy(candidate_cells)
            ue_item["large_scale"] = {
                "distance_2d_m": float(serving_candidate.get("distance_2d_m", geometry.distance_2d_m)),
                "distance_3d_m": float(serving_candidate.get("distance_3d_m", geometry.distance_3d_m)),
                "pathloss_db": float(ue_item["pathloss_db"]),
                "shadowing_db": float(ue_item["shadowing_db"]),
                "shadowing_sigma_db": float(ue_item["shadowing_sigma_db"]),
                "los_state": str(ue_item["los_state"]),
                "o2i_penetration_loss_db": float(ue_item["o2i_penetration_loss_db"]),
            }
            ue_item["small_scale"] = {
                "doppler_hz": float(ue_item["doppler_hz"]),
                "delay_spread_ns": float(ue_item["delay_spread_ns"]),
                "k_factor_db": float(ue_item["k_factor_db"]),
                "angle_spread_deg": float(ue_item["angle_spread_deg"]),
                "spatial_correlation_level": float(ue_item["spatial_correlation_level"]),
                "channel_model_type": str(ue_item["channel_model_type"]),
                "fast_fading_seed": int(ue_item["fast_fading_seed"]),
                "spatial_corr_seed": int(ue_item["spatial_corr_seed"]),
            }
            ue_item["serving_beam_id"] = int(serving_beam_id)
            ue_item["num_rx_ant"] = int(
                ue_item.get("num_rx_ant", antenna_config.get("default_ue_num_rx_ant", template.radio.default_ue_num_rx_ant))
            )
            ue_item["ue_panel_layout"] = str(
                ue_item.get(
                    "ue_panel_layout",
                    antenna_config.get("default_ue_panel_layout", template.radio.default_ue_panel_layout),
                )
            )
            if "arrival_rate_bps" in ue_item and ue_item["arrival_rate_bps"] is not None:
                ue_item["arrival_rate_bps"] = float(ue_item["arrival_rate_bps"]) * float(
                    template.traffic.arrival_rate_scale
                )
            if "burst_size_bytes" in ue_item and ue_item["burst_size_bytes"] is not None:
                ue_item["burst_size_bytes"] = int(
                    max(
                        1,
                        round(
                            float(ue_item["burst_size_bytes"])
                            * float(template.traffic.burstiness_scale)
                        ),
                    )
                )
            if "packet_size_bytes" in ue_item and ue_item["packet_size_bytes"] is not None:
                ue_item["packet_size_bytes"] = int(
                    max(
                        1,
                        round(
                            float(ue_item["packet_size_bytes"])
                            * float(template.traffic.burstiness_scale)
                        ),
                    )
                )
            ue_geometries[ue_id] = geometry

        generated_ue_config["ues"] = ue_items

        scenario_overrides = dict(config.get("overrides", {}) or {})
        self._apply_override_paths(
            {
                "sim": generated_sim_config,
                "cell": generated_cell_config,
                "ue": generated_ue_config,
                "beamforming": generated_beamforming_config,
            },
            scenario_overrides,
        )

        return GeneratedScenarioBundle(
            scenario_name=str(config.get("scenario_name", template.name)),
            scenario_template=template,
            sim_config=generated_sim_config,
            cell_config=generated_cell_config,
            ue_config=generated_ue_config,
            beamforming_config=generated_beamforming_config,
            ue_geometries=ue_geometries,
            scenario_metadata={
                "template": template.name,
                "scenario_type": template.channel.scenario_type,
                "topology": asdict(template.topology),
                "radio": asdict(template.radio),
                "channel": asdict(template.channel),
                "cell_layout": deepcopy(cell_layout),
                "cell_radio_profiles": deepcopy(cell_radio_profiles),
            },
        )
