"""
Built-in scenario matrices and scene families.
"""

from __future__ import annotations

import math
from typing import Any

from models.scenario_matrix import (
    ScenarioMatrix,
    ScenarioMatrixCase,
    ScenarioSweepDimension,
    ScenarioSweepValue,
)


def _build_embb_ue(
    *,
    ue_id: str,
    radius_fraction: float,
    velocity_mps: float,
    heading_deg: float,
    arrival_rate_bps: float,
    indoor: bool,
) -> dict[str, Any]:
    return {
        "ue_id": ue_id,
        "queue_bytes": 1200,
        "throughput_mbps": 0.0,
        "sinr_db": 11.0,
        "radio_quality_db": 10.5,
        "bler": 0.1,
        "slice_id": "embb",
        "traffic_type": "video",
        "latency_budget_ms": 100.0,
        "priority": 1,
        "arrival_model": "on_off",
        "arrival_rate_bps": arrival_rate_bps,
        "burst_size_bytes": 2200,
        "on_duration_slots": 40,
        "off_duration_slots": 16,
        "start_offset_slots": 0,
        "max_queue_bytes": 240000,
        "radius_fraction": radius_fraction,
        "velocity_mps": velocity_mps,
        "heading_deg": heading_deg,
        "is_indoor": indoor,
    }


def _build_urllc_ue(
    *,
    ue_id: str,
    radius_fraction: float,
    velocity_mps: float,
    heading_deg: float,
    arrival_rate_bps: float,
    indoor: bool,
) -> dict[str, Any]:
    return {
        "ue_id": ue_id,
        "queue_bytes": 600,
        "throughput_mbps": 0.0,
        "sinr_db": 12.0,
        "radio_quality_db": 12.0,
        "bler": 0.08,
        "slice_id": "urllc",
        "traffic_type": "control",
        "latency_budget_ms": 10.0,
        "priority": 3,
        "arrival_model": "cbr",
        "arrival_rate_bps": arrival_rate_bps,
        "max_queue_bytes": 100000,
        "radius_fraction": radius_fraction,
        "velocity_mps": velocity_mps,
        "heading_deg": heading_deg,
        "is_indoor": indoor,
    }


def _build_mmtc_ue(
    *,
    ue_id: str,
    radius_fraction: float,
    velocity_mps: float,
    heading_deg: float,
    arrival_rate_bps: float,
    indoor: bool,
) -> dict[str, Any]:
    return {
        "ue_id": ue_id,
        "queue_bytes": 300,
        "throughput_mbps": 0.0,
        "sinr_db": 9.0,
        "radio_quality_db": 8.5,
        "bler": 0.12,
        "slice_id": "mmtc",
        "traffic_type": "iot",
        "latency_budget_ms": 500.0,
        "priority": 2,
        "arrival_model": "poisson",
        "arrival_rate_bps": arrival_rate_bps,
        "packet_size_bytes": 200,
        "max_queue_bytes": 50000,
        "radius_fraction": radius_fraction,
        "velocity_mps": velocity_mps,
        "heading_deg": heading_deg,
        "is_indoor": indoor,
        "num_rx_ant": 1,
    }


def _build_phone_ue(
    *,
    ue_id: str,
    x_m: float,
    y_m: float,
    velocity_mps: float,
    heading_deg: float,
    arrival_rate_bps: float,
    indoor: bool,
    priority: int = 2,
    latency_budget_ms: float = 50.0,
) -> dict[str, Any]:
    return {
        "ue_id": ue_id,
        "queue_bytes": 380,
        "throughput_mbps": 0.0,
        "sinr_db": 10.5,
        "radio_quality_db": 10.0,
        "bler": 0.08,
        "slice_id": "voice",
        "traffic_type": "phone",
        "latency_budget_ms": latency_budget_ms,
        "priority": priority,
        "arrival_model": "cbr",
        "arrival_rate_bps": arrival_rate_bps,
        "max_queue_bytes": 80000,
        "x_m": x_m,
        "y_m": y_m,
        "velocity_mps": velocity_mps,
        "heading_deg": heading_deg,
        "is_indoor": indoor,
    }


def _hex_cell_positions(
    *,
    num_cells: int,
    inter_site_distance_m: float,
    macro_height_m: float = 25.0,
    small_cell_height_m: float = 12.0,
) -> dict[str, dict[str, float]]:
    """Build a deterministic center-plus-hex-rings cell layout."""

    normalized_count = max(1, int(num_cells))
    positions: list[tuple[int, int]] = [(0, 0)]
    ring = 1
    while len(positions) < normalized_count:
        ring_positions = [
            (q, r)
            for q in range(-ring, ring + 1)
            for r in range(-ring, ring + 1)
            if max(abs(q), abs(r), abs(q + r)) == ring
        ]
        ring_positions.sort(
            key=lambda item: (
                math.atan2(
                    (math.sqrt(3.0) / 2.0) * item[1],
                    item[0] + item[1] / 2.0,
                )
                % (2.0 * math.pi)
            )
        )
        positions.extend(ring_positions)
        ring += 1

    layout: dict[str, dict[str, float]] = {}
    for index, (q, r) in enumerate(positions[:normalized_count]):
        x_m = float(inter_site_distance_m) * (float(q) + float(r) / 2.0)
        y_m = float(inter_site_distance_m) * (math.sqrt(3.0) / 2.0) * float(r)
        layout[f"cell_{index}"] = {
            "x_m": float(x_m),
            "y_m": float(y_m),
            "z_m": float(macro_height_m if index == 0 else small_cell_height_m),
        }
    return layout


def _paper_eval_cell_profiles(
    *,
    cell_positions: dict[str, dict[str, float]],
    stress_profile: str = "nominal",
) -> dict[str, dict[str, Any]]:
    """Build FR1/FR2 heterogeneous cell profiles for paper-scale O-RAN scenes."""

    normalized_profile = str(stress_profile).strip().lower()
    interference_profile = normalized_profile in {"interference_burst", "large_interference"}
    profiles: dict[str, dict[str, Any]] = {}
    for cell_id in sorted(cell_positions.keys(), key=lambda item: int(str(item).split("_")[-1])):
        cell_index = int(str(cell_id).split("_")[-1])
        if cell_index == 0:
            profiles[cell_id] = {
                "cell_type": "macro",
                "carrier_frequency_hz": 3.5e9,
                "bandwidth_hz": 40e6,
                "subcarrier_spacing_hz": 30e3,
                "tx_power_dbm": 37.0 if not interference_profile else 38.5,
                "gnb_num_tx_ant": 8,
                "gnb_panel_layout": "4x2",
                "scenario_type": "uma",
                "adjacent_channel_leakage_ratio_db": 45.0 if not interference_profile else 38.0,
                "adjacent_channel_selectivity_db": 33.0,
                "out_of_band_isolation_db": 42.0,
            }
            continue

        use_mmwave = cell_index % 3 == 1
        if use_mmwave:
            profiles[cell_id] = {
                "cell_type": "mmwave_small_cell",
                "carrier_frequency_hz": 28.0e9 if cell_index % 2 else 26.0e9,
                "bandwidth_hz": 100e6,
                "subcarrier_spacing_hz": 120e3,
                "tx_power_dbm": 28.0 if not normalized_profile.startswith("blockage") else 25.0,
                "gnb_num_tx_ant": 16,
                "gnb_panel_layout": "4x4",
                "scenario_type": "umi_street_canyon",
                "adjacent_channel_leakage_ratio_db": 42.0,
                "adjacent_channel_selectivity_db": 30.0,
                "out_of_band_isolation_db": 45.0,
            }
        else:
            profiles[cell_id] = {
                "cell_type": "sub6_small_cell",
                "carrier_frequency_hz": 3.5e9 if interference_profile else 3.55e9 + 10e6 * (cell_index % 2),
                "bandwidth_hz": 40e6,
                "subcarrier_spacing_hz": 30e3,
                "tx_power_dbm": 32.0 if not interference_profile else 35.0,
                "gnb_num_tx_ant": 8,
                "gnb_panel_layout": "4x2",
                "scenario_type": "umi_street_canyon",
                "adjacent_channel_leakage_ratio_db": 44.0 if not interference_profile else 36.0,
                "adjacent_channel_selectivity_db": 32.0 if not interference_profile else 28.0,
                "out_of_band_isolation_db": 42.0,
            }
    return profiles


def _paper_eval_ue_type(index: int, *, profile: str) -> str:
    normalized = str(profile).strip().lower()
    if normalized == "urllc_burst":
        pattern = ["urllc", "embb", "urllc", "embb", "mmtc", "urllc"]
    elif normalized == "mmtc_heavy":
        pattern = ["mmtc", "mmtc", "embb", "mmtc", "urllc"]
    elif normalized in {"dense_hotspot_high_mobility", "hotspot_shift", "high_mobility_interference"}:
        pattern = ["embb", "urllc", "embb", "embb", "phone", "mmtc"]
    else:
        pattern = ["embb", "urllc", "mmtc", "embb", "phone", "embb"]
    return pattern[int(index) % len(pattern)]


def _paper_eval_anchor_cell_ids(
    *,
    cell_positions: dict[str, dict[str, float]],
    profile: str,
) -> list[str]:
    cell_ids = sorted(cell_positions.keys(), key=lambda item: int(str(item).split("_")[-1]))
    normalized = str(profile).strip().lower()
    if len(cell_ids) <= 1:
        return cell_ids
    if normalized in {"dense_hotspot_high_mobility", "urllc_burst"}:
        hotspot_cell = "cell_1" if "cell_1" in cell_positions else cell_ids[1]
        return [hotspot_cell, hotspot_cell, hotspot_cell, cell_ids[0], cell_ids[-1]]
    if normalized == "hotspot_shift":
        first_hotspot = "cell_1" if "cell_1" in cell_positions else cell_ids[1]
        second_hotspot = "cell_4" if "cell_4" in cell_positions else cell_ids[-1]
        return [first_hotspot, first_hotspot, second_hotspot, second_hotspot, cell_ids[0]]
    if normalized == "high_mobility_interference":
        hotspot_cell = "cell_1" if "cell_1" in cell_positions else cell_ids[1]
        return [cell_ids[-1], hotspot_cell, cell_ids[-2], hotspot_cell, cell_ids[0]]
    if normalized in {"cell_edge_weak_sinr", "interference_burst"}:
        return [cell_ids[-1], cell_ids[-2], cell_ids[0], cell_ids[-1]]
    return cell_ids


def _paper_eval_velocity_mps(index: int, *, profile: str) -> float:
    normalized = str(profile).strip().lower()
    if normalized == "low_mobility":
        return [0.2, 0.8, 0.0, 0.5][index % 4]
    if normalized in {"dense_hotspot_high_mobility", "high_mobility", "high_mobility_interference"}:
        return [12.0, 16.5, 20.0, 22.0][index % 4]
    if normalized == "hotspot_shift":
        return [4.0, 8.0, 12.0, 16.0][index % 4]
    return [1.0, 3.0, 6.0, 8.0][index % 4]


def _build_paper_eval_ues(
    *,
    num_ues: int,
    profile: str,
    cell_positions: dict[str, dict[str, float]],
    load_scale: float = 1.0,
) -> list[dict[str, Any]]:
    """Build deterministic mixed-slice UE populations for O-RAN paper experiments."""

    normalized_profile = str(profile).strip().lower()
    anchors = _paper_eval_anchor_cell_ids(cell_positions=cell_positions, profile=normalized_profile)
    ues: list[dict[str, Any]] = []
    for index in range(max(1, int(num_ues))):
        ue_type = _paper_eval_ue_type(index, profile=normalized_profile)
        anchor_cell_id = anchors[index % len(anchors)]
        anchor = cell_positions[anchor_cell_id]
        local_angle_deg = (137.0 * float(index) + 23.0) % 360.0
        if normalized_profile in {"dense_hotspot_high_mobility", "urllc_burst"} and index % 5 < 3:
            radius_m = 35.0 + 8.0 * float(index % 7)
        elif normalized_profile in {"cell_edge_weak_sinr", "interference_burst", "high_mobility_interference"}:
            radius_m = 260.0 + 22.0 * float(index % 6)
        elif normalized_profile == "hotspot_shift" and index % 4 < 2:
            radius_m = 45.0 + 10.0 * float(index % 5)
        else:
            radius_m = 80.0 + 18.0 * float(index % 11)
        theta_rad = math.radians(local_angle_deg)
        x_m = float(anchor["x_m"]) + radius_m * math.cos(theta_rad)
        y_m = float(anchor["y_m"]) + radius_m * math.sin(theta_rad)
        velocity_mps = _paper_eval_velocity_mps(index, profile=normalized_profile)
        heading_deg = (local_angle_deg + 90.0 + 13.0 * float(index % 5)) % 360.0
        indoor = normalized_profile in {"temporary_blockage", "hotspot_shift"} and index % 3 == 0
        indoor = indoor or (index % 9 == 0)

        if ue_type == "urllc":
            ue = _build_urllc_ue(
                ue_id=f"ue_{index:03d}_urllc",
                radius_fraction=0.35,
                velocity_mps=velocity_mps,
                heading_deg=heading_deg,
                arrival_rate_bps=(1.2e6 if normalized_profile != "urllc_burst" else 2.6e6) * load_scale,
                indoor=indoor,
            )
            ue["latency_budget_ms"] = 8.0 if normalized_profile == "urllc_burst" else 10.0
        elif ue_type == "mmtc":
            ue = _build_mmtc_ue(
                ue_id=f"ue_{index:03d}_mmtc",
                radius_fraction=0.45,
                velocity_mps=velocity_mps,
                heading_deg=heading_deg,
                arrival_rate_bps=0.10e6 * load_scale,
                indoor=indoor,
            )
        elif ue_type == "phone":
            ue = _build_phone_ue(
                ue_id=f"ue_{index:03d}_phone",
                x_m=x_m,
                y_m=y_m,
                velocity_mps=velocity_mps,
                heading_deg=heading_deg,
                arrival_rate_bps=0.25e6 * load_scale,
                indoor=indoor,
                latency_budget_ms=40.0,
            )
        else:
            ue = _build_embb_ue(
                ue_id=f"ue_{index:03d}_embb",
                radius_fraction=0.40,
                velocity_mps=velocity_mps,
                heading_deg=heading_deg,
                arrival_rate_bps=(
                    13.0e6
                    if normalized_profile in {"dense_hotspot_high_mobility", "high_mobility_interference"}
                    else 7.0e6
                ) * load_scale,
                indoor=indoor,
            )
            ue["traffic_type"] = "video" if index % 2 == 0 else "broadband"

        ue = _attach_explicit_geometry(
            ue,
            x_m=x_m,
            y_m=y_m,
            velocity_mps=velocity_mps,
            heading_deg=heading_deg,
            indoor=indoor,
        )
        if normalized_profile in {"cell_edge_weak_sinr", "interference_burst", "temporary_blockage", "high_mobility_interference"} and index % 3 == 0:
            ue["los_state"] = "nlos"
            ue["shadowing_db"] = 5.0 + float(index % 4)
            ue["o2i_penetration_loss_db"] = 24.0 if indoor else 6.0
        if normalized_profile == "hotspot_shift" and index % 4 < 2:
            ue["arrival_model"] = "on_off"
            ue["on_duration_slots"] = 16
            ue["off_duration_slots"] = 8
            ue["burst_size_bytes"] = int(3200 + 200 * (index % 4))
        ues.append(ue)
    return ues


def _attach_explicit_geometry(
    ue: dict[str, Any],
    *,
    x_m: float,
    y_m: float,
    velocity_mps: float,
    heading_deg: float,
    indoor: bool,
) -> dict[str, Any]:
    updated = dict(ue)
    updated["x_m"] = float(x_m)
    updated["y_m"] = float(y_m)
    updated["velocity_mps"] = float(velocity_mps)
    updated["heading_deg"] = float(heading_deg)
    updated["is_indoor"] = bool(indoor)
    return updated


def _build_ue_population(
    *,
    num_ues: int,
    mix_profile: str,
    velocity_mps: float,
    indoor_ratio: float,
    embb_arrival_rate_bps: float,
    urllc_arrival_rate_bps: float,
    mmtc_arrival_rate_bps: float,
    radius_start: float = 0.2,
    radius_end: float = 0.85,
) -> list[dict[str, Any]]:
    mix_patterns = {
        "baseline": ["embb", "urllc", "mmtc"],
        "embb_heavy": ["embb", "embb", "urllc", "embb", "mmtc"],
        "mixed_congested": ["embb", "urllc", "embb", "mmtc", "urllc", "embb"],
        "mmtc_heavy": ["mmtc", "mmtc", "embb", "mmtc", "urllc"],
    }
    pattern = mix_patterns.get(str(mix_profile).strip().lower(), mix_patterns["baseline"])
    radius_step = 0.0 if num_ues <= 1 else (float(radius_end) - float(radius_start)) / float(num_ues - 1)

    ues: list[dict[str, Any]] = []
    for index in range(num_ues):
        ue_type = pattern[index % len(pattern)]
        radius_fraction = float(radius_start) + radius_step * float(index)
        heading_deg = (360.0 * float(index) / float(max(num_ues, 1))) % 360.0
        indoor = (float(index) / float(max(num_ues - 1, 1))) < float(indoor_ratio)
        ue_id = f"ue_{index}"
        if ue_type == "urllc":
            ue = _build_urllc_ue(
                ue_id=ue_id,
                radius_fraction=radius_fraction,
                velocity_mps=velocity_mps,
                heading_deg=heading_deg,
                arrival_rate_bps=urllc_arrival_rate_bps,
                indoor=indoor,
            )
        elif ue_type == "mmtc":
            ue = _build_mmtc_ue(
                ue_id=ue_id,
                radius_fraction=radius_fraction,
                velocity_mps=velocity_mps,
                heading_deg=heading_deg,
                arrival_rate_bps=mmtc_arrival_rate_bps,
                indoor=indoor,
            )
        else:
            ue = _build_embb_ue(
                ue_id=ue_id,
                radius_fraction=radius_fraction,
                velocity_mps=velocity_mps,
                heading_deg=heading_deg,
                arrival_rate_bps=embb_arrival_rate_bps,
                indoor=indoor,
            )
        ues.append(ue)
    return ues


def _build_macro_mmwave_benchmark_ues(mode: str = "nominal") -> list[dict[str, Any]]:
    normalized_mode = str(mode).strip().lower()
    hotspot_scale = 1.0
    edge_scale = 1.0
    mobility_scale = 1.0
    edge_nlos = False
    if normalized_mode == "hotspot_overload":
        hotspot_scale = 1.8
    elif normalized_mode == "edge_coverage":
        edge_scale = 1.45
        edge_nlos = True
    elif normalized_mode == "high_mobility":
        mobility_scale = 8.0

    def velocity(base_velocity: float) -> float:
        return float(base_velocity) * float(mobility_scale)

    return [
        _attach_explicit_geometry(
            _build_embb_ue(
                ue_id="ue_hotspot_video_0",
                radius_fraction=0.35,
                velocity_mps=velocity(0.8),
                heading_deg=35.0,
                arrival_rate_bps=14.0e6 * hotspot_scale,
                indoor=True,
            ),
            x_m=112.0,
            y_m=78.0,
            velocity_mps=velocity(0.8),
            heading_deg=35.0,
            indoor=True,
        ),
        _attach_explicit_geometry(
            _build_embb_ue(
                ue_id="ue_hotspot_video_1",
                radius_fraction=0.36,
                velocity_mps=velocity(0.6),
                heading_deg=120.0,
                arrival_rate_bps=12.0e6 * hotspot_scale,
                indoor=True,
            ),
            x_m=128.0,
            y_m=92.0,
            velocity_mps=velocity(0.6),
            heading_deg=120.0,
            indoor=True,
        ),
        _attach_explicit_geometry(
            {
                **_build_embb_ue(
                    ue_id="ue_hotspot_data_0",
                    radius_fraction=0.37,
                    velocity_mps=velocity(1.0),
                    heading_deg=75.0,
                    arrival_rate_bps=10.0e6 * hotspot_scale,
                    indoor=False,
                ),
                "traffic_type": "broadband",
                "slice_id": "embb",
            },
            x_m=138.0,
            y_m=98.0,
            velocity_mps=velocity(1.0),
            heading_deg=75.0,
            indoor=False,
        ),
        _attach_explicit_geometry(
            _build_embb_ue(
                ue_id="ue_hotspot_video_2",
                radius_fraction=0.38,
                velocity_mps=velocity(0.4),
                heading_deg=210.0,
                arrival_rate_bps=13.0e6 * hotspot_scale,
                indoor=True,
            ),
            x_m=118.0,
            y_m=104.0,
            velocity_mps=velocity(0.4),
            heading_deg=210.0,
            indoor=True,
        ),
        _build_phone_ue(
            ue_id="ue_hotspot_phone_0",
            x_m=132.0,
            y_m=72.0,
            velocity_mps=velocity(0.5),
            heading_deg=300.0,
            arrival_rate_bps=0.28e6,
            indoor=True,
        ),
        _attach_explicit_geometry(
            _build_urllc_ue(
                ue_id="ue_hotspot_urllc_0",
                radius_fraction=0.34,
                velocity_mps=velocity(0.7),
                heading_deg=150.0,
                arrival_rate_bps=1.4e6,
                indoor=False,
            ),
            x_m=144.0,
            y_m=84.0,
            velocity_mps=velocity(0.7),
            heading_deg=150.0,
            indoor=False,
        ),
        _attach_explicit_geometry(
            _build_mmtc_ue(
                ue_id="ue_hotspot_iot_0",
                radius_fraction=0.39,
                velocity_mps=velocity(0.1),
                heading_deg=5.0,
                arrival_rate_bps=0.1e6,
                indoor=True,
            ),
            x_m=108.0,
            y_m=96.0,
            velocity_mps=velocity(0.1),
            heading_deg=5.0,
            indoor=True,
        ),
        _attach_explicit_geometry(
            _build_embb_ue(
                ue_id="ue_hotspot_video_3",
                radius_fraction=0.40,
                velocity_mps=velocity(0.3),
                heading_deg=20.0,
                arrival_rate_bps=15.0e6 * hotspot_scale,
                indoor=False,
            ),
            x_m=150.0,
            y_m=106.0,
            velocity_mps=velocity(0.3),
            heading_deg=20.0,
            indoor=False,
        ),
        _attach_explicit_geometry(
            _build_embb_ue(
                ue_id="ue_macro_video_0",
                radius_fraction=0.18,
                velocity_mps=velocity(1.6),
                heading_deg=45.0,
                arrival_rate_bps=6.0e6,
                indoor=False,
            ),
            x_m=25.0,
            y_m=32.0,
            velocity_mps=velocity(1.6),
            heading_deg=45.0,
            indoor=False,
        ),
        _build_phone_ue(
            ue_id="ue_macro_phone_0",
            x_m=-36.0,
            y_m=44.0,
            velocity_mps=velocity(1.2),
            heading_deg=105.0,
            arrival_rate_bps=0.24e6,
            indoor=False,
        ),
        _attach_explicit_geometry(
            _build_urllc_ue(
                ue_id="ue_macro_urllc_0",
                radius_fraction=0.22,
                velocity_mps=velocity(1.8),
                heading_deg=265.0,
                arrival_rate_bps=1.1e6,
                indoor=False,
            ),
            x_m=-64.0,
            y_m=24.0,
            velocity_mps=velocity(1.8),
            heading_deg=265.0,
            indoor=False,
        ),
        _attach_explicit_geometry(
            _build_mmtc_ue(
                ue_id="ue_macro_iot_0",
                radius_fraction=0.24,
                velocity_mps=velocity(0.2),
                heading_deg=0.0,
                arrival_rate_bps=0.08e6,
                indoor=True,
            ),
            x_m=46.0,
            y_m=-38.0,
            velocity_mps=velocity(0.2),
            heading_deg=0.0,
            indoor=True,
        ),
        _attach_explicit_geometry(
            {
                **_build_embb_ue(
                    ue_id="ue_macro_data_0",
                    radius_fraction=0.30,
                    velocity_mps=velocity(1.4),
                    heading_deg=315.0,
                    arrival_rate_bps=5.0e6,
                    indoor=False,
                ),
                "traffic_type": "data",
                "slice_id": "embb",
            },
            x_m=-82.0,
            y_m=-18.0,
            velocity_mps=velocity(1.4),
            heading_deg=315.0,
            indoor=False,
        ),
        _attach_explicit_geometry(
            _build_embb_ue(
                ue_id="ue_edge_video_0",
                radius_fraction=0.72,
                velocity_mps=velocity(2.0),
                heading_deg=350.0,
                arrival_rate_bps=5.5e6 * edge_scale,
                indoor=edge_nlos,
            ),
            x_m=308.0,
            y_m=-58.0,
            velocity_mps=velocity(2.0),
            heading_deg=350.0,
            indoor=edge_nlos,
        ),
        _build_phone_ue(
            ue_id="ue_edge_phone_0",
            x_m=348.0,
            y_m=-12.0,
            velocity_mps=velocity(1.0),
            heading_deg=15.0,
            arrival_rate_bps=0.22e6,
            indoor=edge_nlos,
            latency_budget_ms=40.0,
        ),
        _attach_explicit_geometry(
            _build_urllc_ue(
                ue_id="ue_edge_urllc_0",
                radius_fraction=0.78,
                velocity_mps=velocity(2.5),
                heading_deg=200.0,
                arrival_rate_bps=1.0e6 * edge_scale,
                indoor=edge_nlos,
            ),
            x_m=332.0,
            y_m=8.0,
            velocity_mps=velocity(2.5),
            heading_deg=200.0,
            indoor=edge_nlos,
        ),
        _attach_explicit_geometry(
            _build_mmtc_ue(
                ue_id="ue_edge_iot_0",
                radius_fraction=0.82,
                velocity_mps=velocity(0.4),
                heading_deg=10.0,
                arrival_rate_bps=0.1e6 * edge_scale,
                indoor=True,
            ),
            x_m=366.0,
            y_m=-68.0,
            velocity_mps=velocity(0.4),
            heading_deg=10.0,
            indoor=True,
        ),
        _attach_explicit_geometry(
            {
                **_build_embb_ue(
                    ue_id="ue_edge_video_1",
                    radius_fraction=0.88,
                    velocity_mps=velocity(1.6),
                    heading_deg=160.0,
                    arrival_rate_bps=6.2e6 * edge_scale,
                    indoor=True,
                ),
                "los_state": "nlos" if edge_nlos else "auto",
            },
            x_m=388.0,
            y_m=-22.0,
            velocity_mps=velocity(1.6),
            heading_deg=160.0,
            indoor=True,
        ),
    ]


def _catalog_cases() -> list[ScenarioMatrixCase]:
    return [
        ScenarioMatrixCase(
            case_id="scene_a_light_load_static",
            description="Light-load sanity scene with a few mostly static users.",
            scenario_name="baseline_light_load_static",
            sim_overrides={"total_slots": 20},
            ue_overrides={
                "ues": _build_ue_population(
                    num_ues=3,
                    mix_profile="baseline",
                    velocity_mps=0.2,
                    indoor_ratio=0.0,
                    embb_arrival_rate_bps=3.0e6,
                    urllc_arrival_rate_bps=0.8e6,
                    mmtc_arrival_rate_bps=0.15e6,
                )
            },
            tags=["sanity", "static", "light_load"],
        ),
        ScenarioMatrixCase(
            case_id="scene_b_high_load_few_ues",
            description="High-load throughput-oriented scene with a few good-channel users.",
            scenario_name="high_load_few_ues",
            sim_overrides={
                "total_slots": 24,
                "scheduler_max_ues_per_slot": 4,
                "scheduler_total_prbs_per_slot": 64,
            },
            ue_overrides={
                "ues": _build_ue_population(
                    num_ues=4,
                    mix_profile="embb_heavy",
                    velocity_mps=0.5,
                    indoor_ratio=0.0,
                    embb_arrival_rate_bps=10.0e6,
                    urllc_arrival_rate_bps=1.0e6,
                    mmtc_arrival_rate_bps=0.2e6,
                    radius_start=0.18,
                    radius_end=0.55,
                )
            },
            tags=["throughput", "high_load", "few_ues"],
        ),
        ScenarioMatrixCase(
            case_id="scene_c_mixed_traffic_congested",
            description="Mixed-traffic congested scene for fairness and QoS studies.",
            scenario_name="mixed_traffic_congested",
            sim_overrides={
                "total_slots": 24,
                "scheduler_max_ues_per_slot": 6,
                "scheduler_total_prbs_per_slot": 64,
            },
            ue_overrides={
                "ues": _build_ue_population(
                    num_ues=12,
                    mix_profile="mixed_congested",
                    velocity_mps=1.0,
                    indoor_ratio=0.25,
                    embb_arrival_rate_bps=7.5e6,
                    urllc_arrival_rate_bps=1.2e6,
                    mmtc_arrival_rate_bps=0.35e6,
                    radius_start=0.15,
                    radius_end=0.9,
                )
            },
            tags=["mixed_traffic", "congestion", "fairness"],
        ),
        ScenarioMatrixCase(
            case_id="scene_d_high_mobility_sparse",
            description="Sparse high-mobility scene for Doppler and stale-CSI studies.",
            scenario_name="high_mobility_sparse",
            sim_overrides={"total_slots": 24, "scheduler_max_ues_per_slot": 4},
            ue_overrides={
                "ues": _build_ue_population(
                    num_ues=4,
                    mix_profile="baseline",
                    velocity_mps=32.0,
                    indoor_ratio=0.0,
                    embb_arrival_rate_bps=5.0e6,
                    urllc_arrival_rate_bps=0.9e6,
                    mmtc_arrival_rate_bps=0.2e6,
                    radius_start=0.25,
                    radius_end=0.85,
                )
            },
            tags=["mobility", "doppler", "sparse"],
        ),
        ScenarioMatrixCase(
            case_id="scene_e_cell_edge_nlos",
            description="Coverage-limited NLOS users with stronger HARQ pressure.",
            scenario_name="cell_edge_nlos",
            sim_overrides={"total_slots": 24, "scheduler_max_ues_per_slot": 4},
            ue_overrides={
                "ues": _build_ue_population(
                    num_ues=6,
                    mix_profile="mixed_congested",
                    velocity_mps=1.0,
                    indoor_ratio=0.5,
                    embb_arrival_rate_bps=5.5e6,
                    urllc_arrival_rate_bps=1.0e6,
                    mmtc_arrival_rate_bps=0.25e6,
                    radius_start=0.7,
                    radius_end=0.98,
                )
            },
            tags=["coverage", "edge", "nlos"],
        ),
        ScenarioMatrixCase(
            case_id="scene_f_indoor_hotspot",
            description="Indoor hotspot with wider bandwidth and dense users.",
            scenario_name="indoor_hotspot",
            sim_overrides={
                "total_slots": 24,
                "scheduler_max_ues_per_slot": 6,
                "scheduler_total_prbs_per_slot": 96,
            },
            ue_overrides={
                "ues": _build_ue_population(
                    num_ues=8,
                    mix_profile="embb_heavy",
                    velocity_mps=0.1,
                    indoor_ratio=1.0,
                    embb_arrival_rate_bps=8.5e6,
                    urllc_arrival_rate_bps=1.0e6,
                    mmtc_arrival_rate_bps=0.15e6,
                    radius_start=0.1,
                    radius_end=0.55,
                )
            },
            tags=["indoor", "hotspot", "wideband"],
        ),
        ScenarioMatrixCase(
            case_id="scene_g_mmtc_sparse_reporting",
            description="Large sparse-reporting mMTC scene with many connected users.",
            scenario_name="mmtc_sparse_reporting",
            sim_overrides={"total_slots": 24, "scheduler_max_ues_per_slot": 8},
            ue_overrides={
                "ues": _build_ue_population(
                    num_ues=20,
                    mix_profile="mmtc_heavy",
                    velocity_mps=0.1,
                    indoor_ratio=0.2,
                    embb_arrival_rate_bps=2.0e6,
                    urllc_arrival_rate_bps=0.6e6,
                    mmtc_arrival_rate_bps=0.08e6,
                    radius_start=0.15,
                    radius_end=0.95,
                )
            },
            tags=["mmtc", "sparse_reporting", "many_ues"],
        ),
        ScenarioMatrixCase(
            case_id="scene_h_mixed_control_delay",
            description="Mixed-traffic baseline for observation-delay control studies.",
            scenario_name="mixed_control_delay_scan",
            sim_overrides={"total_slots": 24, "scheduler_max_ues_per_slot": 6},
            ue_overrides={
                "ues": _build_ue_population(
                    num_ues=10,
                    mix_profile="mixed_congested",
                    velocity_mps=1.5,
                    indoor_ratio=0.2,
                    embb_arrival_rate_bps=6.5e6,
                    urllc_arrival_rate_bps=1.1e6,
                    mmtc_arrival_rate_bps=0.25e6,
                    radius_start=0.18,
                    radius_end=0.88,
                )
            },
            tags=["control", "delay_scan", "mixed_traffic"],
        ),
    ]


def build_builtin_scenario_matrices() -> dict[str, ScenarioMatrix]:
    control_delay_base = ScenarioMatrixCase(
        case_id="mixed_control_delay_scan",
        description="Observation-delay sweep over the mixed control baseline.",
        scenario_name="mixed_control_delay_scan",
        sim_overrides={"total_slots": 20, "scheduler_max_ues_per_slot": 6},
        ue_overrides={
            "ues": _build_ue_population(
                num_ues=10,
                mix_profile="mixed_congested",
                velocity_mps=1.5,
                indoor_ratio=0.2,
                embb_arrival_rate_bps=6.5e6,
                urllc_arrival_rate_bps=1.1e6,
                mmtc_arrival_rate_bps=0.25e6,
                radius_start=0.18,
                radius_end=0.88,
            )
        },
        tags=["control", "delay_scan"],
    )
    radio_scan_base = ScenarioMatrixCase(
        case_id="high_load_radio_scan",
        description="Bandwidth and numerology sweep over the high-load few-UE baseline.",
        scenario_name="high_load_few_ues",
        sim_overrides={
            "total_slots": 20,
            "scheduler_max_ues_per_slot": 4,
            "scheduler_total_prbs_per_slot": 64,
        },
        ue_overrides={
            "ues": _build_ue_population(
                num_ues=4,
                mix_profile="embb_heavy",
                velocity_mps=0.5,
                indoor_ratio=0.0,
                embb_arrival_rate_bps=10.0e6,
                urllc_arrival_rate_bps=1.0e6,
                mmtc_arrival_rate_bps=0.2e6,
                radius_start=0.18,
                radius_end=0.55,
            )
        },
        tags=["radio_scan", "throughput"],
    )
    paper_small_cells = _hex_cell_positions(num_cells=3, inter_site_distance_m=420.0)
    paper_medium_cells = _hex_cell_positions(num_cells=7, inter_site_distance_m=420.0)
    paper_large_cells = _hex_cell_positions(num_cells=19, inter_site_distance_m=420.0)
    paper_eval_cases = [
        ScenarioMatrixCase(
            case_id="oran_small_debug_mixed",
            description="Small 3-cell mixed eMBB/URLLC/mMTC scene for debugging and ablations.",
            scenario_name="oran_paper_eval_base",
            template_overrides={
                "topology": {
                    "num_cells": 3,
                    "cell_radius_m": 420.0,
                    "inter_site_distance_m": 420.0,
                    "default_velocity_mps": 1.0,
                    "candidate_cell_limit": 3,
                },
                "metadata": {
                    "cell_positions": paper_small_cells,
                    "cell_profiles": _paper_eval_cell_profiles(
                        cell_positions=paper_small_cells,
                        stress_profile="nominal",
                    ),
                },
            },
            sim_overrides={"total_slots": 48, "scheduler_max_ues_per_slot": 8},
            ue_overrides={
                "ues": _build_paper_eval_ues(
                    num_ues=24,
                    profile="low_mobility",
                    cell_positions=paper_small_cells,
                    load_scale=0.90,
                )
            },
            tags=["paper_eval", "small", "mixed_traffic", "debug"],
            coordinates={
                "scale": "small",
                "cells": 3,
                "ues": 24,
                "mobility": "low",
                "stress": "nominal",
            },
        ),
        ScenarioMatrixCase(
            case_id="oran_medium_nominal_mixed",
            description="Medium 7-cell, 42-UE mixed-service reference scene for main O-RAN orchestration evaluation.",
            scenario_name="oran_paper_eval_base",
            template_overrides={
                "topology": {
                    "num_cells": 7,
                    "cell_radius_m": 450.0,
                    "inter_site_distance_m": 420.0,
                    "default_velocity_mps": 3.0,
                    "candidate_cell_limit": 4,
                },
                "metadata": {
                    "cell_positions": paper_medium_cells,
                    "cell_profiles": _paper_eval_cell_profiles(
                        cell_positions=paper_medium_cells,
                        stress_profile="nominal",
                    ),
                },
            },
            sim_overrides={"total_slots": 60, "scheduler_max_ues_per_slot": 10},
            ue_overrides={
                "ues": _build_paper_eval_ues(
                    num_ues=42,
                    profile="medium_mixed",
                    cell_positions=paper_medium_cells,
                    load_scale=1.00,
                )
            },
            tags=["paper_eval", "medium", "mixed_traffic", "main"],
            coordinates={
                "scale": "medium",
                "cells": 7,
                "ues": 42,
                "mobility": "medium",
                "stress": "nominal",
            },
        ),
        ScenarioMatrixCase(
            case_id="oran_medium_dense_hotspot_high_mobility",
            description="Medium 7-cell dense hotspot with high-mobility and latency-sensitive users.",
            scenario_name="oran_paper_eval_stress",
            template_overrides={
                "topology": {
                    "num_cells": 7,
                    "cell_radius_m": 500.0,
                    "inter_site_distance_m": 420.0,
                    "default_velocity_mps": 18.0,
                    "candidate_cell_limit": 5,
                    "handover_margin_db": 1.8,
                },
                "metadata": {
                    "cell_positions": paper_medium_cells,
                    "cell_profiles": _paper_eval_cell_profiles(
                        cell_positions=paper_medium_cells,
                        stress_profile="hotspot",
                    ),
                },
            },
            sim_overrides={"total_slots": 60, "scheduler_max_ues_per_slot": 10},
            ue_overrides={
                "ues": _build_paper_eval_ues(
                    num_ues=56,
                    profile="dense_hotspot_high_mobility",
                    cell_positions=paper_medium_cells,
                    load_scale=1.25,
                )
            },
            tags=["paper_eval", "medium", "hotspot", "high_mobility", "latency_sensitive"],
            coordinates={
                "scale": "medium",
                "cells": 7,
                "ues": 56,
                "mobility": "high",
                "stress": "dense_hotspot",
            },
        ),
        ScenarioMatrixCase(
            case_id="oran_medium_urllc_burst",
            description="Medium 7-cell scenario with sudden URLLC/control traffic burst.",
            scenario_name="oran_paper_eval_stress",
            template_overrides={
                "topology": {
                    "num_cells": 7,
                    "cell_radius_m": 450.0,
                    "inter_site_distance_m": 420.0,
                    "candidate_cell_limit": 5,
                },
                "metadata": {
                    "cell_positions": paper_medium_cells,
                    "cell_profiles": _paper_eval_cell_profiles(
                        cell_positions=paper_medium_cells,
                        stress_profile="urllc_burst",
                    ),
                },
            },
            sim_overrides={"total_slots": 60, "scheduler_max_ues_per_slot": 10},
            ue_overrides={
                "ues": _build_paper_eval_ues(
                    num_ues=48,
                    profile="urllc_burst",
                    cell_positions=paper_medium_cells,
                    load_scale=1.20,
                )
            },
            tags=["paper_eval", "medium", "urllc", "burst", "latency_sensitive"],
            coordinates={
                "scale": "medium",
                "cells": 7,
                "ues": 48,
                "mobility": "medium",
                "stress": "urllc_burst",
            },
        ),
        ScenarioMatrixCase(
            case_id="oran_medium_interference_edge",
            description="Medium 7-cell edge-user scene with same-band interference burst and weak SINR.",
            scenario_name="oran_paper_eval_stress",
            template_overrides={
                "topology": {
                    "num_cells": 7,
                    "cell_radius_m": 540.0,
                    "inter_site_distance_m": 420.0,
                    "candidate_cell_limit": 5,
                },
                "channel": {
                    "los_mode": "auto",
                    "shadowing_sigma_db_nlos": 9.5,
                    "o2i_penetration_loss_db": 24.0,
                },
                "metadata": {
                    "cell_positions": paper_medium_cells,
                    "cell_profiles": _paper_eval_cell_profiles(
                        cell_positions=paper_medium_cells,
                        stress_profile="interference_burst",
                    ),
                },
            },
            sim_overrides={"total_slots": 60, "scheduler_max_ues_per_slot": 10},
            ue_overrides={
                "ues": _build_paper_eval_ues(
                    num_ues=48,
                    profile="interference_burst",
                    cell_positions=paper_medium_cells,
                    load_scale=1.15,
                )
            },
            tags=["paper_eval", "medium", "interference", "cell_edge", "weak_sinr"],
            coordinates={
                "scale": "medium",
                "cells": 7,
                "ues": 48,
                "mobility": "medium",
                "stress": "interference_edge",
            },
        ),
        ScenarioMatrixCase(
            case_id="oran_medium_hotspot_shift_blockage",
            description="Medium 7-cell scene where hotspot demand shifts and some users experience temporary blockage-style NLOS stress.",
            scenario_name="oran_paper_eval_stress",
            template_overrides={
                "topology": {
                    "num_cells": 7,
                    "cell_radius_m": 500.0,
                    "inter_site_distance_m": 420.0,
                    "candidate_cell_limit": 5,
                },
                "channel": {
                    "shadowing_sigma_db_nlos": 10.0,
                    "o2i_penetration_loss_db": 26.0,
                },
                "metadata": {
                    "cell_positions": paper_medium_cells,
                    "cell_profiles": _paper_eval_cell_profiles(
                        cell_positions=paper_medium_cells,
                        stress_profile="blockage",
                    ),
                },
            },
            sim_overrides={"total_slots": 60, "scheduler_max_ues_per_slot": 10},
            ue_overrides={
                "ues": _build_paper_eval_ues(
                    num_ues=50,
                    profile="hotspot_shift",
                    cell_positions=paper_medium_cells,
                    load_scale=1.20,
                )
            },
            tags=["paper_eval", "medium", "hotspot_shift", "blockage", "nlos"],
            coordinates={
                "scale": "medium",
                "cells": 7,
                "ues": 50,
                "mobility": "medium_high",
                "stress": "hotspot_shift_blockage",
            },
        ),
        ScenarioMatrixCase(
            case_id="oran_medium_high_mobility_interference",
            description=(
                "Medium 7-cell compound OOD scene combining high-mobility users, edge weak-SINR placement, "
                "and same-band inter-cell interference."
            ),
            scenario_name="oran_paper_eval_stress",
            template_overrides={
                "topology": {
                    "num_cells": 7,
                    "cell_radius_m": 560.0,
                    "inter_site_distance_m": 420.0,
                    "default_velocity_mps": 20.0,
                    "candidate_cell_limit": 5,
                    "handover_margin_db": 1.2,
                },
                "channel": {
                    "los_mode": "auto",
                    "shadowing_sigma_db_nlos": 10.5,
                    "o2i_penetration_loss_db": 24.0,
                },
                "metadata": {
                    "cell_positions": paper_medium_cells,
                    "cell_profiles": _paper_eval_cell_profiles(
                        cell_positions=paper_medium_cells,
                        stress_profile="interference_burst",
                    ),
                },
            },
            sim_overrides={"total_slots": 60, "scheduler_max_ues_per_slot": 10},
            ue_overrides={
                "ues": _build_paper_eval_ues(
                    num_ues=54,
                    profile="high_mobility_interference",
                    cell_positions=paper_medium_cells,
                    load_scale=1.18,
                )
            },
            tags=["paper_eval", "medium", "compound_shift", "high_mobility", "interference", "cell_edge"],
            coordinates={
                "scale": "medium",
                "cells": 7,
                "ues": 54,
                "mobility": "high",
                "stress": "high_mobility_interference",
            },
        ),
        ScenarioMatrixCase(
            case_id="oran_medium_low_load_energy_saving",
            description="Medium 7-cell low-load mixed-service scene for energy-saving versus latency/coverage transfer.",
            scenario_name="oran_paper_eval_base",
            template_overrides={
                "topology": {
                    "num_cells": 7,
                    "cell_radius_m": 450.0,
                    "inter_site_distance_m": 420.0,
                    "default_velocity_mps": 1.0,
                    "candidate_cell_limit": 4,
                },
                "metadata": {
                    "cell_positions": paper_medium_cells,
                    "cell_profiles": _paper_eval_cell_profiles(
                        cell_positions=paper_medium_cells,
                        stress_profile="nominal",
                    ),
                },
            },
            sim_overrides={"total_slots": 60, "scheduler_max_ues_per_slot": 10},
            ue_overrides={
                "ues": _build_paper_eval_ues(
                    num_ues=36,
                    profile="low_mobility",
                    cell_positions=paper_medium_cells,
                    load_scale=0.38,
                )
            },
            tags=["paper_eval", "medium", "low_load", "energy_saving", "coverage_guard"],
            coordinates={
                "scale": "medium",
                "cells": 7,
                "ues": 36,
                "mobility": "low",
                "stress": "low_load_energy_saving",
            },
        ),
        ScenarioMatrixCase(
            case_id="oran_large_19cell_scalability",
            description="Large 19-cell, 120-UE scalability scene for orchestration and logging overhead evaluation.",
            scenario_name="oran_paper_eval_base",
            template_overrides={
                "topology": {
                    "num_cells": 19,
                    "cell_radius_m": 500.0,
                    "inter_site_distance_m": 420.0,
                    "default_velocity_mps": 5.0,
                    "candidate_cell_limit": 5,
                },
                "metadata": {
                    "cell_positions": paper_large_cells,
                    "cell_profiles": _paper_eval_cell_profiles(
                        cell_positions=paper_large_cells,
                        stress_profile="large_interference",
                    ),
                },
            },
            sim_overrides={"total_slots": 56, "scheduler_max_ues_per_slot": 12},
            ue_overrides={
                "ues": _build_paper_eval_ues(
                    num_ues=120,
                    profile="medium_mixed",
                    cell_positions=paper_large_cells,
                    load_scale=1.05,
                )
            },
            tags=["paper_eval", "large", "scalability", "mixed_traffic"],
            coordinates={
                "scale": "large",
                "cells": 19,
                "ues": 120,
                "mobility": "medium",
                "stress": "scalability",
            },
        ),
    ]
    return {
        "scenario_catalog": ScenarioMatrix(
            name="scenario_catalog",
            description="Catalog of reusable scene families A-H for regression and analysis.",
            cases=_catalog_cases(),
        ),
        "control_delay_scan": ScenarioMatrix(
            name="control_delay_scan",
            description="Mixed-traffic observation and measurement-window scan.",
            base_case=control_delay_base,
            sweep_dimensions=[
                ScenarioSweepDimension(
                    name="obs_delay_slots",
                    target="template",
                    path="control.obs_delay_slots",
                    values=[
                        ScenarioSweepValue(label="d1", value=1),
                        ScenarioSweepValue(label="d3", value=3),
                        ScenarioSweepValue(label="d5", value=5),
                    ],
                ),
                ScenarioSweepDimension(
                    name="sinr_window_slots",
                    target="template",
                    path="control.measurement_sinr_window_slots",
                    values=[
                        ScenarioSweepValue(label="w4", value=4),
                        ScenarioSweepValue(label="w8", value=8),
                        ScenarioSweepValue(label="w12", value=12),
                    ],
                ),
            ],
        ),
        "radio_bandwidth_scs_scan": ScenarioMatrix(
            name="radio_bandwidth_scs_scan",
            description="Radio sweep over bandwidth and SCS for throughput sensitivity checks.",
            base_case=radio_scan_base,
            sweep_dimensions=[
                ScenarioSweepDimension(
                    name="bandwidth_hz",
                    target="template",
                    path="radio.bandwidth_hz",
                    values=[
                        ScenarioSweepValue(label="bw20", value=20e6),
                        ScenarioSweepValue(label="bw40", value=40e6),
                        ScenarioSweepValue(label="bw80", value=80e6),
                    ],
                ),
                ScenarioSweepDimension(
                    name="subcarrier_spacing_hz",
                    target="template",
                    path="radio.subcarrier_spacing_hz",
                    values=[
                        ScenarioSweepValue(label="scs15", value=15e3),
                        ScenarioSweepValue(label="scs30", value=30e3),
                        ScenarioSweepValue(label="scs60", value=60e3),
                    ],
                ),
            ],
        ),
        "oran_paper_eval_suite": ScenarioMatrix(
            name="oran_paper_eval_suite",
            description=(
                "Paper-ready O-RAN evaluation suite with small, medium, and large multi-cell "
                "scenes spanning mixed traffic, high mobility, URLLC bursts, interference, "
                "blockage, hotspot shift, and scalability."
            ),
            cases=paper_eval_cases,
        ),
        "orchestration_benchmark_suite": ScenarioMatrix(
            name="orchestration_benchmark_suite",
            description="Fixed 1 macro + 2 mmWave small-cell orchestration benchmark suite.",
            cases=[
                ScenarioMatrixCase(
                    case_id="benchmark_nominal_cluster",
                    description="Nominal mixed-service cluster with one macro and two embedded mmWave cells.",
                    scenario_name="heterogeneous_macro_mmwave_benchmark",
                    sim_overrides={"total_slots": 30, "scheduler_max_ues_per_slot": 8},
                    ue_overrides={"ues": _build_macro_mmwave_benchmark_ues("nominal")},
                    tags=["benchmark", "nominal", "macro_mmwave"],
                ),
                ScenarioMatrixCase(
                    case_id="benchmark_hotspot_overload",
                    description="Hotspot overload benchmark focused on the mmWave hotspot small cell.",
                    scenario_name="heterogeneous_macro_mmwave_benchmark",
                    sim_overrides={"total_slots": 30, "scheduler_max_ues_per_slot": 8},
                    ue_overrides={"ues": _build_macro_mmwave_benchmark_ues("hotspot_overload")},
                    tags=["benchmark", "hotspot", "overload", "macro_mmwave"],
                ),
                ScenarioMatrixCase(
                    case_id="benchmark_edge_coverage",
                    description="Coverage-stressed benchmark focused on the edge small cell and macro fallback.",
                    scenario_name="heterogeneous_macro_mmwave_benchmark",
                    sim_overrides={"total_slots": 30, "scheduler_max_ues_per_slot": 8},
                    ue_overrides={"ues": _build_macro_mmwave_benchmark_ues("edge_coverage")},
                    tags=["benchmark", "edge", "coverage", "macro_mmwave"],
                ),
                ScenarioMatrixCase(
                    case_id="benchmark_high_mobility",
                    description="High-mobility benchmark on the same heterogeneous macro/mmWave topology.",
                    scenario_name="heterogeneous_macro_mmwave_benchmark",
                    sim_overrides={"total_slots": 30, "scheduler_max_ues_per_slot": 8},
                    ue_overrides={"ues": _build_macro_mmwave_benchmark_ues("high_mobility")},
                    tags=["benchmark", "mobility", "macro_mmwave"],
                ),
                ScenarioMatrixCase(
                    case_id="benchmark_stale_observation",
                    description="Stale-observation benchmark for orchestration robustness on the same topology.",
                    scenario_name="heterogeneous_macro_mmwave_benchmark",
                    template_overrides={
                        "control": {
                            "obs_delay_slots": 5,
                            "measurement_queue_delay_slots": 4,
                            "measurement_sinr_window_slots": 14,
                            "measurement_throughput_granularity_slots": 14,
                        }
                    },
                    sim_overrides={"total_slots": 30, "scheduler_max_ues_per_slot": 8},
                    ue_overrides={"ues": _build_macro_mmwave_benchmark_ues("nominal")},
                    tags=["benchmark", "stale_observation", "macro_mmwave"],
                ),
            ],
        ),
    }
