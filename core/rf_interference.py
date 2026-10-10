"""
RF interference helper utilities shared by mobility and PHY runtime.
"""

from __future__ import annotations

import math
from typing import Any


def classify_rf_band(carrier_frequency_hz: float) -> str:
    if float(carrier_frequency_hz) >= 24.0e9:
        return "mmwave"
    if float(carrier_frequency_hz) >= 6.0e9:
        return "midband_high"
    return "sub6"


def build_default_cell_rf_profile(
    *,
    cell_type: str,
    carrier_frequency_hz: float,
) -> dict[str, Any]:
    normalized_cell_type = str(cell_type).strip().lower()
    rf_band = classify_rf_band(float(carrier_frequency_hz))

    defaults_by_band: dict[str, dict[str, Any]] = {
        "sub6": {
            "adjacent_channel_leakage_ratio_db": 45.0,
            "adjacent_channel_selectivity_db": 33.0,
            "out_of_band_isolation_db": 42.0,
        },
        "midband_high": {
            "adjacent_channel_leakage_ratio_db": 40.0,
            "adjacent_channel_selectivity_db": 31.0,
            "out_of_band_isolation_db": 40.0,
        },
        "mmwave": {
            "adjacent_channel_leakage_ratio_db": 34.0,
            "adjacent_channel_selectivity_db": 27.0,
            "out_of_band_isolation_db": 36.0,
        },
    }
    profile: dict[str, Any] = dict(defaults_by_band[rf_band])

    if "macro" in normalized_cell_type:
        profile["adjacent_channel_leakage_ratio_db"] += 2.0
        profile["adjacent_channel_selectivity_db"] += 1.5
    elif "small" in normalized_cell_type:
        profile["adjacent_channel_leakage_ratio_db"] -= 1.0
    elif "hotspot" in normalized_cell_type:
        profile["out_of_band_isolation_db"] -= 1.0
    elif "edge" in normalized_cell_type:
        profile["adjacent_channel_selectivity_db"] += 1.0

    profile["rf_band"] = rf_band
    profile["rf_interference_class"] = (
        "macro_strict" if "macro" in normalized_cell_type else rf_band
    )
    return profile


def resolve_cell_rf_profile(raw_profile: dict[str, Any] | None) -> dict[str, Any]:
    profile = dict(raw_profile or {})
    cell_type = str(profile.get("cell_type", "macro"))
    carrier_frequency_hz = float(profile.get("carrier_frequency_hz", 3.5e9))
    merged_profile = build_default_cell_rf_profile(
        cell_type=cell_type,
        carrier_frequency_hz=carrier_frequency_hz,
    )
    merged_profile.update(profile)
    merged_profile["cell_type"] = cell_type
    merged_profile["carrier_frequency_hz"] = carrier_frequency_hz
    return merged_profile


def build_frequency_interval_hz(
    *,
    carrier_frequency_hz: float,
    bandwidth_hz: float,
) -> tuple[float, float]:
    normalized_bandwidth_hz = max(1.0, float(bandwidth_hz))
    center_frequency_hz = float(carrier_frequency_hz)
    lower_hz = center_frequency_hz - 0.5 * normalized_bandwidth_hz
    upper_hz = center_frequency_hz + 0.5 * normalized_bandwidth_hz
    return (float(lower_hz), float(upper_hz))


def compute_interval_overlap_fraction(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    width_a = max(0.0, float(end_a) - float(start_a))
    if width_a <= 0.0:
        return 0.0
    overlap = max(0.0, min(float(end_a), float(end_b)) - max(float(start_a), float(start_b)))
    return max(0.0, min(1.0, overlap / width_a))


def compute_interval_guard_band(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    if float(end_a) < float(start_b):
        return max(0.0, float(start_b) - float(end_a))
    if float(end_b) < float(start_a):
        return max(0.0, float(start_a) - float(end_b))
    return 0.0


def compute_frequency_coupling_fraction(
    *,
    target_interval_hz: tuple[float, float],
    interferer_interval_hz: tuple[float, float],
    target_rf_profile: dict[str, Any] | None,
    interferer_rf_profile: dict[str, Any] | None,
    enable_cross_frequency_interference: bool,
    far_frequency_leakage_floor: float,
    guard_decay_scale: float,
) -> tuple[float, float, float, float]:
    overlap_fraction = compute_interval_overlap_fraction(
        float(target_interval_hz[0]),
        float(target_interval_hz[1]),
        float(interferer_interval_hz[0]),
        float(interferer_interval_hz[1]),
    )
    if overlap_fraction > 0.0:
        return (float(overlap_fraction), float(overlap_fraction), 0.0, 0.0)

    guard_band_hz = compute_interval_guard_band(
        float(target_interval_hz[0]),
        float(target_interval_hz[1]),
        float(interferer_interval_hz[0]),
        float(interferer_interval_hz[1]),
    )
    if not enable_cross_frequency_interference:
        return (0.0, 0.0, float(guard_band_hz), math.inf)

    normalized_target_rf_profile = resolve_cell_rf_profile(target_rf_profile)
    normalized_interferer_rf_profile = resolve_cell_rf_profile(interferer_rf_profile)
    target_width_hz = max(1.0, float(target_interval_hz[1]) - float(target_interval_hz[0]))
    interferer_width_hz = max(
        1.0,
        float(interferer_interval_hz[1]) - float(interferer_interval_hz[0]),
    )
    reference_width_hz = max(1.0, min(target_width_hz, interferer_width_hz))
    guard_ratio = float(guard_band_hz) / float(reference_width_hz)
    guard_decay_db = 10.0 * float(guard_ratio) / max(1e-6, float(guard_decay_scale))
    adjacent_rejection_db = float(
        normalized_interferer_rf_profile.get("adjacent_channel_leakage_ratio_db", 45.0)
    ) + float(normalized_target_rf_profile.get("adjacent_channel_selectivity_db", 33.0))
    far_out_rejection_db = float(
        normalized_interferer_rf_profile.get("out_of_band_isolation_db", 42.0)
    ) + float(normalized_target_rf_profile.get("out_of_band_isolation_db", 42.0))
    effective_rejection_db = max(adjacent_rejection_db + guard_decay_db, far_out_rejection_db)
    leakage_fraction = 10.0 ** (-effective_rejection_db / 10.0)
    coupling_fraction = max(float(far_frequency_leakage_floor), float(leakage_fraction))
    return (
        max(0.0, min(1.0, coupling_fraction)),
        0.0,
        float(guard_band_hz),
        float(effective_rejection_db),
    )
