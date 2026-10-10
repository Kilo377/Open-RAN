"""
NR PHY/MAC reference helpers used to keep simulator numerology closer to
3GPP-style carrier definitions.
"""

from __future__ import annotations

import math


_NR_MAX_PRBS_TABLE: dict[tuple[str, int, int], int] = {
    ("FR1", 5, 15): 25,
    ("FR1", 10, 15): 52,
    ("FR1", 15, 15): 79,
    ("FR1", 20, 15): 106,
    ("FR1", 25, 15): 133,
    ("FR1", 30, 15): 160,
    ("FR1", 40, 15): 216,
    ("FR1", 50, 15): 270,
    ("FR1", 5, 30): 11,
    ("FR1", 10, 30): 24,
    ("FR1", 15, 30): 38,
    ("FR1", 20, 30): 51,
    ("FR1", 25, 30): 65,
    ("FR1", 30, 30): 78,
    ("FR1", 40, 30): 106,
    ("FR1", 50, 30): 133,
    ("FR1", 60, 30): 162,
    ("FR1", 70, 30): 189,
    ("FR1", 80, 30): 217,
    ("FR1", 90, 30): 245,
    ("FR1", 100, 30): 273,
    ("FR2", 50, 60): 66,
    ("FR2", 100, 60): 132,
    ("FR2", 200, 60): 264,
    ("FR2", 50, 120): 32,
    ("FR2", 100, 120): 66,
    ("FR2", 200, 120): 132,
    ("FR2", 400, 120): 264,
}


def derive_nr_frequency_range(carrier_frequency_hz: float) -> str:
    return "FR2" if float(carrier_frequency_hz) >= 24.25e9 else "FR1"


def derive_nr_numerology_index(subcarrier_spacing_hz: float) -> int:
    normalized_scs_hz = max(15e3, float(subcarrier_spacing_hz))
    return max(0, int(round(math.log2(normalized_scs_hz / 15e3))))


def lookup_nr_max_prbs(
    *,
    bandwidth_hz: float,
    subcarrier_spacing_hz: float,
    carrier_frequency_hz: float,
) -> tuple[int, str]:
    frequency_range = derive_nr_frequency_range(float(carrier_frequency_hz))
    bandwidth_mhz = int(round(float(bandwidth_hz) / 1e6))
    scs_khz = int(round(float(subcarrier_spacing_hz) / 1e3))
    exact_key = (frequency_range, bandwidth_mhz, scs_khz)
    exact_match = _NR_MAX_PRBS_TABLE.get(exact_key)
    if exact_match is not None:
        return int(exact_match), "3gpp_max_transmission_bandwidth_configuration"

    prb_bandwidth_hz = 12.0 * max(1.0, float(subcarrier_spacing_hz))
    fallback_prbs = max(1, int((0.92 * float(bandwidth_hz)) // prb_bandwidth_hz))
    return int(fallback_prbs), "guardband_aware_formula_fallback"

