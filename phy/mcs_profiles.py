"""
Shared NR-style MCS profile helpers used by scheduler-side grant planning
and PHY execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class McsProfile:
    """A lightweight MCS profile shared across scheduler and PHY layers."""

    mcs_index: int
    mcs_table_id: str
    modulation_name: str
    bits_per_symbol: int
    code_rate: float
    num_layers: int = 1

    @property
    def spectral_efficiency(self) -> float:
        return float(self.bits_per_symbol) * float(self.code_rate) * float(self.num_layers)


_MCS_BITS_PER_SYMBOL = {
    "qpsk": 2,
    "16qam": 4,
    "64qam": 6,
    "256qam": 8,
}

_MCS_TABLES: dict[str, list[tuple[str, int]]] = {
    # Representative 3GPP-style low-SE table, shaped to stay more conservative
    # than the baseline qam64 table while preserving monotonic growth.
    "qam64_low_se": [
        ("qpsk", 30),
        ("qpsk", 40),
        ("qpsk", 50),
        ("qpsk", 64),
        ("qpsk", 78),
        ("qpsk", 99),
        ("qpsk", 120),
        ("qpsk", 157),
        ("qpsk", 193),
        ("qpsk", 251),
        ("qpsk", 308),
        ("qpsk", 379),
        ("qpsk", 449),
        ("qpsk", 526),
        ("16qam", 340),
        ("16qam", 378),
        ("16qam", 434),
        ("16qam", 490),
        ("16qam", 553),
        ("16qam", 616),
        ("64qam", 438),
        ("64qam", 466),
        ("64qam", 517),
        ("64qam", 567),
        ("64qam", 616),
        ("64qam", 666),
        ("64qam", 719),
        ("64qam", 772),
        ("64qam", 822),
    ],
    # 3GPP TS 38.214-style qam64 table values (target code rate scaled by 1024).
    "qam64": [
        ("qpsk", 120),
        ("qpsk", 157),
        ("qpsk", 193),
        ("qpsk", 251),
        ("qpsk", 308),
        ("qpsk", 379),
        ("qpsk", 449),
        ("qpsk", 526),
        ("qpsk", 602),
        ("qpsk", 679),
        ("16qam", 340),
        ("16qam", 378),
        ("16qam", 434),
        ("16qam", 490),
        ("16qam", 553),
        ("16qam", 616),
        ("16qam", 658),
        ("64qam", 438),
        ("64qam", 466),
        ("64qam", 517),
        ("64qam", 567),
        ("64qam", 616),
        ("64qam", 666),
        ("64qam", 719),
        ("64qam", 772),
        ("64qam", 822),
        ("64qam", 873),
        ("64qam", 910),
        ("64qam", 948),
    ],
    # 3GPP-style qam256 table values used for high-SE eMBB scheduling.
    "qam256": [
        ("qpsk", 120),
        ("qpsk", 193),
        ("qpsk", 308),
        ("qpsk", 449),
        ("qpsk", 602),
        ("16qam", 378),
        ("16qam", 434),
        ("16qam", 490),
        ("16qam", 553),
        ("16qam", 616),
        ("64qam", 438),
        ("64qam", 466),
        ("64qam", 517),
        ("64qam", 567),
        ("64qam", 616),
        ("64qam", 666),
        ("64qam", 719),
        ("64qam", 772),
        ("64qam", 822),
        ("64qam", 873),
        ("256qam", 682),
        ("256qam", 711),
        ("256qam", 754),
        ("256qam", 797),
        ("256qam", 841),
        ("256qam", 885),
        ("256qam", 916),
        ("256qam", 948),
    ],
}


def normalize_mcs_table_id(mcs_table_id: str | None) -> str:
    normalized_table_id = str(mcs_table_id or "qam64").strip().lower()
    if normalized_table_id not in set(_MCS_TABLES):
        return "qam64"
    return normalized_table_id


def normalize_tbs_selection_method(tbs_selection_method: str | None) -> str:
    normalized_method = str(tbs_selection_method or "resource_elements").strip().lower()
    if normalized_method not in {"resource_elements", "guarded_resource_elements", "latency_guarded"}:
        return "resource_elements"
    return normalized_method


def estimate_dmrs_overhead_fraction(
    dmrs_pattern_id: str | None,
    *,
    num_layers: int = 1,
) -> float:
    normalized_pattern_id = str(dmrs_pattern_id or "dmrs_type1_default").strip().lower()
    base_fraction = 0.0714
    if normalized_pattern_id == "dmrs_type1_len2":
        base_fraction = 0.1428
    elif normalized_pattern_id == "dmrs_type1_dense":
        base_fraction = 0.1785
    return min(0.30, max(0.0, base_fraction + 0.01 * max(0, int(num_layers) - 1)))


def _lookup_table_entry(
    mcs_index: int,
    mcs_table_id: str | None,
) -> tuple[int, str, int]:
    normalized_table_id = normalize_mcs_table_id(mcs_table_id)
    table = _MCS_TABLES[normalized_table_id]
    normalized_index = max(0, min(int(mcs_index), len(table) - 1))
    modulation_name, code_rate_x1024 = table[normalized_index]
    return normalized_index, str(modulation_name), int(code_rate_x1024)


def get_mcs_profile(
    mcs_index: int,
    mcs_table_id: str | None = None,
    *,
    num_layers: int = 1,
) -> McsProfile:
    """Returns a 3GPP-style MCS profile for the requested MCS index and table."""

    normalized_index, modulation_name, code_rate_x1024 = _lookup_table_entry(
        mcs_index,
        mcs_table_id,
    )
    normalized_layers = max(1, int(num_layers))
    normalized_table_id = normalize_mcs_table_id(mcs_table_id)
    return McsProfile(
        mcs_index=normalized_index,
        mcs_table_id=normalized_table_id,
        modulation_name=modulation_name,
        bits_per_symbol=int(_MCS_BITS_PER_SYMBOL[modulation_name]),
        code_rate=float(code_rate_x1024) / 1024.0,
        num_layers=normalized_layers,
    )


def estimate_mcs_required_sinr_db(
    mcs_index: int,
    mcs_table_id: str | None = None,
    *,
    num_layers: int = 1,
    bler_target: float = 0.1,
) -> float:
    profile = get_mcs_profile(
        mcs_index,
        mcs_table_id,
        num_layers=num_layers,
    )
    spectral_efficiency = max(0.05, float(profile.spectral_efficiency))
    target_penalty_db = max(0.0, min(3.0, -math.log10(max(1e-4, float(bler_target) / 0.1))))
    layer_penalty_db = 0.6 * max(0, int(profile.num_layers) - 1)
    return -6.0 + 3.5 * spectral_efficiency + target_penalty_db + layer_penalty_db


def select_mcs_index_from_sinr(
    effective_sinr_db: float,
    *,
    mcs_table_id: str | None = None,
    num_layers: int = 1,
    max_mcs_index: int | None = None,
    bler_target: float = 0.1,
    selection_margin_db: float = 1.5,
) -> int:
    normalized_table_id = normalize_mcs_table_id(mcs_table_id)
    table = _MCS_TABLES[normalized_table_id]
    index_limit = len(table) - 1 if max_mcs_index is None else min(len(table) - 1, int(max_mcs_index))
    effective_margin_db = float(effective_sinr_db) - float(selection_margin_db)
    best_index = 0
    for candidate_index in range(0, max(0, index_limit) + 1):
        threshold_db = estimate_mcs_required_sinr_db(
            candidate_index,
            normalized_table_id,
            num_layers=num_layers,
            bler_target=bler_target,
        )
        if effective_margin_db >= threshold_db:
            best_index = candidate_index
        else:
            break
    return int(best_index)


def estimate_tbs_bytes(
    *,
    resource_elements: int,
    mcs_index: int,
    mcs_table_id: str | None = None,
    num_layers: int = 1,
    tbs_selection_method: str | None = None,
    dmrs_pattern_id: str | None = None,
) -> int:
    normalized_re = max(0, int(resource_elements))
    if normalized_re <= 0:
        return 0

    normalized_layers = max(1, int(num_layers))
    dmrs_overhead_fraction = estimate_dmrs_overhead_fraction(
        dmrs_pattern_id,
        num_layers=normalized_layers,
    )
    effective_re = int(max(0.0, float(normalized_re) * (1.0 - dmrs_overhead_fraction)))

    guard_factor = 1.0
    normalized_method = normalize_tbs_selection_method(tbs_selection_method)
    if normalized_method == "guarded_resource_elements":
        guard_factor = 0.90
    elif normalized_method == "latency_guarded":
        guard_factor = 0.85
    effective_re = int(max(0.0, float(effective_re) * guard_factor))

    mcs_profile = get_mcs_profile(
        mcs_index,
        mcs_table_id,
        num_layers=normalized_layers,
    )
    gross_bits = (
        float(effective_re)
        * float(mcs_profile.bits_per_symbol)
        * float(mcs_profile.code_rate)
        * float(mcs_profile.num_layers)
    )
    return int(max(0.0, gross_bits / 8.0))

