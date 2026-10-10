"""
NR-like downlink link-adaptation logic used by the custom PHY adapter.
"""

from __future__ import annotations

import math

from models.mcs_decision import McsDecision
from models.phy_input import PhyUeInput
from phy.mcs_profiles import (
    estimate_dmrs_overhead_fraction,
    estimate_mcs_required_sinr_db,
    get_mcs_profile,
    normalize_mcs_table_id,
    select_mcs_index_from_sinr,
)


class SimpleLinkAdaptation:
    """
    Implements an NR-style downlink link-adaptation module.
    INPUT
    max_mcs_index: The maximum allowed MCS index.
    bler_target: The target BLER value.
    OUTPUT
    A reusable link adaptation object.
    EXAMPLE OUTPUT
    link_adapter = SimpleLinkAdaptation(max_mcs_index=27, bler_target=0.1)
    mcs_decision = link_adapter.select_mcs(ue_input, effective_sinr_db)
    """

    def __init__(
        self,
        max_mcs_index: int = 27,
        bler_target: float = 0.1,
    ) -> None:
        """
        Initializes the link adaptation module.
        INPUT
        max_mcs_index: The maximum allowed MCS index.
        bler_target: The target BLER value.
        OUTPUT
        None.
        EXAMPLE OUTPUT
        self.max_mcs_index = 27
        """
        self.max_mcs_index = max_mcs_index
        self.bler_target = bler_target

    def select_mcs(
        self,
        ue_input: PhyUeInput,
        effective_sinr_db: float,
    ) -> McsDecision:
        """
        Selects one simplified MCS decision for one UE.
        INPUT
        ue_input: The PHY UE input object.
        effective_sinr_db: The effective SINR value in dB.
        OUTPUT
        One McsDecision object.
        EXAMPLE OUTPUT
        McsDecision(ue_id="ue_0", mcs_index=8, modulation_order=4, code_rate=0.5)
        """
        requested_layers = max(
            1,
            int(getattr(ue_input, "layer_count", getattr(ue_input, "rank_indicator", 1)) or 1),
        )
        mcs_table_id = normalize_mcs_table_id(getattr(ue_input, "mcs_table_id", "qam64"))
        selection_margin_db = 1.5
        if float(getattr(ue_input, "latency_budget_ms", 100.0) or 100.0) <= 20.0:
            selection_margin_db += 0.5
        if float(getattr(ue_input, "priority", 1) or 1) >= 3:
            selection_margin_db += 0.25

        mcs_index = select_mcs_index_from_sinr(
            float(effective_sinr_db),
            mcs_table_id=mcs_table_id,
            num_layers=requested_layers,
            max_mcs_index=int(self.max_mcs_index),
            bler_target=float(self.bler_target),
            selection_margin_db=selection_margin_db,
        )
        mcs_profile = get_mcs_profile(
            mcs_index,
            mcs_table_id,
            num_layers=requested_layers,
        )

        return McsDecision(
            ue_id=ue_input.ue_id,
            mcs_index=mcs_index,
            modulation_order=int(mcs_profile.bits_per_symbol),
            code_rate=float(mcs_profile.code_rate),
            num_layers=int(mcs_profile.num_layers),
        )

    def estimate_bler(
        self,
        effective_sinr_db: float,
        mcs_decision: McsDecision,
    ) -> float:
        """
        Estimates one simplified BLER value from effective SINR and MCS.
        INPUT
        effective_sinr_db: The effective SINR value in dB.
        mcs_decision: The selected MCS decision.
        OUTPUT
        One BLER estimate.
        EXAMPLE OUTPUT
        0.08
        """
        threshold_db = estimate_mcs_required_sinr_db(
            int(mcs_decision.mcs_index),
            "qam64" if int(mcs_decision.modulation_order) <= 6 else "qam256",
            num_layers=max(1, int(mcs_decision.num_layers)),
            bler_target=float(self.bler_target),
        )
        margin_db = float(effective_sinr_db) - float(threshold_db) - 1.5
        bler = float(self.bler_target) * math.exp(-0.35 * float(margin_db))
        bler = max(0.001, min(0.99, bler))
        return bler

    def estimate_throughput_mbps(
        self,
        ue_input: PhyUeInput,
        mcs_decision: McsDecision,
        bler: float,
    ) -> float:
        """
        Estimates one simplified throughput value in Mbps.
        INPUT
        ue_input: The PHY UE input object.
        mcs_decision: The selected MCS decision.
        bler: The estimated BLER value.
        OUTPUT
        One throughput estimate in Mbps.
        EXAMPLE OUTPUT
        8.2
        """
        if not ue_input.is_scheduled:
            return 0.0

        allocated_prbs = max(0, int(getattr(ue_input, "allocated_prbs", 0)))
        allocated_symbols = max(0, int(getattr(ue_input, "allocated_symbols", 0)))
        slot_duration_ms = max(0.1, float(getattr(ue_input, "slot_duration_ms", 1.0) or 1.0))
        resource_elements = allocated_prbs * allocated_symbols * 12
        dmrs_overhead_fraction = estimate_dmrs_overhead_fraction(
            getattr(ue_input, "dmrs_pattern_id", None),
            num_layers=max(1, int(mcs_decision.num_layers)),
        )
        effective_re = int(resource_elements * (1.0 - dmrs_overhead_fraction))
        effective_bits = (
            float(effective_re)
            * float(mcs_decision.modulation_order)
            * float(mcs_decision.code_rate)
            * float(max(1, int(mcs_decision.num_layers)))
            * max(0.0, 1.0 - float(bler))
        )
        return max(0.0, effective_bits / 1e6 / (slot_duration_ms / 1000.0))

    def update_queue_bytes(
        self,
        ue_input: PhyUeInput,
        throughput_mbps: float,
    ) -> int:
        """
        Updates one UE queue estimate from scheduled throughput.
        INPUT
        ue_input: The PHY UE input object.
        throughput_mbps: The estimated throughput in Mbps.
        OUTPUT
        One updated queue size in bytes.
        EXAMPLE OUTPUT
        850
        """
        if not ue_input.is_scheduled:
            return ue_input.queue_bytes

        slot_duration_ms = max(0.1, float(getattr(ue_input, "slot_duration_ms", 1.0) or 1.0))
        served_bytes = int(float(throughput_mbps) * 125000.0 * (slot_duration_ms / 1000.0))
        return max(0, ue_input.queue_bytes - served_bytes)
