"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: custom_phy_adapter.py
Description: This file defines a structured custom PHY adapter. The
current implementation is not a full waveform-level PHY, but it already
separates channel modeling, beamforming gain computation, link
adaptation, and probabilistic HARQ ACK/NACK decisions so that it can
later be replaced or refined toward a real PHY.
"""

from core.debug import debug_print
from models.mcs_decision import McsDecision
from models.phy_input import PhyInput
from models.phy_result import PhyResult, PhyUeResult
from phy.beamforming import SimpleBeamformingModel
from phy.channel_model import SimpleChannelModel
from phy.link_adaptation import SimpleLinkAdaptation
from phy.mcs_profiles import estimate_tbs_bytes, get_mcs_profile, normalize_mcs_table_id
from phy.mimo_link_model import SingleUserMimoModel
from phy.phy_base import PhyBase


class CustomPhyAdapter(PhyBase):
    """
    Implements a structured custom PHY adapter.
    INPUT
    codebook_map: The configured codebook mapping dictionary.
    beam_profile_map: The configured beam profile mapping dictionary.
    noise_floor_dbm: The noise floor in dBm.
    base_pathloss_compensation_db: The base pathloss compensation value.
    bler_target: The target BLER value.
    max_mcs_index: The maximum allowed MCS index.
    OUTPUT
    A reusable custom PHY adapter object.
    EXAMPLE OUTPUT
    phy = CustomPhyAdapter(...)
    phy_result = phy.run_slot(phy_input)
    """

    def __init__(
        self,
        codebook_map: dict,
        beam_profile_map: dict,
        noise_floor_dbm: float = -94.0,
        base_pathloss_compensation_db: float = 98.0,
        bler_target: float = 0.1,
        max_mcs_index: int = 15,
        min_valid_sinr_db: float = -40.0,
        max_valid_sinr_db: float = 80.0,
    ) -> None:
        self.noise_floor_dbm = noise_floor_dbm
        self.base_pathloss_compensation_db = base_pathloss_compensation_db
        self.min_valid_sinr_db = float(min_valid_sinr_db)
        self.max_valid_sinr_db = float(max_valid_sinr_db)
        self.channel_model = SimpleChannelModel()
        self.beamforming_model = SimpleBeamformingModel(
            codebook_map=codebook_map,
            beam_profile_map=beam_profile_map,
        )
        self.link_adaptation = SimpleLinkAdaptation(
            max_mcs_index=max_mcs_index,
            bler_target=bler_target,
        )
        self.mimo_model = SingleUserMimoModel()

    def _clip_sinr_db(self, sinr_db: float) -> float:
        minimum = min(float(self.min_valid_sinr_db), float(self.max_valid_sinr_db))
        maximum = max(float(self.min_valid_sinr_db), float(self.max_valid_sinr_db))
        return max(minimum, min(maximum, float(sinr_db)))

    def _estimate_transport_block_size_bytes(
        self,
        ue_input,
        mcs_decision: McsDecision,
    ) -> int:
        resource_elements = self._count_resource_elements(ue_input)
        if resource_elements <= 0:
            return 0

        tb_size_bytes = estimate_tbs_bytes(
            resource_elements=resource_elements,
            mcs_index=int(mcs_decision.mcs_index),
            mcs_table_id=str(getattr(ue_input, "mcs_table_id", "qam64")),
            num_layers=int(mcs_decision.num_layers),
            tbs_selection_method=str(getattr(ue_input, "tbs_selection_method", "resource_elements")),
            dmrs_pattern_id=getattr(ue_input, "dmrs_pattern_id", None),
        )
        requested_bytes = int(getattr(ue_input, "requested_bytes", getattr(ue_input, "queue_bytes", 0)))
        grant_bytes_budget = int(getattr(ue_input, "grant_bytes_budget", requested_bytes))
        effective_budget = grant_bytes_budget if grant_bytes_budget > 0 else requested_bytes
        return min(max(0, requested_bytes), max(0, effective_budget), tb_size_bytes)

    def _count_resource_elements(self, ue_input) -> int:
        resource_grid_slice = getattr(ue_input, "resource_grid_slice", None)
        if resource_grid_slice is not None:
            explicit_resource_elements = int(getattr(resource_grid_slice, "resource_elements", 0))
            if explicit_resource_elements > 0:
                return explicit_resource_elements

        allocated_subcarrier_indices = list(getattr(ue_input, "allocated_subcarrier_indices", []) or [])
        allocated_symbol_indices = list(getattr(ue_input, "allocated_symbol_indices", []) or [])
        if allocated_subcarrier_indices and allocated_symbol_indices:
            return int(len(allocated_subcarrier_indices) * len(allocated_symbol_indices))

        resource_block_indices = list(getattr(ue_input, "resource_block_indices", []) or [])
        if resource_block_indices and allocated_symbol_indices:
            return int(len(resource_block_indices) * 12 * len(allocated_symbol_indices))

        return (
            max(0, int(getattr(ue_input, "allocated_prbs", 0)))
            * max(0, int(getattr(ue_input, "allocated_symbols", 0)))
            * 12
        )

    def _apply_grant_mcs_controls(
        self,
        ue_input,
        mcs_decision: McsDecision,
    ) -> McsDecision:
        adjusted_index = int(mcs_decision.mcs_index)
        target_mcs = getattr(ue_input, "target_mcs", None)
        mcs_cap = getattr(ue_input, "mcs_cap", None)
        requested_layers_raw = getattr(ue_input, "layer_count", None)
        if requested_layers_raw is None:
            requested_layers_raw = getattr(ue_input, "rank_indicator", 1)
        requested_layers = max(1, int(requested_layers_raw or 1))

        if target_mcs is not None:
            adjusted_index = int(target_mcs)
        if mcs_cap is not None:
            adjusted_index = min(int(adjusted_index), int(mcs_cap))
        adjusted_index = max(0, min(int(self.link_adaptation.max_mcs_index), int(adjusted_index)))

        if adjusted_index == int(mcs_decision.mcs_index):
            return McsDecision(
                ue_id=str(getattr(ue_input, "ue_id", "unknown_ue")),
                mcs_index=int(mcs_decision.mcs_index),
                modulation_order=int(mcs_decision.modulation_order),
                code_rate=float(mcs_decision.code_rate),
                num_layers=requested_layers,
            )

        mcs_profile = get_mcs_profile(
            adjusted_index,
            normalize_mcs_table_id(getattr(ue_input, "mcs_table_id", "qam64")),
            num_layers=requested_layers,
        )
        return McsDecision(
            ue_id=str(getattr(ue_input, "ue_id", "unknown_ue")),
            mcs_index=int(adjusted_index),
            modulation_order=int(mcs_profile.bits_per_symbol),
            code_rate=float(mcs_profile.code_rate),
            num_layers=requested_layers,
        )

    def _build_repeatable_random_value(
        self,
        slot_id: int,
        ue_id: str,
        retransmission_count: int,
    ) -> float:
        ue_hash = sum(ord(character) for character in ue_id)
        raw_value = (
            37 * slot_id
            + 17 * ue_hash
            + 53 * retransmission_count
            + 19
        ) % 1000
        return raw_value / 1000.0

    def _decide_ack(
        self,
        slot_id: int,
        ue_id: str,
        bler: float,
        is_scheduled: bool,
        retransmission_count: int,
    ) -> bool:
        if not is_scheduled:
            return False

        success_probability = max(0.0, min(1.0, 1.0 - bler))
        random_value = self._build_repeatable_random_value(
            slot_id=slot_id,
            ue_id=ue_id,
            retransmission_count=retransmission_count,
        )
        return random_value <= success_probability

    def run_slot(self, phy_input: PhyInput) -> PhyResult:
        """
        Executes one structured custom PHY slot update.
        INPUT
        phy_input: The PHY input object for the current slot.
        OUTPUT
        One PhyResult object containing updated per-UE results.
        EXAMPLE OUTPUT
        PhyResult(slot_id=1, ue_results=[...])
        """
        ue_results = []

        runtime_context = phy_input.runtime_context
        if runtime_context is None:
            raise ValueError("CustomPhyAdapter requires phy_input.runtime_context")

        slot_duration_ms = float(getattr(runtime_context, "slot_duration_ms", 1.0))
        slot_duration_s = max(slot_duration_ms, 1e-6) / 1000.0

        scheduled_ue_inputs = [
            ue_input for ue_input in phy_input.ue_inputs
            if bool(getattr(ue_input, "is_scheduled", False))
        ]

        for ue_input in scheduled_ue_inputs:
            channel_sinr_db = self.channel_model.compute_effective_sinr(
                ue_input=ue_input,
                runtime_context=runtime_context,
            )

            beam_gain_db = self.beamforming_model.compute_beam_gain(
                ue_input=ue_input,
                runtime_context=runtime_context,
            )

            tx_power_gain_db = phy_input.tx_power_dbm - 30.0
            pathloss_compensation_db = 0.05 * (
                self.base_pathloss_compensation_db
                - runtime_context.pathloss_db
            )

            effective_sinr_db = (
                channel_sinr_db
                + beam_gain_db
                + tx_power_gain_db
                + pathloss_compensation_db
            )
            effective_sinr_db = self._clip_sinr_db(effective_sinr_db)
            mimo_evaluation = self.mimo_model.evaluate(
                pre_eq_sinr_db=effective_sinr_db,
                num_tx_ant=int(getattr(ue_input, "num_tx_ant", 1) or 1),
                num_rx_ant=int(getattr(ue_input, "num_rx_ant", 1) or 1),
                layer_count=int(getattr(ue_input, "layer_count", 1) or 1),
                rank_indicator=int(getattr(ue_input, "rank_indicator", 1) or 1),
                precoder_id=int(getattr(ue_input, "precoder_id", 0) or 0),
                combiner_id=int(getattr(ue_input, "combiner_id", 0) or 0),
                reported_ri=getattr(ue_input, "reported_ri", None),
                reported_pmi=getattr(ue_input, "reported_pmi", None),
                delay_spread_ns=float(getattr(ue_input, "link_delay_spread_ns", 300.0) or 300.0),
                doppler_hz=float(getattr(ue_input, "link_doppler_hz", getattr(runtime_context, "doppler_hz", 0.0)) or 0.0),
                channel_model_type=str(getattr(ue_input, "link_channel_model_type", runtime_context.channel_profile)),
                k_factor_db=float(getattr(ue_input, "link_k_factor_db", 0.0) or 0.0),
                angle_spread_deg=float(getattr(ue_input, "link_angle_spread_deg", 35.0) or 35.0),
                spatial_correlation_level=float(
                    getattr(ue_input, "link_spatial_correlation_level", 0.5) or 0.5
                ),
            )
            post_eq_sinr_db = float(mimo_evaluation.post_eq_sinr_db)
            spatial_group_id = getattr(ue_input, "spatial_group_id", None)
            spatial_group_size = max(1, int(getattr(ue_input, "spatial_group_size", 1) or 1))
            spatial_compatibility_score = float(getattr(ue_input, "spatial_compatibility_score", 0.0) or 0.0)
            spatial_multiplexing_gain_db = 0.0
            spatial_mode = str(mimo_evaluation.spatial_mode)
            inter_cell_interference_penalty_db = float(
                getattr(ue_input, "inter_cell_interference_penalty_db", 0.0) or 0.0
            )
            if spatial_group_id is not None and spatial_group_size > 1:
                spatial_multiplexing_gain_db = min(2.0, 1.5 * max(0.0, spatial_compatibility_score))
                post_eq_sinr_db += float(spatial_multiplexing_gain_db)
                spatial_mode = "mu_mimo"
            post_eq_sinr_db -= float(inter_cell_interference_penalty_db)
            post_eq_sinr_db = self._clip_sinr_db(post_eq_sinr_db)

            mcs_decision = self.link_adaptation.select_mcs(
                ue_input=ue_input,
                effective_sinr_db=post_eq_sinr_db,
            )
            mcs_decision = self._apply_grant_mcs_controls(
                ue_input=ue_input,
                mcs_decision=mcs_decision,
            )

            bler = self.link_adaptation.estimate_bler(
                effective_sinr_db=post_eq_sinr_db,
                mcs_decision=mcs_decision,
            )

            transport_block_size_bytes = self._estimate_transport_block_size_bytes(
                ue_input=ue_input,
                mcs_decision=mcs_decision,
            )

            retransmission_count = int(getattr(ue_input, "harq_retransmission_round", 0))
            ack = self._decide_ack(
                slot_id=phy_input.slot_id,
                ue_id=ue_input.ue_id,
                bler=bler,
                is_scheduled=ue_input.is_scheduled,
                retransmission_count=retransmission_count,
            )
            throughput_mbps = (
                float(transport_block_size_bytes * 8.0) / 1e6 / max(slot_duration_s, 1e-9)
                if ack else 0.0
            )
            delivered_bytes = int(transport_block_size_bytes) if ack else 0
            resource_grid_re_count = self._count_resource_elements(ue_input)

            ue_results.append(
                PhyUeResult(
                    ue_id=ue_input.ue_id,
                    sinr_db=post_eq_sinr_db,
                    pre_eq_sinr_db=effective_sinr_db,
                    post_eq_sinr_db=post_eq_sinr_db,
                    bler=bler,
                    throughput_mbps=throughput_mbps,
                    queue_bytes=ue_input.queue_bytes,
                    effective_sinr_db=post_eq_sinr_db,
                    beam_gain_db=beam_gain_db,
                    channel_rank_limit=int(mimo_evaluation.channel_rank_limit),
                    precoder_match_score=float(mimo_evaluation.precoder_match_score),
                    spatial_mode=str(spatial_mode),
                    spatial_group_id=(int(spatial_group_id) if spatial_group_id is not None else None),
                    spatial_group_size=int(spatial_group_size),
                    spatial_compatibility_score=float(spatial_compatibility_score),
                    spatial_multiplexing_gain_db=float(spatial_multiplexing_gain_db),
                    mcs_index=mcs_decision.mcs_index,
                    modulation_order=mcs_decision.modulation_order,
                    code_rate=mcs_decision.code_rate,
                    num_layers=mcs_decision.num_layers,
                    is_scheduled=ue_input.is_scheduled,
                    allocated_prbs=int(getattr(ue_input, "allocated_prbs", 0)),
                    allocated_symbols=int(getattr(ue_input, "allocated_symbols", 0)),
                    start_prb=int(getattr(ue_input, "start_prb", 0)),
                    start_symbol=int(getattr(ue_input, "start_symbol", 0)),
                    is_retx=bool(getattr(ue_input, "is_retx", False)),
                    requested_bytes=int(getattr(ue_input, "requested_bytes", 0)),
                    grant_bytes_budget=int(getattr(ue_input, "grant_bytes_budget", 0)),
                    mcs_table_id=str(getattr(ue_input, "mcs_table_id", "qam64")),
                    tbs_selection_method=str(getattr(ue_input, "tbs_selection_method", "resource_elements")),
                    dmrs_pattern_id=getattr(ue_input, "dmrs_pattern_id", None),
                    harq_process_id=getattr(ue_input, "harq_process_id", None),
                    harq_redundancy_version=getattr(ue_input, "harq_redundancy_version", None),
                    harq_ndi=getattr(ue_input, "harq_ndi", None),
                    harq_retransmission_round=int(getattr(ue_input, "harq_retransmission_round", 0)),
                    cqi_validity_age_slots=int(getattr(ue_input, "cqi_validity_age_slots", 0)),
                    ri_validity_age_slots=int(getattr(ue_input, "ri_validity_age_slots", 0)),
                    pmi_validity_age_slots=int(getattr(ue_input, "pmi_validity_age_slots", 0)),
                    channel_quality_confidence=(
                        float(getattr(ue_input, "channel_quality_confidence", 1.0))
                        if getattr(ue_input, "channel_quality_confidence", None) is not None
                        else None
                    ),
                    link_adaptation_reason=getattr(ue_input, "link_adaptation_reason", None),
                    scheduled_bytes=int(transport_block_size_bytes),
                    delivered_bytes=delivered_bytes,
                    tb_size_bytes=transport_block_size_bytes,
                    transport_block_size_bytes=transport_block_size_bytes,
                    ack=ack,
                    harq_retransmission_count=retransmission_count,
                    resource_grid_re_count=int(resource_grid_re_count),
                    interference_penalty_db=float(inter_cell_interference_penalty_db),
                    intra_cell_interference_penalty_db=0.0,
                    inter_cell_interference_dbm=float(
                        getattr(ue_input, "inter_cell_interference_dbm", -120.0) or -120.0
                    ),
                    inter_cell_interference_penalty_db=float(inter_cell_interference_penalty_db),
                    inter_cell_interference_overlap_fraction=float(
                        getattr(ue_input, "inter_cell_interference_overlap_fraction", 0.0) or 0.0
                    ),
                    inter_cell_frequency_coupling_fraction=float(
                        getattr(ue_input, "inter_cell_frequency_coupling_fraction", 0.0) or 0.0
                    ),
                    inter_cell_interference_source_count=int(
                        getattr(ue_input, "inter_cell_interference_source_count", 0) or 0
                    ),
                )
            )

        debug_print(
            f"Custom PHY executed for slot {phy_input.slot_id}. "
            f"tx_power_dbm={phy_input.tx_power_dbm:.2f}, "
            f"codebook_id={phy_input.current_codebook_id}, "
            f"beam_profile_id={phy_input.current_beam_profile_id}, "
            f"scheduled_ues={phy_input.scheduled_ue_ids}."
        )

        scheduled_results = [item for item in ue_results if bool(getattr(item, "is_scheduled", False))]
        if scheduled_results:
            avg_sinr_db = sum(float(item.sinr_db) for item in scheduled_results) / float(len(scheduled_results))
            avg_bler = sum(float(item.bler) for item in scheduled_results) / float(len(scheduled_results))
            total_tb_size_bytes = sum(int(getattr(item, "transport_block_size_bytes", 0)) for item in scheduled_results)
        else:
            avg_sinr_db = 0.0
            avg_bler = 0.0
            total_tb_size_bytes = 0

        return PhyResult(
            slot_id=phy_input.slot_id,
            cell_id=str(getattr(phy_input, "cell_id", "cell_0")),
            ue_results=ue_results,
            avg_sinr_db=avg_sinr_db,
            avg_bler=avg_bler,
            total_tb_size_bytes=total_tb_size_bytes,
        )
