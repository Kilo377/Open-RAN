"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: sionna_phy_adapter.py
Description: This file defines the Sionna PHY adapter that translates the
simulator PHY input into batched engine input, then maps engine output
back to simulator PHY result objects. In O-RAN-style terminology, this is
an abstract O-DU Low / L1-facing executor for downlink PDSCH-like transport,
not an explicit O-RU waveform implementation.
"""

from typing import Any, Dict, List

from core.debug import debug_print
from models.phy_result import PhyResult, PhyUeResult
from phy.sionna_engine import SionnaEngine


class SionnaPhyAdapter:
    """
    L1 Design:
    Converts simulator PHY input into Sionna engine input,
    executes batched PHY simulation, and converts engine output
    back into simulator PHY result objects.
    """

    def __init__(self, sionna_config: Dict[str, Any] | None = None) -> None:
        self.sionna_config = sionna_config or {}
        self.engine = SionnaEngine(self.sionna_config)

    def run_slot(self, phy_input) -> PhyResult:
        return self.execute(phy_input)

    def execute(self, phy_input) -> PhyResult:
        ue_input_list = list(getattr(phy_input, "ue_inputs", []))
        scheduled_ue_inputs = [
            ue_input for ue_input in ue_input_list
            if bool(getattr(ue_input, "is_scheduled", False))
        ]

        engine_inputs = []
        for ue_input in scheduled_ue_inputs:
            engine_inputs.append(
                self.engine.build_engine_input(
                    phy_input=phy_input,
                    ue_input=ue_input,
                )
            )

        engine_results = (
            self.engine.run_batch_link_simulation(engine_inputs)
            if engine_inputs else []
        )

        ue_results: List[PhyUeResult] = []
        for ue_input, engine_result in zip(scheduled_ue_inputs, engine_results):
            ue_results.append(
                self._build_ue_result(
                    ue_input=ue_input,
                    engine_result=engine_result,
                )
            )

        phy_result = self._build_phy_result(
            phy_input=phy_input,
            ue_results=ue_results,
        )

        scheduled_ue_ids = list(getattr(phy_input, "scheduled_ue_ids", []))
        scheduled_mask = [
            bool(getattr(item, "is_scheduled", False))
            for item in ue_input_list
        ]
        debug_print(
            "Sionna PHY adapter executed for slot "
            f"{getattr(phy_input, 'slot_id', -1)}. "
            f"tx_power_dbm={getattr(phy_input, 'tx_power_dbm', 0.0):.2f}, "
            f"codebook_id={getattr(phy_input, 'current_codebook_id', -1)}, "
            f"beam_profile_id={getattr(phy_input, 'current_beam_profile_id', -1)}, "
            f"scheduled_ue_ids={scheduled_ue_ids}, "
            f"ue_input_is_scheduled={scheduled_mask}."
        )

        return phy_result

    def _build_ue_result(self, ue_input, engine_result) -> PhyUeResult:
        is_scheduled = bool(getattr(ue_input, "is_scheduled", False))
        allocated_prbs = int(getattr(ue_input, "allocated_prbs", 0))
        allocated_symbols = int(getattr(ue_input, "allocated_symbols", 0))
        start_prb = int(getattr(ue_input, "start_prb", 0))
        start_symbol = int(getattr(ue_input, "start_symbol", 0))
        is_retx = bool(getattr(ue_input, "is_retx", False))
        requested_bytes = int(getattr(ue_input, "requested_bytes", 0))
        grant_bytes_budget = int(getattr(ue_input, "grant_bytes_budget", 0))
        mcs_table_id = str(getattr(ue_input, "mcs_table_id", "qam64"))
        tbs_selection_method = str(getattr(ue_input, "tbs_selection_method", "resource_elements"))
        dmrs_pattern_id = getattr(ue_input, "dmrs_pattern_id", None)
        harq_process_id = getattr(ue_input, "harq_process_id", None)
        harq_redundancy_version = getattr(ue_input, "harq_redundancy_version", None)
        harq_ndi = getattr(ue_input, "harq_ndi", None)
        harq_retransmission_round = int(getattr(ue_input, "harq_retransmission_round", 0) or 0)
        cqi_validity_age_slots = int(getattr(ue_input, "cqi_validity_age_slots", 0) or 0)
        ri_validity_age_slots = int(getattr(ue_input, "ri_validity_age_slots", 0) or 0)
        pmi_validity_age_slots = int(getattr(ue_input, "pmi_validity_age_slots", 0) or 0)
        spatial_group_id = getattr(ue_input, "spatial_group_id", None)
        spatial_group_size = int(getattr(ue_input, "spatial_group_size", 1) or 1)
        spatial_compatibility_score = float(getattr(ue_input, "spatial_compatibility_score", 0.0) or 0.0)
        resource_grid_re_count = 0
        resource_grid_overlap_fraction = 0.0
        interference_penalty_db = 0.0
        intra_cell_interference_penalty_db = 0.0
        inter_cell_interference_dbm = float(getattr(ue_input, "inter_cell_interference_dbm", -120.0) or -120.0)
        inter_cell_interference_penalty_db = float(
            getattr(ue_input, "inter_cell_interference_penalty_db", 0.0) or 0.0
        )
        inter_cell_interference_overlap_fraction = float(
            getattr(ue_input, "inter_cell_interference_overlap_fraction", 0.0) or 0.0
        )
        inter_cell_frequency_coupling_fraction = float(
            getattr(ue_input, "inter_cell_frequency_coupling_fraction", 0.0) or 0.0
        )
        inter_cell_interference_source_count = int(
            getattr(ue_input, "inter_cell_interference_source_count", 0) or 0
        )
        spatial_multiplexing_gain_db = 0.0
        phy_mode = "system_level_approx"
        waveform_backend = None
        effective_data_re_count = 0
        dmrs_overhead_fraction = 0.0
        channel_quality_confidence = getattr(ue_input, "channel_quality_confidence", None)
        link_adaptation_reason = getattr(ue_input, "link_adaptation_reason", None)

        if isinstance(engine_result, dict):
            sinr_db = float(engine_result.get("sinr_db", engine_result.get("effective_sinr_db", 0.0)))
            effective_sinr_db = float(engine_result.get("effective_sinr_db", sinr_db))
            pre_eq_sinr_db = float(engine_result.get("pre_eq_sinr_db", effective_sinr_db))
            post_eq_sinr_db = float(engine_result.get("post_eq_sinr_db", effective_sinr_db))
            beam_gain_db = float(engine_result.get("beam_gain_db", 0.0))
            bler = float(engine_result.get("bler", 0.0))
            ber = float(engine_result.get("ber", 0.0))
            mcs_index = int(engine_result.get("mcs_index", 0))
            modulation_order = int(engine_result.get("modulation_order", 0))
            code_rate = float(engine_result.get("code_rate", 0.0))
            num_layers = int(engine_result.get("num_layers", 1))
            tb_size_bytes = int(engine_result.get("tb_size_bytes", engine_result.get("transport_block_size_bytes", 0)))
            ack = bool(engine_result.get("ack", False))
            throughput_mbps = float(engine_result.get("throughput_mbps", 0.0))
            cqi = int(engine_result.get("cqi", 0))
            rank = int(engine_result.get("rank", 1))
            pmi = int(engine_result.get("pmi", 0))
            ri = int(engine_result.get("ri", 1))
            channel_rank_limit = int(engine_result.get("channel_rank_limit", rank))
            precoder_match_score = float(engine_result.get("precoder_match_score", 1.0))
            spatial_mode = str(engine_result.get("spatial_mode", "su_siso"))
            spatial_group_id = engine_result.get("spatial_group_id", spatial_group_id)
            spatial_group_size = int(engine_result.get("spatial_group_size", spatial_group_size))
            spatial_compatibility_score = float(
                engine_result.get("spatial_compatibility_score", spatial_compatibility_score)
            )
            spatial_multiplexing_gain_db = float(
                engine_result.get("spatial_multiplexing_gain_db", 0.0)
            )
            allocated_prbs = int(engine_result.get("allocated_prbs", allocated_prbs))
            allocated_symbols = int(engine_result.get("allocated_symbols", allocated_symbols))
            start_prb = int(engine_result.get("start_prb", start_prb))
            start_symbol = int(engine_result.get("start_symbol", start_symbol))
            is_retx = bool(engine_result.get("is_retx", is_retx))
            requested_bytes = int(engine_result.get("requested_bytes", requested_bytes))
            grant_bytes_budget = int(engine_result.get("grant_bytes_budget", grant_bytes_budget))
            mcs_table_id = str(engine_result.get("mcs_table_id", mcs_table_id))
            tbs_selection_method = str(engine_result.get("tbs_selection_method", tbs_selection_method))
            dmrs_pattern_id = engine_result.get("dmrs_pattern_id", dmrs_pattern_id)
            harq_process_id = engine_result.get("harq_process_id", harq_process_id)
            harq_redundancy_version = engine_result.get("harq_redundancy_version", harq_redundancy_version)
            harq_ndi = engine_result.get("harq_ndi", harq_ndi)
            harq_retransmission_round = int(
                engine_result.get("harq_retransmission_round", harq_retransmission_round)
            )
            cqi_validity_age_slots = int(engine_result.get("cqi_validity_age_slots", cqi_validity_age_slots))
            ri_validity_age_slots = int(engine_result.get("ri_validity_age_slots", ri_validity_age_slots))
            pmi_validity_age_slots = int(engine_result.get("pmi_validity_age_slots", pmi_validity_age_slots))
            channel_quality_confidence = engine_result.get(
                "channel_quality_confidence",
                channel_quality_confidence,
            )
            link_adaptation_reason = engine_result.get(
                "link_adaptation_reason",
                link_adaptation_reason,
            )
            scheduled_bytes = int(engine_result.get("scheduled_bytes", tb_size_bytes))
            delivered_bytes = int(engine_result.get("delivered_bytes", tb_size_bytes if ack else 0))
            resource_grid_re_count = int(engine_result.get("resource_grid_re_count", 0))
            resource_grid_overlap_fraction = float(engine_result.get("resource_grid_overlap_fraction", 0.0))
            interference_penalty_db = float(engine_result.get("interference_penalty_db", 0.0))
            intra_cell_interference_penalty_db = float(
                engine_result.get("intra_cell_interference_penalty_db", 0.0)
            )
            inter_cell_interference_dbm = float(
                engine_result.get("inter_cell_interference_dbm", inter_cell_interference_dbm)
            )
            inter_cell_interference_penalty_db = float(
                engine_result.get(
                    "inter_cell_interference_penalty_db",
                    inter_cell_interference_penalty_db,
                )
            )
            inter_cell_interference_overlap_fraction = float(
                engine_result.get(
                    "inter_cell_interference_overlap_fraction",
                    inter_cell_interference_overlap_fraction,
                )
            )
            inter_cell_frequency_coupling_fraction = float(
                engine_result.get(
                    "inter_cell_frequency_coupling_fraction",
                    inter_cell_frequency_coupling_fraction,
                )
            )
            inter_cell_interference_source_count = int(
                engine_result.get(
                    "inter_cell_interference_source_count",
                    inter_cell_interference_source_count,
                )
            )
            phy_mode = str(engine_result.get("phy_mode", phy_mode))
            waveform_backend = engine_result.get("waveform_backend", waveform_backend)
            effective_data_re_count = int(engine_result.get("effective_data_re_count", 0))
            dmrs_overhead_fraction = float(engine_result.get("dmrs_overhead_fraction", 0.0))
        else:
            sinr_db = float(getattr(engine_result, "sinr_db", getattr(engine_result, "effective_sinr_db", 0.0)))
            effective_sinr_db = float(getattr(engine_result, "effective_sinr_db", sinr_db))
            pre_eq_sinr_db = float(getattr(engine_result, "pre_eq_sinr_db", effective_sinr_db))
            post_eq_sinr_db = float(getattr(engine_result, "post_eq_sinr_db", effective_sinr_db))
            beam_gain_db = float(getattr(engine_result, "beam_gain_db", 0.0))
            bler = float(getattr(engine_result, "bler", 0.0))
            ber = float(getattr(engine_result, "ber", 0.0))
            mcs_index = int(getattr(engine_result, "mcs_index", 0))
            modulation_order = int(getattr(engine_result, "modulation_order", 0))
            code_rate = float(getattr(engine_result, "code_rate", 0.0))
            num_layers = int(getattr(engine_result, "num_layers", 1))
            tb_size_bytes = int(getattr(engine_result, "tb_size_bytes", getattr(engine_result, "transport_block_size_bytes", 0)))
            ack = bool(getattr(engine_result, "ack", False))
            throughput_mbps = float(getattr(engine_result, "throughput_mbps", 0.0))
            cqi = int(getattr(engine_result, "cqi", 0))
            rank = int(getattr(engine_result, "rank", 1))
            pmi = int(getattr(engine_result, "pmi", 0))
            ri = int(getattr(engine_result, "ri", 1))
            channel_rank_limit = int(getattr(engine_result, "channel_rank_limit", rank))
            precoder_match_score = float(getattr(engine_result, "precoder_match_score", 1.0))
            spatial_mode = str(getattr(engine_result, "spatial_mode", "su_siso"))
            spatial_group_id = getattr(engine_result, "spatial_group_id", spatial_group_id)
            spatial_group_size = int(getattr(engine_result, "spatial_group_size", spatial_group_size))
            spatial_compatibility_score = float(
                getattr(engine_result, "spatial_compatibility_score", spatial_compatibility_score)
            )
            spatial_multiplexing_gain_db = float(
                getattr(engine_result, "spatial_multiplexing_gain_db", 0.0)
            )
            allocated_prbs = int(getattr(engine_result, "allocated_prbs", allocated_prbs))
            allocated_symbols = int(getattr(engine_result, "allocated_symbols", allocated_symbols))
            start_prb = int(getattr(engine_result, "start_prb", start_prb))
            start_symbol = int(getattr(engine_result, "start_symbol", start_symbol))
            is_retx = bool(getattr(engine_result, "is_retx", is_retx))
            requested_bytes = int(getattr(engine_result, "requested_bytes", requested_bytes))
            grant_bytes_budget = int(getattr(engine_result, "grant_bytes_budget", grant_bytes_budget))
            mcs_table_id = str(getattr(engine_result, "mcs_table_id", mcs_table_id))
            tbs_selection_method = str(getattr(engine_result, "tbs_selection_method", tbs_selection_method))
            dmrs_pattern_id = getattr(engine_result, "dmrs_pattern_id", dmrs_pattern_id)
            harq_process_id = getattr(engine_result, "harq_process_id", harq_process_id)
            harq_redundancy_version = getattr(engine_result, "harq_redundancy_version", harq_redundancy_version)
            harq_ndi = getattr(engine_result, "harq_ndi", harq_ndi)
            harq_retransmission_round = int(
                getattr(engine_result, "harq_retransmission_round", harq_retransmission_round)
            )
            cqi_validity_age_slots = int(getattr(engine_result, "cqi_validity_age_slots", cqi_validity_age_slots))
            ri_validity_age_slots = int(getattr(engine_result, "ri_validity_age_slots", ri_validity_age_slots))
            pmi_validity_age_slots = int(getattr(engine_result, "pmi_validity_age_slots", pmi_validity_age_slots))
            channel_quality_confidence = getattr(
                engine_result,
                "channel_quality_confidence",
                getattr(ue_input, "channel_quality_confidence", None),
            )
            link_adaptation_reason = getattr(
                engine_result,
                "link_adaptation_reason",
                getattr(ue_input, "link_adaptation_reason", None),
            )
            scheduled_bytes = int(getattr(engine_result, "scheduled_bytes", tb_size_bytes))
            delivered_bytes = int(getattr(engine_result, "delivered_bytes", tb_size_bytes if ack else 0))
            resource_grid_re_count = int(getattr(engine_result, "resource_grid_re_count", 0))
            resource_grid_overlap_fraction = float(getattr(engine_result, "resource_grid_overlap_fraction", 0.0))
            interference_penalty_db = float(getattr(engine_result, "interference_penalty_db", 0.0))
            intra_cell_interference_penalty_db = float(
                getattr(engine_result, "intra_cell_interference_penalty_db", 0.0)
            )
            inter_cell_interference_dbm = float(
                getattr(engine_result, "inter_cell_interference_dbm", inter_cell_interference_dbm)
            )
            inter_cell_interference_penalty_db = float(
                getattr(
                    engine_result,
                    "inter_cell_interference_penalty_db",
                    inter_cell_interference_penalty_db,
                )
            )
            inter_cell_interference_overlap_fraction = float(
                getattr(
                    engine_result,
                    "inter_cell_interference_overlap_fraction",
                    inter_cell_interference_overlap_fraction,
                )
            )
            inter_cell_frequency_coupling_fraction = float(
                getattr(
                    engine_result,
                    "inter_cell_frequency_coupling_fraction",
                    inter_cell_frequency_coupling_fraction,
                )
            )
            inter_cell_interference_source_count = int(
                getattr(
                    engine_result,
                    "inter_cell_interference_source_count",
                    inter_cell_interference_source_count,
                )
            )
            phy_mode = str(getattr(engine_result, "phy_mode", phy_mode))
            waveform_backend = getattr(engine_result, "waveform_backend", waveform_backend)
            effective_data_re_count = int(getattr(engine_result, "effective_data_re_count", 0))
            dmrs_overhead_fraction = float(getattr(engine_result, "dmrs_overhead_fraction", 0.0))

        return PhyUeResult(
            ue_id=str(getattr(ue_input, "ue_id", "unknown_ue")),
            is_scheduled=is_scheduled,
            allocated_prbs=allocated_prbs,
            allocated_symbols=allocated_symbols,
            start_prb=start_prb,
            start_symbol=start_symbol,
            is_retx=is_retx,
            requested_bytes=requested_bytes,
            grant_bytes_budget=grant_bytes_budget,
            mcs_table_id=mcs_table_id,
            tbs_selection_method=tbs_selection_method,
            dmrs_pattern_id=dmrs_pattern_id,
            harq_process_id=(
                int(harq_process_id) if harq_process_id is not None else None
            ),
            harq_redundancy_version=(
                int(harq_redundancy_version) if harq_redundancy_version is not None else None
            ),
            harq_ndi=(int(harq_ndi) if harq_ndi is not None else None),
            harq_retransmission_round=int(harq_retransmission_round),
            cqi_validity_age_slots=int(cqi_validity_age_slots),
            ri_validity_age_slots=int(ri_validity_age_slots),
            pmi_validity_age_slots=int(pmi_validity_age_slots),
            channel_quality_confidence=(
                float(channel_quality_confidence)
                if channel_quality_confidence is not None
                else None
            ),
            link_adaptation_reason=(
                str(link_adaptation_reason)
                if link_adaptation_reason is not None
                else None
            ),
            scheduled_bytes=scheduled_bytes,
            delivered_bytes=delivered_bytes,
            ack=ack,
            sinr_db=sinr_db,
            effective_sinr_db=effective_sinr_db,
            pre_eq_sinr_db=pre_eq_sinr_db,
            post_eq_sinr_db=post_eq_sinr_db,
            bler=bler,
            ber=ber,
            mcs_index=mcs_index,
            modulation_order=modulation_order,
            code_rate=code_rate,
            num_layers=num_layers,
            tb_size_bytes=tb_size_bytes,
            transport_block_size_bytes=tb_size_bytes,
            throughput_mbps=throughput_mbps,
            queue_bytes=int(getattr(ue_input, "queue_bytes", 0)),
            cqi=cqi,
            rank=rank,
            pmi=pmi,
            ri=ri,
            beam_gain_db=beam_gain_db,
            channel_rank_limit=channel_rank_limit,
            precoder_match_score=precoder_match_score,
            spatial_mode=spatial_mode,
            spatial_group_id=(int(spatial_group_id) if spatial_group_id is not None else None),
            spatial_group_size=int(spatial_group_size),
            spatial_compatibility_score=float(spatial_compatibility_score),
            spatial_multiplexing_gain_db=float(spatial_multiplexing_gain_db),
            harq_retransmission_count=int(harq_retransmission_round),
            resource_grid_re_count=resource_grid_re_count,
            resource_grid_overlap_fraction=resource_grid_overlap_fraction,
            interference_penalty_db=interference_penalty_db,
            intra_cell_interference_penalty_db=intra_cell_interference_penalty_db,
            inter_cell_interference_dbm=inter_cell_interference_dbm,
            inter_cell_interference_penalty_db=inter_cell_interference_penalty_db,
            inter_cell_interference_overlap_fraction=inter_cell_interference_overlap_fraction,
            inter_cell_frequency_coupling_fraction=inter_cell_frequency_coupling_fraction,
            inter_cell_interference_source_count=inter_cell_interference_source_count,
            phy_mode=phy_mode,
            waveform_backend=(str(waveform_backend) if waveform_backend is not None else None),
            effective_data_re_count=effective_data_re_count,
            dmrs_overhead_fraction=dmrs_overhead_fraction,
        )

    def _build_phy_result(
        self,
        phy_input,
        ue_results: List[PhyUeResult],
    ) -> PhyResult:
        scheduled_results = [item for item in ue_results if getattr(item, "is_scheduled", False)]

        if scheduled_results:
            avg_sinr_db = sum(item.sinr_db for item in scheduled_results) / len(scheduled_results)
            avg_bler = sum(item.bler for item in scheduled_results) / len(scheduled_results)
            total_tb_size_bytes = sum(item.tb_size_bytes for item in scheduled_results)
        else:
            avg_sinr_db = 0.0
            avg_bler = 0.0
            total_tb_size_bytes = 0

        return PhyResult(
            slot_id=int(getattr(phy_input, "slot_id", -1)),
            cell_id=str(getattr(phy_input, "cell_id", "cell_0")),
            ue_results=ue_results,
            avg_sinr_db=avg_sinr_db,
            avg_bler=avg_bler,
            total_tb_size_bytes=total_tb_size_bytes,
        )


