"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: link_adapter.py
Description: This file defines a simplified fake PHY adapter. In this
project stage, it does not call a real PHY engine. Instead, it consumes a
standardized PHY input object and returns a standardized PHY result object.
"""

from core.debug import debug_print
from models.phy_input import PhyInput
from models.phy_result import PhyResult, PhyUeResult
from phy.phy_base import PhyBase


class FakePhyAdapter(PhyBase):
    """
    Produces placeholder PHY results for the current slot.
    INPUT
    No explicit input during initialization.
    OUTPUT
    A fake PHY adapter object that returns PHY result objects.
    EXAMPLE OUTPUT
    phy = FakePhyAdapter()
    phy_result = phy.run_slot(phy_input)
    """

    def _estimate_tb_size_bytes(self, ue_input, *, phase: int, codebook_gain: float) -> int:
        resource_units = max(0, int(getattr(ue_input, "allocated_prbs", 0))) * max(0, int(getattr(ue_input, "allocated_symbols", 0)))
        if resource_units <= 0:
            return 0

        bytes_per_unit = 4.0 + 0.2 * float(phase) + 0.4 * float(codebook_gain) + 0.2 * float(getattr(ue_input, "priority", 0))
        raw_bytes = int(max(0.0, float(resource_units) * bytes_per_unit))
        requested_bytes = int(getattr(ue_input, "requested_bytes", getattr(ue_input, "queue_bytes", 0)))
        grant_bytes_budget = int(getattr(ue_input, "grant_bytes_budget", requested_bytes))
        effective_budget = grant_bytes_budget if grant_bytes_budget > 0 else requested_bytes
        return min(max(0, requested_bytes), max(0, effective_budget), raw_bytes)

    def run_slot(self, phy_input: PhyInput) -> PhyResult:
        """
        Executes one fake PHY slot update using the given PHY input.
        INPUT
        phy_input: The PHY input object for the current slot.
        OUTPUT
        One PhyResult object containing updated per-UE results.
        EXAMPLE OUTPUT
        PhyResult(
            slot_id=1,
            ue_results=[PhyUeResult(...), PhyUeResult(...)]
        )
        """
        base_power_gain = phy_input.tx_power_dbm - 30.0
        base_codebook_gain = float(phy_input.current_codebook_id)
        base_beam_gain = 0.5 * float(phy_input.current_beam_profile_id)

        ue_results = []

        slot_duration_ms = 1.0
        if phy_input.runtime_context is not None:
            slot_duration_ms = float(getattr(phy_input.runtime_context, "slot_duration_ms", 1.0))
        slot_duration_s = max(slot_duration_ms, 1e-6) / 1000.0

        for index, ue_input in enumerate(phy_input.ue_inputs):
            phase = (phy_input.slot_id + index) % 5
            allocated_prbs = int(getattr(ue_input, "allocated_prbs", 0))
            allocated_symbols = int(getattr(ue_input, "allocated_symbols", 0))
            resource_units = allocated_prbs * allocated_symbols

            updated_sinr_db = (
                8.0
                + index
                + phase
                + base_power_gain
                + base_codebook_gain
                + base_beam_gain
                + 0.02 * float(resource_units)
            )
            inter_cell_interference_penalty_db = float(
                getattr(ue_input, "inter_cell_interference_penalty_db", 0.0) or 0.0
            )
            updated_sinr_db -= inter_cell_interference_penalty_db

            if ue_input.is_scheduled and resource_units > 0:
                updated_bler = max(0.01, 0.22 - 0.02 * phase - 0.005 * base_codebook_gain - 0.001 * float(resource_units))
                tb_size_bytes = self._estimate_tb_size_bytes(
                    ue_input,
                    phase=phase,
                    codebook_gain=base_codebook_gain,
                )
                ack = True
                delivered_bytes = int(tb_size_bytes)
                updated_throughput_mbps = float(tb_size_bytes * 8.0) / 1e6 / max(slot_duration_s, 1e-9)
                updated_queue_bytes = max(0, int(getattr(ue_input, "queue_bytes", 0)) - int(tb_size_bytes))
            else:
                updated_bler = min(1.0, float(getattr(ue_input, "bler", 0.0)) + 0.01)
                updated_throughput_mbps = 0.0
                updated_queue_bytes = int(getattr(ue_input, "queue_bytes", 0))
                tb_size_bytes = 0
                delivered_bytes = 0
                ack = False

            ue_results.append(
                PhyUeResult(
                    ue_id=ue_input.ue_id,
                    is_scheduled=ue_input.is_scheduled,
                    allocated_prbs=allocated_prbs,
                    allocated_symbols=allocated_symbols,
                    start_prb=int(getattr(ue_input, "start_prb", 0)),
                    start_symbol=int(getattr(ue_input, "start_symbol", 0)),
                    is_retx=bool(getattr(ue_input, "is_retx", False)),
                    requested_bytes=int(getattr(ue_input, "requested_bytes", 0)),
                    grant_bytes_budget=int(getattr(ue_input, "grant_bytes_budget", 0)),
                    channel_quality_confidence=(
                        float(getattr(ue_input, "channel_quality_confidence", 1.0))
                        if getattr(ue_input, "channel_quality_confidence", None) is not None
                        else None
                    ),
                    link_adaptation_reason=getattr(ue_input, "link_adaptation_reason", None),
                    scheduled_bytes=tb_size_bytes,
                    delivered_bytes=delivered_bytes,
                    ack=ack,
                    sinr_db=updated_sinr_db,
                    bler=updated_bler,
                    throughput_mbps=updated_throughput_mbps,
                    queue_bytes=updated_queue_bytes,
                    tb_size_bytes=tb_size_bytes,
                    transport_block_size_bytes=tb_size_bytes,
                    resource_grid_re_count=int(max(0, allocated_prbs) * max(0, allocated_symbols) * 12),
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
            f"Fake PHY executed for slot {phy_input.slot_id}. "
            f"tx_power_dbm={phy_input.tx_power_dbm:.2f}, "
            f"codebook_id={phy_input.current_codebook_id}, "
            f"beam_profile_id={phy_input.current_beam_profile_id}, "
            f"scheduled_ues={phy_input.scheduled_ue_ids}."
        )

        scheduled_results = [item for item in ue_results if bool(getattr(item, "is_scheduled", False))]
        if scheduled_results:
            avg_sinr_db = sum(float(item.sinr_db) for item in scheduled_results) / float(len(scheduled_results))
            avg_bler = sum(float(item.bler) for item in scheduled_results) / float(len(scheduled_results))
            total_tb_size_bytes = sum(int(item.tb_size_bytes) for item in scheduled_results)
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


