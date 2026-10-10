import unittest
from typing import Any, cast

from core.simulator import Simulator
from state.antenna_state import AntennaState
from state.beam_state import BeamState
from state.beamforming_state import BeamformingState
from state.cell_state import CellState
from state.ue_state import UeState


class TestBeamformingStateEntities(unittest.TestCase):
    def test_sync_beam_state_updates_per_ue_entities_from_cell(self) -> None:
        simulator = cast(Any, Simulator.__new__(Simulator))
        simulator.state = type(
            "State",
            (),
            {
                "cell": CellState(
                    cell_id="cell_0",
                    tx_power_dbm=30.0,
                    current_codebook_id=2,
                    current_beam_profile_id=1,
                ),
                "beam": BeamState(
                    codebook_id=0,
                    beam_profile_id=0,
                    selected_beam_id=0,
                    beam_gain_db=0.0,
                    codebook_map={
                        0: {"base_gain_db": 0.0, "beams": {0: 0.0, 1: 0.5}},
                        2: {"base_gain_db": 1.5, "beams": {0: 0.0, 1: 1.0}},
                    },
                    beam_profile_map={
                        0: {"profile_gain_db": 0.0},
                        1: {"profile_gain_db": 0.8},
                    },
                ),
                "antenna": AntennaState(
                    gnb_num_tx_ant=8,
                    ue_num_rx_ant_by_ue={"ue_0": 2},
                    carrier_frequency_hz=3.5e9,
                    subcarrier_spacing_hz=30e3,
                    gnb_panel_layout="4x2",
                    ue_panel_layout_by_ue={"ue_0": "1x2"},
                    polarization="dual",
                ),
                "ues": [
                    UeState(
                        ue_id="ue_0",
                        queue_bytes=100,
                        throughput_mbps=0.0,
                        sinr_db=10.0,
                        radio_quality_db=10.0,
                        bler=0.1,
                        serving_beam_id=1,
                        slice_id="embb",
                        traffic_type="video",
                        latency_budget_ms=50.0,
                        priority=1,
                        rank=1,
                        pmi=0,
                        ri=1,
                    )
                ],
                "beamforming": {
                    "ue_0": BeamformingState(
                        ue_id="ue_0",
                        codebook_id=0,
                        beam_profile_id=0,
                        serving_beam_id=1,
                        precoder_id=0,
                        combiner_id=3,
                        rank=2,
                        layer_count=2,
                        beam_gain_db=0.0,
                        beam_direction_deg=0.0,
                        precoding_matrix_label="cb0-r2-b1",
                    )
                },
            },
        )()

        simulator._sync_beam_state_from_cell()

        state = cast(Any, simulator.state)
        beamforming = state.beamforming["ue_0"]
        self.assertEqual(state.beam.codebook_id, 2)
        self.assertEqual(state.beam.beam_profile_id, 1)
        self.assertEqual(beamforming.codebook_id, 2)
        self.assertEqual(beamforming.beam_profile_id, 1)
        self.assertEqual(beamforming.precoder_id, 2)
        self.assertEqual(beamforming.serving_beam_id, 1)
        self.assertEqual(beamforming.rank, 2)
        self.assertEqual(beamforming.layer_count, 2)
        self.assertEqual(beamforming.combiner_id, 3)
        self.assertEqual(beamforming.beam_gain_db, 3.3)
        self.assertEqual(len(beamforming.precoder_matrix), 8)
        self.assertEqual(len(beamforming.precoder_matrix[0]), 2)
        self.assertEqual(len(beamforming.combiner_vector), 2)
        self.assertEqual(beamforming.precoder_match_score, 1.0)


if __name__ == "__main__":
    unittest.main()
