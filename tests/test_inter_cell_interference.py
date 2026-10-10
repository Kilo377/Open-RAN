import tempfile
import unittest
from types import SimpleNamespace
from typing import Any

from core.simulator import Simulator
from models.scenario import ScenarioConfig
from phy.link_adapter import FakePhyAdapter
from ran.scheduler import SimpleScheduler


class TestInterCellInterference(unittest.TestCase):
    def _build_simulator(
        self,
        *,
        cell_0_frequency_hz: float,
        cell_1_frequency_hz: float,
    ) -> Simulator:
        config = ScenarioConfig(
            total_slots=1,
            slot_duration_ms=1.0,
            xapp_interval_slots=100,
            rapp_interval_slots=100,
            obs_delay_slots=0,
            config_actuation_delay_slots=0,
            command_actuation_delay_slots=0,
            measurement_queue_delay_slots=0,
            measurement_sinr_window_slots=4,
            measurement_sinr_noise_std_db=0.0,
            measurement_bler_window_tbs=4,
            measurement_throughput_granularity_slots=1,
        )
        controller_registry = SimpleNamespace(xapps=[], rapps=[])
        scheduler = SimpleScheduler(
            total_prbs_per_slot=24,
            total_symbols_per_slot=14,
            max_ues_per_slot=1,
            min_prbs_per_grant=6,
            min_symbols_per_grant=4,
        )
        phy_adapter = FakePhyAdapter()
        cell_config = {
            "cell_id": "cell_0",
            "tx_power_dbm": 30.0,
            "current_codebook_id": 1,
            "current_beam_profile_id": 1,
            "managed_cell_ids": ["cell_0", "cell_1"],
            "neighbor_cell_positions": {
                "cell_0": {"x_m": 0.0, "y_m": 0.0, "z_m": 25.0},
                "cell_1": {"x_m": 40.0, "y_m": 0.0, "z_m": 25.0},
            },
            "cell_radio_profiles": {
                "cell_0": {
                    "cell_type": "macro",
                    "carrier_frequency_hz": cell_0_frequency_hz,
                    "bandwidth_hz": 40e6,
                    "subcarrier_spacing_hz": 30e3,
                    "tx_power_dbm": 30.0,
                },
                "cell_1": {
                    "cell_type": "small_cell",
                    "carrier_frequency_hz": cell_1_frequency_hz,
                    "bandwidth_hz": 40e6,
                    "subcarrier_spacing_hz": 30e3,
                    "tx_power_dbm": 30.0,
                },
            },
            "handover_margin_db": 6.0,
            "handover_delay_slots": 2,
            "traffic_steering_preference": "balanced",
            "mobility_bias_db": 0.0,
        }
        ue_config = {
            "ues": [
                {
                    "ue_id": "ue_cell_0",
                    "queue_bytes": 800,
                    "throughput_mbps": 0.0,
                    "sinr_db": 8.0,
                    "radio_quality_db": 10.0,
                    "bler": 0.1,
                    "serving_beam_id": 0,
                    "slice_id": "embb",
                    "traffic_type": "video",
                    "latency_budget_ms": 50.0,
                    "priority": 1,
                    "serving_cell_id": "cell_0",
                    "x_m": 4.0,
                    "y_m": 0.0,
                    "z_m": 1.5,
                    "pathloss_db": 108.0,
                    "shadowing_db": 1.0,
                    "doppler_hz": 2.0,
                },
                {
                    "ue_id": "ue_cell_1",
                    "queue_bytes": 800,
                    "throughput_mbps": 0.0,
                    "sinr_db": 8.0,
                    "radio_quality_db": 10.0,
                    "bler": 0.1,
                    "serving_beam_id": 0,
                    "slice_id": "urllc",
                    "traffic_type": "control",
                    "latency_budget_ms": 10.0,
                    "priority": 3,
                    "serving_cell_id": "cell_1",
                    "x_m": 39.0,
                    "y_m": 0.0,
                    "z_m": 1.5,
                    "pathloss_db": 108.0,
                    "shadowing_db": 1.0,
                    "doppler_hz": 2.0,
                },
            ]
        }
        beamforming_config = {
            "antenna": {
                "gnb_num_tx_ant": 4,
                "default_ue_num_rx_ant": 2,
                "carrier_frequency_hz": cell_0_frequency_hz,
                "bandwidth_hz": 40e6,
                "subcarrier_spacing_hz": 30e3,
                "cyclic_prefix_type": "normal",
                "gnb_panel_layout": "2x2",
                "default_ue_panel_layout": "1x2",
                "polarization": "single",
            },
            "codebooks": {
                0: {"beam_gain_db": 0.0},
                1: {"beam_gain_db": 1.5},
                2: {"beam_gain_db": 3.0},
            },
            "beam_profiles": {
                0: {"profile_gain_db": 0.0},
                1: {"profile_gain_db": 1.0},
            },
        }
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        simulator = Simulator(
            config=config,
            controller_registry=controller_registry,
            scheduler=scheduler,
            phy_adapter=phy_adapter,
            cell_config=cell_config,
            ue_config=ue_config,
            beamforming_config=beamforming_config,
            output_dir=temp_dir.name,
        )
        cast_simulator = simulator
        cast_simulator.mobility_manager = SimpleNamespace(update=lambda state: None)  # type: ignore[assignment]
        for connection in getattr(cast_simulator.state, "connections", {}).values():
            setattr(connection, "connection_state", "connected")
            setattr(connection, "handover_pending_target_cell_id", None)
            setattr(connection, "handover_started_slot", None)
        return simulator

    def test_same_frequency_cells_incur_inter_cell_interference_penalty(self) -> None:
        simulator = self._build_simulator(
            cell_0_frequency_hz=3.5e9,
            cell_1_frequency_hz=3.5e9,
        )

        observation = simulator.step()
        ue_observations = {
            item.ue_id: item for item in observation.ue_observations
        }

        self.assertGreater(
            float(observation.inter_cell_interference_penalty_db_mean or 0.0),
            0.0,
        )
        self.assertGreater(
            float(ue_observations["ue_cell_0"].inter_cell_interference_penalty_db or 0.0),
            0.0,
        )
        self.assertGreater(
            float(ue_observations["ue_cell_1"].inter_cell_interference_penalty_db or 0.0),
            0.0,
        )
        self.assertEqual(
            int(ue_observations["ue_cell_0"].inter_cell_interference_source_count or 0),
            1,
        )

    def test_separated_frequency_cells_still_have_small_leakage_interference(self) -> None:
        simulator = self._build_simulator(
            cell_0_frequency_hz=3.5e9,
            cell_1_frequency_hz=28.0e9,
        )

        observation = simulator.step()
        ue_observations = {
            item.ue_id: item for item in observation.ue_observations
        }

        self.assertGreater(
            float(observation.inter_cell_interference_penalty_db_mean or 0.0),
            0.0,
        )
        self.assertLess(
            float(observation.inter_cell_interference_penalty_db_mean or 0.0),
            0.1,
        )
        self.assertGreater(
            float(ue_observations["ue_cell_0"].inter_cell_interference_penalty_db or 0.0),
            0.0,
        )
        self.assertGreater(
            float(ue_observations["ue_cell_0"].inter_cell_frequency_coupling_fraction or 0.0),
            0.0,
        )
        self.assertEqual(
            float(ue_observations["ue_cell_0"].inter_cell_interference_overlap_fraction or 0.0),
            0.0,
        )
        self.assertEqual(
            int(ue_observations["ue_cell_0"].inter_cell_interference_source_count or 0),
            1,
        )

    def test_same_frequency_interference_is_stronger_than_far_frequency_leakage(self) -> None:
        same_frequency_simulator = self._build_simulator(
            cell_0_frequency_hz=3.5e9,
            cell_1_frequency_hz=3.5e9,
        )
        far_frequency_simulator = self._build_simulator(
            cell_0_frequency_hz=3.5e9,
            cell_1_frequency_hz=28.0e9,
        )

        same_frequency_observation = same_frequency_simulator.step()
        far_frequency_observation = far_frequency_simulator.step()

        self.assertGreater(
            float(same_frequency_observation.inter_cell_interference_penalty_db_mean or 0.0),
            float(far_frequency_observation.inter_cell_interference_penalty_db_mean or 0.0),
        )


if __name__ == "__main__":
    unittest.main()
