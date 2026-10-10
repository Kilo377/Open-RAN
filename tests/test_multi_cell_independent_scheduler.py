import tempfile
import unittest
from types import SimpleNamespace

from core.simulator import Simulator
from models.scenario import ScenarioConfig
from phy.link_adapter import FakePhyAdapter
from ran.scheduler import SimpleScheduler


class TestMultiCellIndependentScheduler(unittest.TestCase):
    def test_multi_cell_runtime_schedules_each_serving_cell_independently(self) -> None:
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
            min_prbs_per_grant=4,
            min_symbols_per_grant=4,
        )
        phy_adapter = FakePhyAdapter()
        cell_config = {
            "cell_id": "cell_0",
            "tx_power_dbm": 30.0,
            "current_codebook_id": 0,
            "current_beam_profile_id": 0,
            "managed_cell_ids": ["cell_0", "cell_1"],
            "neighbor_cell_positions": {
                "cell_0": {"x_m": 0.0, "y_m": 0.0, "z_m": 25.0},
                "cell_1": {"x_m": 120.0, "y_m": 0.0, "z_m": 10.0},
            },
            "cell_radio_profiles": {
                "cell_0": {
                    "cell_type": "macro",
                    "carrier_frequency_hz": 3.5e9,
                    "bandwidth_hz": 40e6,
                    "subcarrier_spacing_hz": 30e3,
                    "tx_power_dbm": 30.0,
                },
                "cell_1": {
                    "cell_type": "mmwave_hotspot",
                    "carrier_frequency_hz": 28.0e9,
                    "bandwidth_hz": 100e6,
                    "subcarrier_spacing_hz": 120e3,
                    "tx_power_dbm": 28.0,
                },
            },
            "handover_margin_db": 3.0,
            "handover_delay_slots": 2,
        }
        ue_config = {
            "ues": [
                {
                    "ue_id": "ue_macro",
                    "queue_bytes": 600,
                    "throughput_mbps": 0.0,
                    "sinr_db": 8.0,
                    "radio_quality_db": 9.0,
                    "bler": 0.1,
                    "serving_beam_id": 0,
                    "slice_id": "embb",
                    "traffic_type": "video",
                    "latency_budget_ms": 50.0,
                    "priority": 1,
                    "serving_cell_id": "cell_0",
                    "x_m": 20.0,
                    "y_m": 0.0,
                    "z_m": 1.5,
                    "pathloss_db": 110.0,
                    "shadowing_db": 3.0,
                    "doppler_hz": 3.0,
                },
                {
                    "ue_id": "ue_mmwave",
                    "queue_bytes": 600,
                    "throughput_mbps": 0.0,
                    "sinr_db": 10.0,
                    "radio_quality_db": 12.0,
                    "bler": 0.05,
                    "serving_beam_id": 1,
                    "slice_id": "urllc",
                    "traffic_type": "control",
                    "latency_budget_ms": 10.0,
                    "priority": 3,
                    "serving_cell_id": "cell_1",
                    "x_m": 118.0,
                    "y_m": 2.0,
                    "z_m": 1.5,
                    "pathloss_db": 92.0,
                    "shadowing_db": 2.0,
                    "doppler_hz": 3.0,
                },
            ]
        }
        beamforming_config = {
            "antenna": {
                "gnb_num_tx_ant": 4,
                "default_ue_num_rx_ant": 2,
                "carrier_frequency_hz": 3.5e9,
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

        with tempfile.TemporaryDirectory() as output_dir:
            simulator = Simulator(
                config=config,
                controller_registry=controller_registry,
                scheduler=scheduler,
                phy_adapter=phy_adapter,
                cell_config=cell_config,
                ue_config=ue_config,
                beamforming_config=beamforming_config,
                output_dir=output_dir,
            )

            observation = simulator.step()

        self.assertEqual(set(getattr(simulator.state, "cluster_scheduled_ue_ids", [])), {"ue_macro", "ue_mmwave"})
        self.assertEqual(
            set(simulator.state.managed_cells["cell_0"].scheduler.scheduled_ue_ids),
            {"ue_macro"},
        )
        self.assertEqual(
            set(simulator.state.managed_cells["cell_1"].scheduler.scheduled_ue_ids),
            {"ue_mmwave"},
        )
        self.assertEqual(set(observation.scheduled_ue_ids), {"ue_macro", "ue_mmwave"})
        self.assertEqual(observation.num_scheduled_ues, 2)

        cell_observations = {
            cell_observation.cell_id: cell_observation
            for cell_observation in observation.cell_observations
        }
        self.assertEqual(cell_observations["cell_0"].scheduled_ue_ids, ["ue_macro"])
        self.assertEqual(cell_observations["cell_1"].scheduled_ue_ids, ["ue_mmwave"])
        self.assertEqual(cell_observations["cell_0"].scheduled_ue_count, 1)
        self.assertEqual(cell_observations["cell_1"].scheduled_ue_count, 1)

        latest_metrics = simulator.slot_records[-1]["true_metrics"]
        self.assertEqual(int(latest_metrics["scheduled_ue_count"]), 2)
        self.assertEqual(int(latest_metrics["managed_cell_count"]), 2)
        self.assertGreater(int(getattr(simulator.state.last_phy_result, "total_tb_size_bytes", 0)), 0)


if __name__ == "__main__":
    unittest.main()
