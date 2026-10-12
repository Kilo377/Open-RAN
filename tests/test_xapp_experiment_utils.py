import unittest
from pathlib import Path

from experiments.xapp_experiment_utils import (
    filter_ric_config_xapps,
    load_xapp_experiment_configs,
)
from scenarios.scenario_generator import ScenarioGenerator
from scenarios.scenario_registry import ScenarioRegistry


class TestXAppExperimentUtils(unittest.TestCase):
    def test_filter_ric_config_xapps_keeps_only_selected_names(self) -> None:
        ric_config = {
            "xapps": [
                {"name": "xapp_a", "type": "beam_xapp"},
                {"name": "xapp_b", "type": "traffic_steering_xapp"},
            ],
            "rapps": [{"name": "rapp_a", "type": "codebook_rapp"}],
        }

        filtered = filter_ric_config_xapps(ric_config, enabled_xapp_names={"xapp_b"})

        self.assertEqual([item["name"] for item in filtered["xapps"]], ["xapp_b"])
        self.assertEqual([item["name"] for item in filtered["rapps"]], ["rapp_a"])

    def test_stress_config_bundle_loads_and_generates(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        configs, _ = load_xapp_experiment_configs(
            project_root,
            ric_filename="ric_sensitivity.yaml",
            scenario_filename="scenario_sensitivity_stress.yaml",
            cell_filename="cell_sensitivity_stress.yaml",
            ue_filename="ue_sensitivity_stress.yaml",
            phy_filename="phy_sensitivity_stress.yaml",
        )

        template = ScenarioRegistry().resolve(configs.get("scenario"))
        self.assertEqual(template.name, "heterogeneous_macro_mmwave_stress")

        bundle = ScenarioGenerator().generate(
            template=template,
            scenario_config=configs.get("scenario"),
            sim_config=configs["sim"],
            cell_config=configs["cell"],
            ue_config=configs["ue"],
            beamforming_config=configs["beamforming"],
        )

        self.assertEqual(bundle.scenario_name, "heterogeneous_macro_mmwave_stress")
        self.assertEqual(len(bundle.ue_config["ues"]), 8)
        self.assertEqual(bundle.sim_config["total_slots"], 60)
        self.assertEqual(bundle.cell_config["handover_cost_db"], 3.0)


if __name__ == "__main__":
    unittest.main()
