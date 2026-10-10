import unittest

from models.intent_orchestration import GoalKpiTarget, StructuredIntentGoal
from models.observation import Observation
from ric.non_rt.intent_orchestration_helpers import available_xapp_catalog_for_llm
from ric.non_rt.xapp_selection_rapp import XAppSelectionRApp


class _StubLlmClient:
    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = []
        self.enabled = True
        self.model_label = "stub-llm"
        self.generated_by_tag = "stub_llm"

    def generate_json_or_raise(self, *, system_prompt, user_prompt):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        if len(self._payloads) == 0:
            raise AssertionError("No stub payload available for generate_json_or_raise().")
        return self._payloads.pop(0)


class TestXAppSelectionRApp(unittest.TestCase):
    def _make_goal(self, intent_text: str = "boost throughput while protecting urllc") -> StructuredIntentGoal:
        return StructuredIntentGoal(
            goal_id="goal:test_selection",
            intent_id="intent:test_selection",
            intent_text=intent_text,
            summary="Increase throughput while keeping reliability stable.",
            risk_posture="balanced",
            target_entities=["cluster"],
            kpi_targets=[
                GoalKpiTarget(
                    kpi_name="total_throughput_mbps",
                    objective="maximize",
                    target_value=40.0,
                    weight=1.0,
                ),
                GoalKpiTarget(
                    kpi_name="avg_bler",
                    objective="minimize",
                    target_value=0.15,
                    weight=0.6,
                ),
            ],
        )

    def _make_observation(self) -> Observation:
        return Observation(
            slot_id=12,
            avg_sinr_db=18.5,
            avg_bler=0.11,
            total_queue_bytes=24000,
            total_throughput_mbps=22.5,
            cell_id="cell_0",
            current_codebook_id=0,
            current_beam_profile_id=0,
            tx_power_dbm=30.0,
            scheduler_mode="round_robin",
        )

    def test_available_xapp_catalog_includes_variant_metadata(self) -> None:
        catalog = {
            str(item.get("xapp_name", "")): item
            for item in available_xapp_catalog_for_llm()
            if str(item.get("xapp_name", "")) != ""
        }

        variant_spec = catalog["traffic_steering_xapp_quality_first"]
        base_spec = catalog["traffic_steering_xapp"]

        self.assertEqual(variant_spec["xapp_category"], "traffic_steering_xapp")
        self.assertTrue(bool(variant_spec["is_variant"]))
        self.assertEqual(variant_spec["variant_params"]["edge_sinr_threshold_db"], 9.0)
        self.assertFalse(bool(base_spec["is_variant"]))
        self.assertEqual(base_spec["variant_params"], {})

    def test_llm_selection_uses_exact_variant_names_and_full_catalog_prompt(self) -> None:
        app = XAppSelectionRApp()
        app.client = _StubLlmClient(
            [
                {
                    "selected_xapps": [
                        "massive_mimo_beamformer_xapp_capacity_push",
                        "traffic_steering_xapp_quality_first",
                        "urllc_guard_xapp_balanced",
                    ],
                    "orchestration_mode": "parallel_with_guardrail",
                    "rationale_by_xapp": {
                        "massive_mimo_beamformer_xapp_capacity_push": "Pushes spectral efficiency for eMBB.",
                    },
                    "confidence_by_xapp": {
                        "traffic_steering_xapp_quality_first": 0.72,
                    },
                }
            ]
        )

        plan = app._try_llm_selection(
            goal=self._make_goal(),
            observation=self._make_observation(),
            reflection=None,
        )

        self.assertListEqual(
            plan.selected_xapps,
            [
                "massive_mimo_beamformer_xapp_capacity_push",
                "traffic_steering_xapp_quality_first",
                "urllc_guard_xapp_balanced",
            ],
        )
        self.assertEqual(plan.candidate_scores[1].xapp_name, "traffic_steering_xapp_quality_first")
        self.assertAlmostEqual(plan.candidate_scores[1].confidence, 0.72)
        self.assertGreater(len(plan.hard_safety_envelope), 0)
        self.assertIsNotNone(plan.exploration_health_gate)
        self.assertGreaterEqual(len(plan.forbidden_combinations), 1)
        self.assertIn('"available_xapps"', app.client.calls[0]["user_prompt"])
        self.assertIn("traffic_steering_xapp_quality_first", app.client.calls[0]["user_prompt"])
        self.assertIn("variant_params", app.client.calls[0]["user_prompt"])

    def test_legacy_category_selection_still_resolves_to_variants(self) -> None:
        app = XAppSelectionRApp()
        app.client = _StubLlmClient(
            [
                {
                    "selected_categories": [
                        "massive_mimo_beamformer_xapp",
                        "urllc_guard_xapp",
                    ],
                    "orchestration_mode": "parallel_with_guardrail",
                    "rationale_by_category": {
                        "massive_mimo_beamformer_xapp": "Capacity push helps throughput.",
                        "urllc_guard_xapp": "Guard reliability.",
                    },
                    "confidence_by_category": {
                        "massive_mimo_beamformer_xapp": 0.8,
                        "urllc_guard_xapp": 0.75,
                    },
                }
            ]
        )

        plan = app._try_llm_selection(
            goal=self._make_goal(),
            observation=self._make_observation(),
            reflection=None,
        )

        self.assertListEqual(
            plan.selected_xapps,
            [
                "massive_mimo_beamformer_xapp_capacity_push",
                "urllc_guard_xapp_balanced",
            ],
        )


if __name__ == "__main__":
    unittest.main()
