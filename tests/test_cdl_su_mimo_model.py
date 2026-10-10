import unittest

from phy.cdl_su_mimo_model import CdlSuMimoModel
from phy.sionna_engine import SionnaEngine


class TestCdlSuMimoModel(unittest.TestCase):
    def test_engine_clamps_low_coderate_for_sionna_ldpc_support(self) -> None:
        engine = SionnaEngine({"min_supported_sionna_coderate": 0.34})

        self.assertAlmostEqual(
            engine._normalize_target_coderate_for_sionna(0.30),
            0.34,
            places=6,
        )
        self.assertAlmostEqual(
            engine._normalize_target_coderate_for_sionna(0.50),
            0.50,
            places=6,
        )

    def test_waveform_context_accounts_for_dmrs_overhead(self) -> None:
        model = CdlSuMimoModel({"waveform_backend": "cdl_test_backend"})
        context = model.build_waveform_context(
            engine_input={
                "layer_count": 2,
                "dmrs_pattern_id": "dense",
                "subcarrier_spacing_hz": 30e3,
                "cyclic_prefix_type": "normal",
                "delay_spread_ns": 180.0,
                "k_factor_db": 6.0,
                "angle_spread_deg": 35.0,
                "spatial_correlation_level": 0.25,
                "los_state": "los",
                "channel_profile": "tdl_d",
            },
            resource_grid_view={"resource_elements": 1000},
            pre_eq_sinr_db=10.0,
            mimo_state={"precoder_match_score": 0.9},
        )

        self.assertEqual(context.phy_mode, "sionna_cdl_su_mimo")
        self.assertEqual(context.waveform_backend, "cdl_test_backend")
        self.assertGreater(context.dmrs_overhead_fraction, 0.1)
        self.assertLess(context.effective_data_resource_elements, 1000)

    def test_waveform_context_penalizes_high_delay_spread(self) -> None:
        model = CdlSuMimoModel({})
        low_delay = model.build_waveform_context(
            engine_input={
                "layer_count": 1,
                "dmrs_pattern_id": None,
                "subcarrier_spacing_hz": 30e3,
                "cyclic_prefix_type": "normal",
                "delay_spread_ns": 80.0,
                "k_factor_db": 0.0,
                "angle_spread_deg": 20.0,
                "spatial_correlation_level": 0.2,
                "los_state": "nlos",
                "channel_profile": "tdl_c",
            },
            resource_grid_view={"resource_elements": 600},
            pre_eq_sinr_db=8.0,
            mimo_state={"precoder_match_score": 0.8},
        )
        high_delay = model.build_waveform_context(
            engine_input={
                "layer_count": 1,
                "dmrs_pattern_id": None,
                "subcarrier_spacing_hz": 30e3,
                "cyclic_prefix_type": "normal",
                "delay_spread_ns": 1800.0,
                "k_factor_db": 0.0,
                "angle_spread_deg": 20.0,
                "spatial_correlation_level": 0.2,
                "los_state": "nlos",
                "channel_profile": "tdl_c",
            },
            resource_grid_view={"resource_elements": 600},
            pre_eq_sinr_db=8.0,
            mimo_state={"precoder_match_score": 0.8},
        )

        self.assertGreater(low_delay.sinr_adjustment_db, high_delay.sinr_adjustment_db)
        self.assertGreater(high_delay.delay_selectivity_penalty_db, low_delay.delay_selectivity_penalty_db)

    def test_engine_normalizes_cdl_phy_mode_aliases(self) -> None:
        engine = SionnaEngine({"phy_mode": "cdl_su_mimo"})
        self.assertEqual(engine.phy_mode, "sionna_cdl_su_mimo")

    def test_engine_clips_sinr_into_supported_range(self) -> None:
        engine = SionnaEngine(
            {
                "min_valid_sinr_db": -35.0,
                "max_valid_sinr_db": 45.0,
            }
        )

        self.assertEqual(engine._clip_sinr_db(-80.0), -35.0)
        self.assertEqual(engine._clip_sinr_db(90.0), 45.0)
        self.assertEqual(engine._clip_sinr_db(12.5), 12.5)


if __name__ == "__main__":
    unittest.main()
