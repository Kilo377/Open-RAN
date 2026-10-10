import unittest

from phy.sionna_engine import SionnaEngine


class TestSionnaFrequencyNumerologyEffects(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SionnaEngine(
            {
                "bandwidth_hz": 20e6,
                "noise_power_mode": "thermal",
                "noise_figure_db": 7.0,
                "interference_margin_db": 0.0,
            }
        )

    def test_noise_power_increases_with_bandwidth(self) -> None:
        narrow_bw_noise = self.engine._compute_noise_power_dbm(bandwidth_hz=20e6)
        wide_bw_noise = self.engine._compute_noise_power_dbm(bandwidth_hz=80e6)

        self.assertGreater(wide_bw_noise, narrow_bw_noise)

    def test_higher_carrier_frequency_reduces_effective_sinr(self) -> None:
        base_engine_input = {
            "slot_id": 1,
            "ue_id": "ue_0",
            "tx_power_dbm": 30.0,
            "radio_quality_db": 8.0,
            "codebook_id": 0,
            "beam_profile_id": 0,
            "precoder_id": 0,
            "layer_count": 1,
            "rank_indicator": 1,
            "num_tx_ant": 4,
            "num_rx_ant": 2,
            "beamforming_gain_db": 0.0,
            "fast_fading_seed": 17,
            "doppler_hz": 10.0,
            "pathloss_db": 105.0,
            "shadowing_db": 3.0,
            "bandwidth_hz": 20e6,
        }

        sub6_sinr_db = self.engine._compute_link_sinr_db(
            {
                **base_engine_input,
                "carrier_frequency_hz": 3.5e9,
            }
        )
        mmwave_like_sinr_db = self.engine._compute_link_sinr_db(
            {
                **base_engine_input,
                "carrier_frequency_hz": 28e9,
            }
        )

        self.assertLess(mmwave_like_sinr_db, sub6_sinr_db)


if __name__ == "__main__":
    unittest.main()
