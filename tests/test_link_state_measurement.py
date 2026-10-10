import unittest
from types import SimpleNamespace

from core.measurement_updater import MeasurementUpdater
from state.link_state import LinkState


class TestLinkStateMeasurement(unittest.TestCase):
    def test_measurement_uses_per_ue_link_pathloss(self) -> None:
        updater = MeasurementUpdater(
            rng_seed=1,
            slot_duration_ms=1.0,
            queue_delay_slots=1,
            sinr_window_slots=4,
            sinr_noise_std_db=0.0,
            bler_window_tbs=8,
            throughput_granularity_slots=4,
        )

        state = SimpleNamespace(
            slot_id=3,
            cell=SimpleNamespace(
                tx_power_dbm=30.0,
                current_codebook_id=0,
                current_beam_profile_id=0,
            ),
            channel=SimpleNamespace(
                pathloss_db=120.0,
                shadowing_db=6.0,
                doppler_hz=30.0,
            ),
            scheduler=SimpleNamespace(scheduled_ue_ids=[]),
            ues=[
                SimpleNamespace(
                    ue_id="ue_strong",
                    queue_bytes=100,
                    radio_quality_db=10.0,
                    ack=False,
                    transport_block_size_bytes=0,
                ),
                SimpleNamespace(
                    ue_id="ue_weak",
                    queue_bytes=100,
                    radio_quality_db=10.0,
                    ack=False,
                    transport_block_size_bytes=0,
                ),
            ],
            links={
                "ue_strong": LinkState(
                    ue_id="ue_strong",
                    serving_cell_id="cell_0",
                    pathloss_db=100.0,
                    shadowing_db=0.0,
                    doppler_hz=10.0,
                    fast_fading_seed=20,
                ),
                "ue_weak": LinkState(
                    ue_id="ue_weak",
                    serving_cell_id="cell_0",
                    pathloss_db=130.0,
                    shadowing_db=8.0,
                    doppler_hz=10.0,
                    fast_fading_seed=20,
                ),
            },
            measurement=None,
        )

        updater.update(state)

        strong_sinr = state.measurement.ue_measurements["ue_strong"].sinr_db
        weak_sinr = state.measurement.ue_measurements["ue_weak"].sinr_db
        self.assertGreater(strong_sinr, weak_sinr)


if __name__ == "__main__":
    unittest.main()
