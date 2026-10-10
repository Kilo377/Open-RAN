import unittest

from state.link_state import LinkState
from state.propagation_state import LargeScaleLinkState, SmallScaleLinkState


class TestLinkStateLayering(unittest.TestCase):
    def test_link_state_builds_large_and_small_scale_views_from_flat_fields(self) -> None:
        link = LinkState(
            ue_id="ue_0",
            serving_cell_id="cell_0",
            los_state="nlos",
            distance_2d_m=120.0,
            distance_3d_m=125.0,
            pathloss_db=133.0,
            shadowing_db=7.5,
            o2i_penetration_loss_db=15.0,
            doppler_hz=80.0,
            delay_spread_ns=500.0,
            k_factor_db=2.0,
            angle_spread_deg=22.0,
            spatial_correlation_level=0.7,
            channel_model_type="tdl_d",
            fast_fading_seed=101,
            spatial_corr_seed=202,
        )

        self.assertEqual(link.large_scale.los_state, "nlos")
        self.assertEqual(link.large_scale.distance_2d_m, 120.0)
        self.assertEqual(link.large_scale.pathloss_db, 133.0)
        self.assertEqual(link.large_scale.o2i_penetration_loss_db, 15.0)
        self.assertEqual(link.small_scale.doppler_hz, 80.0)
        self.assertEqual(link.small_scale.delay_spread_ns, 500.0)
        self.assertEqual(link.small_scale.channel_model_type, "tdl_d")
        self.assertEqual(link.small_scale.fast_fading_seed, 101)

    def test_link_state_can_apply_nested_propagation_views_back_to_flat_fields(self) -> None:
        link = LinkState(
            ue_id="ue_1",
            serving_cell_id="cell_0",
            large_scale=LargeScaleLinkState(
                distance_2d_m=55.0,
                distance_3d_m=58.0,
                pathloss_db=118.0,
                shadowing_db=3.0,
                los_state="los",
                o2i_penetration_loss_db=0.0,
            ),
            small_scale=SmallScaleLinkState(
                doppler_hz=12.0,
                delay_spread_ns=150.0,
                k_factor_db=9.0,
                angle_spread_deg=40.0,
                spatial_correlation_level=0.2,
                channel_model_type="tdl_c",
                fast_fading_seed=11,
                spatial_corr_seed=22,
            ),
        )

        link.apply_propagation_views()

        self.assertEqual(link.pathloss_db, 118.0)
        self.assertEqual(link.shadowing_db, 3.0)
        self.assertEqual(link.distance_2d_m, 55.0)
        self.assertEqual(link.doppler_hz, 12.0)
        self.assertEqual(link.delay_spread_ns, 150.0)
        self.assertEqual(link.fast_fading_seed, 11)


if __name__ == "__main__":
    unittest.main()
