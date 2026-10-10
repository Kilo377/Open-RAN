"""
Propagation sub-states for explicit large-scale and small-scale link modeling.
"""

from dataclasses import dataclass


@dataclass
class LargeScaleLinkState:
    """
    Stores slow-varying, geometry-driven propagation terms.
    """

    distance_2d_m: float = 0.0
    distance_3d_m: float = 0.0
    pathloss_db: float = 130.0
    shadowing_db: float = 6.0
    shadowing_sigma_db: float = 6.0
    los_state: str = "los"
    o2i_penetration_loss_db: float = 0.0


@dataclass
class SmallScaleLinkState:
    """
    Stores fast-varying or channel-profile-specific propagation terms.
    """

    doppler_hz: float = 30.0
    delay_spread_ns: float = 300.0
    k_factor_db: float = 0.0
    angle_spread_deg: float = 35.0
    spatial_correlation_level: float = 0.5
    channel_model_type: str = "tdl_c"
    fast_fading_seed: int = 0
    spatial_corr_seed: int = 0
