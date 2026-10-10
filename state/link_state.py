"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: link_state.py
Description: This file defines the per-UE serving-link state used by the
simulator. The link state captures large-scale channel parameters and a
small set of PHY-facing indicators that should belong to the gNB-UE link
rather than the UE business state.
"""

from dataclasses import dataclass, field

from state.propagation_state import LargeScaleLinkState, SmallScaleLinkState


@dataclass
class LinkState:
    """
    Stores one explicit serving link between a UE and one cell.
    """

    ue_id: str
    serving_cell_id: str
    scenario_type: str = "umi_street_canyon"
    los_state: str = "los"
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 1.5
    distance_2d_m: float = 0.0
    distance_3d_m: float = 0.0
    velocity_mps: float = 0.0
    heading_deg: float = 0.0
    is_indoor: bool = False
    pathloss_db: float = 130.0
    shadowing_db: float = 6.0
    shadowing_sigma_db: float = 6.0
    o2i_penetration_loss_db: float = 0.0
    doppler_hz: float = 30.0
    delay_spread_ns: float = 300.0
    k_factor_db: float = 0.0
    angle_spread_deg: float = 35.0
    spatial_correlation_level: float = 0.5
    channel_model_type: str = "tdl_c"
    fast_fading_seed: int = 0
    spatial_corr_seed: int = 0
    large_scale: LargeScaleLinkState = field(default_factory=LargeScaleLinkState)
    small_scale: SmallScaleLinkState = field(default_factory=SmallScaleLinkState)
    last_pre_eq_sinr_db: float | None = None
    last_post_eq_sinr_db: float | None = None
    cqi: int = 0
    ri: int = 1
    pmi: int = 0

    def __post_init__(self) -> None:
        if self.large_scale != LargeScaleLinkState():
            self._apply_large_scale_view()
        if self.small_scale != SmallScaleLinkState():
            self._apply_small_scale_view()
        self.refresh_propagation_views()

    def refresh_propagation_views(self) -> None:
        """
        Synchronizes nested propagation views from the flat compatibility fields.
        """
        self.large_scale = LargeScaleLinkState(
            distance_2d_m=float(self.distance_2d_m),
            distance_3d_m=float(self.distance_3d_m),
            pathloss_db=float(self.pathloss_db),
            shadowing_db=float(self.shadowing_db),
            shadowing_sigma_db=float(self.shadowing_sigma_db),
            los_state=str(self.los_state),
            o2i_penetration_loss_db=float(self.o2i_penetration_loss_db),
        )
        self.small_scale = SmallScaleLinkState(
            doppler_hz=float(self.doppler_hz),
            delay_spread_ns=float(self.delay_spread_ns),
            k_factor_db=float(self.k_factor_db),
            angle_spread_deg=float(self.angle_spread_deg),
            spatial_correlation_level=float(self.spatial_correlation_level),
            channel_model_type=str(self.channel_model_type),
            fast_fading_seed=int(self.fast_fading_seed),
            spatial_corr_seed=int(self.spatial_corr_seed),
        )

    def _apply_large_scale_view(self) -> None:
        large_scale = getattr(self, "large_scale", None)
        if large_scale is not None:
            self.distance_2d_m = float(getattr(large_scale, "distance_2d_m", self.distance_2d_m))
            self.distance_3d_m = float(getattr(large_scale, "distance_3d_m", self.distance_3d_m))
            self.pathloss_db = float(getattr(large_scale, "pathloss_db", self.pathloss_db))
            self.shadowing_db = float(getattr(large_scale, "shadowing_db", self.shadowing_db))
            self.shadowing_sigma_db = float(
                getattr(large_scale, "shadowing_sigma_db", self.shadowing_sigma_db)
            )
            self.los_state = str(getattr(large_scale, "los_state", self.los_state))
            self.o2i_penetration_loss_db = float(
                getattr(large_scale, "o2i_penetration_loss_db", self.o2i_penetration_loss_db)
            )

    def _apply_small_scale_view(self) -> None:
        small_scale = getattr(self, "small_scale", None)
        if small_scale is not None:
            self.doppler_hz = float(getattr(small_scale, "doppler_hz", self.doppler_hz))
            self.delay_spread_ns = float(
                getattr(small_scale, "delay_spread_ns", self.delay_spread_ns)
            )
            self.k_factor_db = float(getattr(small_scale, "k_factor_db", self.k_factor_db))
            self.angle_spread_deg = float(
                getattr(small_scale, "angle_spread_deg", self.angle_spread_deg)
            )
            self.spatial_correlation_level = float(
                getattr(
                    small_scale,
                    "spatial_correlation_level",
                    self.spatial_correlation_level,
                )
            )
            self.channel_model_type = str(
                getattr(small_scale, "channel_model_type", self.channel_model_type)
            )
            self.fast_fading_seed = int(
                getattr(small_scale, "fast_fading_seed", self.fast_fading_seed)
            )
            self.spatial_corr_seed = int(
                getattr(small_scale, "spatial_corr_seed", self.spatial_corr_seed)
            )

    def apply_propagation_views(self) -> None:
        """
        Applies nested propagation views back to the flat compatibility fields.
        """
        self._apply_large_scale_view()
        self._apply_small_scale_view()
