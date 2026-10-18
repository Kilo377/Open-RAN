"""Select the Jetson CPU ray-tracing backend before Sionna imports Mitsuba."""
import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")
