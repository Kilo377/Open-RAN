"""Evidence attribution helpers for outcome-grounded orchestration memory."""

from __future__ import annotations

from models.intent_orchestration import OrchestrationMemoryCase
from ric.near_rt.xapp_variant_catalog import canonical_xapp_category


_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "mobility_predictor_xapp": ("mobility", "handover", "ping-pong", "ping pong"),
    "traffic_steering_xapp": ("steering", "load steering", "cio", "traffic"),
    "power_saving_xapp": ("power_saving", "power saving", "energy", "sleep"),
    "cell_sleep_xapp": ("cell_sleep", "cell sleep", "sleep", "energy"),
    "spectrum_optimizer_xapp": ("spectrum", "interference", "prb", "packing"),
    "latency_scheduler_xapp": ("scheduler", "latency", "queue", "backlog"),
    "massive_mimo_beamformer_xapp": ("beam", "mimo", "sinr", "coverage"),
    "ran_slicing_manager_xapp": ("slice", "slicing", "tenant", "sla"),
    "urllc_guard_xapp": ("urllc", "guard", "reliability", "bler", "harq"),
}


def memory_case_text(memory_case: OrchestrationMemoryCase) -> str:
    """Return compact lowercase evidence text for one memory case."""

    reflection = getattr(memory_case, "reflection", None)
    reflection_parts = [
        str(getattr(reflection, "outcome_summary", "") or ""),
        str(getattr(reflection, "self_critique", "") or ""),
        str(getattr(reflection, "improvement_hypothesis", "") or ""),
        " ".join(str(item) for item in list(getattr(reflection, "recommended_changes", []) or [])),
    ]
    return " ".join(
        [
            str(getattr(memory_case, "failure_reason", "") or ""),
            str(getattr(memory_case, "summary", "") or ""),
            *reflection_parts,
        ]
    ).lower()


def xapp_categories_from_names(selected_xapps: list[str]) -> list[str]:
    categories: list[str] = []
    for xapp_name in list(selected_xapps or []):
        category_name = canonical_xapp_category(str(xapp_name).strip())
        if category_name != "" and category_name not in categories:
            categories.append(category_name)
    return categories


def protected_failure_categories(memory_case: OrchestrationMemoryCase) -> set[str]:
    """Categories explicitly recommended to keep despite a failed outcome."""

    text = memory_case_text(memory_case)
    protected: set[str] = set()
    if not any(token in text for token in ("keep", "retain", "add", "preserve")):
        return protected
    if "beam or spectrum" in text:
        protected.update({"massive_mimo_beamformer_xapp", "spectrum_optimizer_xapp"})
    if "scheduler or steering" in text:
        protected.update({"latency_scheduler_xapp", "traffic_steering_xapp"})
    for category_name, keywords in _CATEGORY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            protected.add(category_name)
    return protected


def attributed_failure_categories(memory_case: OrchestrationMemoryCase) -> set[str]:
    """Infer which categories should actually be avoided from failure evidence.

    A failed plan is not treated as evidence against every xApp it used. Only
    categories tied to an explicit avoid/conflict/risk cue are attributed.
    """

    selected_categories = set(
        xapp_categories_from_names(list(getattr(memory_case, "selected_xapps", []) or []))
    )
    if len(selected_categories) == 0:
        return set()

    text = memory_case_text(memory_case)
    avoid: set[str] = set()
    avoid_cues = ("avoid", "replace", "remove", "conflict", "unsafe", "forbidden", "mutex")
    if any(cue in text for cue in avoid_cues):
        for category_name, keywords in _CATEGORY_KEYWORDS.items():
            if category_name in selected_categories and any(keyword in text for keyword in keywords):
                avoid.add(category_name)

    if any(token in text for token in ("ping-pong", "ping pong", "handover")) and any(
        token in text for token in ("aggressive", "steering", "cio", "oscillat")
    ):
        avoid.add("traffic_steering_xapp")
    if "low hysteresis" in text or "handover margin" in text:
        avoid.add("mobility_predictor_xapp")
    if any(token in text for token in ("latency", "queue", "urllc")) and any(
        token in text for token in ("sleep", "energy")
    ):
        avoid.update({"power_saving_xapp", "cell_sleep_xapp"})
    if "throughput-only" in text and "unsafe" in text:
        avoid.update({"traffic_steering_xapp", "spectrum_optimizer_xapp"})

    return set(category for category in avoid if category in selected_categories) - protected_failure_categories(
        memory_case
    )


def attributed_failure_category_set(memory_cases: list[OrchestrationMemoryCase] | None) -> set[str]:
    categories: set[str] = set()
    for memory_case in list(memory_cases or []):
        categories.update(attributed_failure_categories(memory_case))
    return categories


def memory_case_outcome_quality(memory_case: OrchestrationMemoryCase) -> float:
    """Score how useful a successful case is beyond meeting a coarse intent threshold."""

    throughput_mbps = float(getattr(memory_case, "throughput_mbps", 0.0) or 0.0)
    avg_bler = float(getattr(memory_case, "avg_bler", 0.0) or 0.0)
    queue_bytes = float(getattr(memory_case, "queue_bytes", 0.0) or 0.0)
    throughput_quality = max(0.0, min(1.0, throughput_mbps / 10.0))
    bler_quality = max(0.0, min(1.0, 1.0 - (avg_bler / 0.18)))
    queue_quality = max(0.0, min(1.0, 1.0 - (queue_bytes / 3_500_000.0)))
    return float(
        max(
            0.0,
            min(
                1.0,
                (0.55 * throughput_quality)
                + (0.35 * bler_quality)
                + (0.10 * queue_quality),
            ),
        )
    )
