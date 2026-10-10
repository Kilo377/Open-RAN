"""
Helpers for reading cluster-level runtime views while preserving the
legacy single-active-cell execution path.
"""

from __future__ import annotations

from typing import Any


def get_managed_cells(state: Any) -> dict[str, Any]:
    managed_cells = getattr(state, "managed_cells", {})
    if isinstance(managed_cells, dict):
        return managed_cells
    return {}


def is_multi_cell_runtime(state: Any) -> bool:
    managed_cells = get_managed_cells(state)
    return str(getattr(state, "runtime_mode", "single_active_cell")) != "single_active_cell" and len(
        managed_cells
    ) > 1


def get_cluster_scheduled_ue_ids(state: Any) -> list[str]:
    explicit = getattr(state, "cluster_scheduled_ue_ids", None)
    if isinstance(explicit, list):
        return [str(item) for item in explicit]

    if is_multi_cell_runtime(state):
        aggregated_ids: list[str] = []
        seen_ids: set[str] = set()
        for runtime_state in get_managed_cells(state).values():
            runtime_scheduler = getattr(runtime_state, "scheduler", None)
            runtime_ids = list(
                getattr(runtime_state, "scheduled_ue_ids", [])
                or getattr(runtime_scheduler, "scheduled_ue_ids", [])
                or []
            )
            for ue_id in runtime_ids:
                normalized_ue_id = str(ue_id)
                if normalized_ue_id in seen_ids:
                    continue
                seen_ids.add(normalized_ue_id)
                aggregated_ids.append(normalized_ue_id)
        return aggregated_ids

    scheduler_state = getattr(state, "scheduler", None)
    return [str(item) for item in list(getattr(scheduler_state, "scheduled_ue_ids", []) or [])]


def get_cluster_grants(state: Any) -> list[Any]:
    explicit = getattr(state, "cluster_grants", None)
    if isinstance(explicit, list):
        return list(explicit)

    if is_multi_cell_runtime(state):
        aggregated_grants: list[Any] = []
        for runtime_state in get_managed_cells(state).values():
            runtime_scheduler = getattr(runtime_state, "scheduler", None)
            aggregated_grants.extend(list(getattr(runtime_scheduler, "grants", []) or []))
        return aggregated_grants

    scheduler_state = getattr(state, "scheduler", None)
    return list(getattr(scheduler_state, "grants", []) or [])


def get_cluster_scheduler_demands(state: Any) -> list[Any]:
    explicit = getattr(state, "cluster_last_demands", None)
    if isinstance(explicit, list):
        return list(explicit)

    if is_multi_cell_runtime(state):
        aggregated_demands: list[Any] = []
        for runtime_state in get_managed_cells(state).values():
            runtime_scheduler = getattr(runtime_state, "scheduler", None)
            aggregated_demands.extend(list(getattr(runtime_scheduler, "last_demands", []) or []))
        return aggregated_demands

    scheduler_state = getattr(state, "scheduler", None)
    return list(getattr(scheduler_state, "last_demands", []) or [])


def get_cluster_scheduler_stage_trace(state: Any) -> Any | None:
    explicit = getattr(state, "cluster_last_stage_trace", None)
    if explicit is not None:
        return explicit
    scheduler_state = getattr(state, "scheduler", None)
    return getattr(scheduler_state, "last_stage_trace", None)


def get_cluster_resource_grid_summary(state: Any) -> dict[str, float | int]:
    if is_multi_cell_runtime(state):
        total_resource_elements = 0
        allocated_resource_elements = 0
        scheduled_ue_count = 0
        scheduled_layer_count_total = 0
        mu_mimo_overlap_pair_count = 0
        mu_mimo_group_count = 0
        mu_mimo_ue_count = 0
        adjacent_interference_pair_count = 0
        interference_group_count = 0
        n_prbs_total = 0
        n_ofdm_symbols_per_slot = 0
        fft_size = 0
        overlap_values: list[float] = []
        interference_values: list[float] = []

        for runtime_state in get_managed_cells(state).values():
            resource_grid = getattr(runtime_state, "resource_grid", None)
            if resource_grid is None:
                continue
            total_resource_elements += int(getattr(resource_grid, "total_resource_elements", 0))
            allocated_resource_elements += int(getattr(resource_grid, "allocated_resource_elements", 0))
            scheduled_ue_count += int(getattr(resource_grid, "scheduled_ue_count", 0))
            scheduled_layer_count_total += int(
                getattr(resource_grid, "scheduled_layer_count_total", 0)
            )
            mu_mimo_overlap_pair_count += int(
                getattr(resource_grid, "mu_mimo_overlap_pair_count", 0)
            )
            mu_mimo_group_count += int(getattr(resource_grid, "mu_mimo_group_count", 0))
            mu_mimo_ue_count += int(getattr(resource_grid, "mu_mimo_ue_count", 0))
            adjacent_interference_pair_count += int(
                getattr(resource_grid, "adjacent_interference_pair_count", 0)
            )
            interference_group_count += int(len(getattr(resource_grid, "interference_groups", [])))
            n_prbs_total += int(getattr(resource_grid, "n_prbs_total", 0))
            n_ofdm_symbols_per_slot = max(
                n_ofdm_symbols_per_slot,
                int(getattr(resource_grid, "n_ofdm_symbols_per_slot", 0)),
            )
            fft_size = max(fft_size, int(getattr(resource_grid, "fft_size", 0)))
            overlap_values.append(float(getattr(resource_grid, "overlap_fraction_mean", 0.0)))
            interference_values.append(
                float(getattr(resource_grid, "interference_penalty_db_mean", 0.0))
            )

        utilization = (
            float(allocated_resource_elements) / float(total_resource_elements)
            if total_resource_elements > 0
            else 0.0
        )
        overlap_fraction_mean = (
            float(sum(overlap_values) / float(len(overlap_values)))
            if len(overlap_values) > 0
            else 0.0
        )
        interference_penalty_db_mean = (
            float(sum(interference_values) / float(len(interference_values)))
            if len(interference_values) > 0
            else 0.0
        )
        return {
            "total_resource_elements": int(total_resource_elements),
            "allocated_resource_elements": int(allocated_resource_elements),
            "utilization": float(utilization),
            "scheduled_ue_count": int(scheduled_ue_count),
            "scheduled_layer_count_total": int(scheduled_layer_count_total),
            "mu_mimo_overlap_pair_count": int(mu_mimo_overlap_pair_count),
            "mu_mimo_group_count": int(mu_mimo_group_count),
            "mu_mimo_ue_count": int(mu_mimo_ue_count),
            "adjacent_interference_pair_count": int(adjacent_interference_pair_count),
            "interference_group_count": int(interference_group_count),
            "n_prbs_total": int(n_prbs_total),
            "n_ofdm_symbols_per_slot": int(n_ofdm_symbols_per_slot),
            "fft_size": int(fft_size),
            "overlap_fraction_mean": float(overlap_fraction_mean),
            "interference_penalty_db_mean": float(interference_penalty_db_mean),
        }

    resource_grid = getattr(state, "resource_grid", None)
    return {
        "total_resource_elements": int(getattr(resource_grid, "total_resource_elements", 0)),
        "allocated_resource_elements": int(
            getattr(resource_grid, "allocated_resource_elements", 0)
        ),
        "utilization": float(getattr(resource_grid, "utilization", 0.0)),
        "scheduled_ue_count": int(getattr(resource_grid, "scheduled_ue_count", 0)),
        "scheduled_layer_count_total": int(
            getattr(resource_grid, "scheduled_layer_count_total", 0)
        ),
        "mu_mimo_overlap_pair_count": int(
            getattr(resource_grid, "mu_mimo_overlap_pair_count", 0)
        ),
        "mu_mimo_group_count": int(getattr(resource_grid, "mu_mimo_group_count", 0)),
        "mu_mimo_ue_count": int(getattr(resource_grid, "mu_mimo_ue_count", 0)),
        "adjacent_interference_pair_count": int(
            getattr(resource_grid, "adjacent_interference_pair_count", 0)
        ),
        "interference_group_count": int(len(getattr(resource_grid, "interference_groups", []))),
        "n_prbs_total": int(getattr(resource_grid, "n_prbs_total", 0)),
        "n_ofdm_symbols_per_slot": int(
            getattr(resource_grid, "n_ofdm_symbols_per_slot", 0)
        ),
        "fft_size": int(getattr(resource_grid, "fft_size", 0)),
        "overlap_fraction_mean": float(getattr(resource_grid, "overlap_fraction_mean", 0.0)),
        "interference_penalty_db_mean": float(
            getattr(resource_grid, "interference_penalty_db_mean", 0.0)
        ),
    }
