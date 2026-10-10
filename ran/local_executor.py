"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: local_executor.py
Description: This file defines the local executor used by the simulator.
The local executor applies executable actions directly to the current
global state without modeling transport delay or external protocol steps.
"""

from core.debug import debug_print
from models.action import Action
from ran.executor_base import ExecutorBase
from state.global_state import GlobalState
from types import SimpleNamespace
from typing import Any


class LocalExecutor(ExecutorBase):
    """
    Implements a local immediate executor for slot-level actions.
    INPUT
    No explicit input during initialization.
    OUTPUT
    A local executor object that applies actions directly to the state.
    EXAMPLE OUTPUT
    executor = LocalExecutor()
    executor.apply_action(state, action)
    """

    _SCHEDULER_HINT_TO_MODE = {
        "balanced": "round_robin",
        "latency_bias": "latency_aware",
        "slice_bias": "slice_aware",
    }

    _SCHEDULER_MODE_TO_HINT = {
        "round_robin": "balanced",
        "latency_aware": "latency_bias",
        "slice_aware": "slice_bias",
    }

    _BEAM_PREFERENCE_TO_PROFILE = {
        "balanced": (0, 0),
        "capacity": (1, 0),
        "coverage": (2, 1),
    }

    def _slot_cell_text(self, state: Any) -> str:
        cell_id = str(getattr(getattr(state, "cell", None), "cell_id", "cell_0"))
        return f"cell_id={cell_id} at slot {state.slot_id}"

    def _resolve_target_states(self, state: GlobalState, action: Action) -> list[tuple[str, Any]]:
        managed_cells = getattr(state, "managed_cells", {})
        if not isinstance(managed_cells, dict) or len(managed_cells) == 0:
            primary_cell_id = str(getattr(getattr(state, "cell", None), "cell_id", "cell_0"))
            return [(primary_cell_id, state)]

        apply_scope = str(getattr(action, "apply_scope", "cell")).strip().lower()
        target_cell_id = getattr(action, "target_cell_id", None)
        target_states: list[tuple[str, Any]] = []

        if apply_scope == "cluster":
            for cell_id, runtime_state in managed_cells.items():
                target_states.append(
                    (
                        str(cell_id),
                        SimpleNamespace(
                            slot_id=state.slot_id,
                            cell=runtime_state.cell,
                            scheduler=runtime_state.scheduler,
                        ),
                    )
                )
            return target_states

        if target_cell_id is not None:
            runtime_state = managed_cells.get(str(target_cell_id))
            if runtime_state is not None:
                return [
                    (
                        str(target_cell_id),
                        SimpleNamespace(
                            slot_id=state.slot_id,
                            cell=runtime_state.cell,
                            scheduler=runtime_state.scheduler,
                        ),
                    )
                ]

        primary_cell_id = str(getattr(getattr(state, "cell", None), "cell_id", "cell_0"))
        return [(primary_cell_id, state)]

    def resolve_target_cell_ids(self, state: GlobalState, action: Action) -> list[str]:
        return [str(cell_id) for cell_id, _ in self._resolve_target_states(state, action)]

    def _apply_traffic_steering_preference(
        self,
        state: Any,
        traffic_steering_preference: str,
    ) -> None:
        preference = str(traffic_steering_preference).strip().lower()
        if preference == "":
            return
        state.cell.traffic_steering_preference = preference
        debug_print(
            f"Applied traffic_steering_preference={preference} on {self._slot_cell_text(state)}."
        )

    def _apply_mobility_bias_db(self, state: Any, mobility_bias_db: float) -> None:
        state.cell.mobility_bias_db = float(mobility_bias_db)
        debug_print(
            f"Applied mobility_bias_db={state.cell.mobility_bias_db:.2f} on {self._slot_cell_text(state)}."
        )

    def _apply_handover_cost_db(self, state: Any, handover_cost_db: float) -> None:
        state.cell.handover_cost_db = max(0.0, float(handover_cost_db))
        debug_print(
            f"Applied handover_cost_db={state.cell.handover_cost_db:.2f} on {self._slot_cell_text(state)}."
        )

    def _apply_handover_interruption_slots(self, state: Any, handover_interruption_slots: int) -> None:
        state.cell.handover_interruption_slots = max(0, int(handover_interruption_slots))
        debug_print(
            f"Applied handover_interruption_slots={state.cell.handover_interruption_slots} on {self._slot_cell_text(state)}."
        )

    def _apply_steering_hysteresis_db(self, state: Any, steering_hysteresis_db: float) -> None:
        state.cell.steering_hysteresis_db = max(0.0, float(steering_hysteresis_db))
        debug_print(
            f"Applied steering_hysteresis_db={state.cell.steering_hysteresis_db:.2f} on {self._slot_cell_text(state)}."
        )

    def _apply_measurement_event_trigger(self, state: Any, measurement_event_trigger: str) -> None:
        normalized_trigger = str(measurement_event_trigger).strip().lower()
        if normalized_trigger == "":
            return
        state.cell.measurement_event_trigger = normalized_trigger
        debug_print(
            f"Applied measurement_event_trigger={normalized_trigger} on {self._slot_cell_text(state)}."
        )

    def _apply_measurement_event_offset_db(self, state: Any, measurement_event_offset_db: float) -> None:
        state.cell.measurement_event_offset_db = float(measurement_event_offset_db)
        debug_print(
            f"Applied measurement_event_offset_db={state.cell.measurement_event_offset_db:.2f} on {self._slot_cell_text(state)}."
        )

    def _apply_scheduler_policy_hint(self, state: Any, scheduler_policy_hint: str) -> None:
        hint = str(scheduler_policy_hint).strip().lower()
        if hint == "":
            return

        state.scheduler.scheduler_policy_hint = hint
        mapped_mode = self._SCHEDULER_HINT_TO_MODE.get(hint)
        if mapped_mode is not None:
            state.scheduler.scheduler_mode = mapped_mode

        if hint != "latency_bias":
            state.scheduler.latency_bias = None
        if hint != "slice_bias":
            state.scheduler.slice_weight_adjustment = {}

        debug_print(
            f"Applied scheduler_policy_hint={hint} on {self._slot_cell_text(state)}. "
            f"mapped_mode={state.scheduler.scheduler_mode}."
        )

    def _apply_latency_bias(self, state: Any, latency_bias: float) -> None:
        state.scheduler.latency_bias = max(0.1, float(latency_bias))
        debug_print(
            f"Applied latency_bias={state.scheduler.latency_bias:.2f} on {self._slot_cell_text(state)}."
        )

    def _apply_slice_weight_adjustment(
        self,
        state: Any,
        slice_weight_adjustment: dict[str, float],
    ) -> None:
        normalized_adjustment = {
            str(slice_id).lower(): float(weight)
            for slice_id, weight in dict(slice_weight_adjustment).items()
        }
        state.scheduler.slice_weight_adjustment = normalized_adjustment
        debug_print(
            f"Applied slice_weight_adjustment={normalized_adjustment} on {self._slot_cell_text(state)}."
        )

    def _apply_beam_candidate_preference(
        self,
        state: Any,
        beam_candidate_preference: str,
    ) -> None:
        preference = str(beam_candidate_preference).strip().lower()
        if preference == "":
            return

        state.cell.beam_candidate_preference = preference
        codebook_id, beam_profile_id = self._BEAM_PREFERENCE_TO_PROFILE.get(
            preference,
            self._BEAM_PREFERENCE_TO_PROFILE["balanced"],
        )
        state.cell.current_codebook_id = int(codebook_id)
        state.cell.current_beam_profile_id = int(beam_profile_id)
        debug_print(
            f"Applied beam_candidate_preference={preference} on {self._slot_cell_text(state)}. "
            f"derived_codebook_id={codebook_id}, derived_beam_profile_id={beam_profile_id}."
        )

    def _apply_power_cap_dbm(self, state: Any, power_cap_dbm: float) -> None:
        state.cell.power_cap_dbm = float(power_cap_dbm)
        # In this simulator, the power cap is interpreted as the currently
        # effective transmit-power ceiling and applied immediately.
        state.cell.tx_power_dbm = float(state.cell.power_cap_dbm)
        if str(getattr(state.cell, "cell_sleep_mode", "active")) == "active":
            state.cell.nominal_tx_power_dbm = float(state.cell.tx_power_dbm)
        debug_print(
            f"Applied power_cap_dbm={state.cell.power_cap_dbm:.2f} on {self._slot_cell_text(state)}. "
            f"effective_tx_power_dbm={state.cell.tx_power_dbm:.2f}."
        )

    def _apply_cell_sleep_mode(self, state: Any, cell_sleep_mode: str) -> None:
        normalized_mode = str(cell_sleep_mode).strip().lower()
        if normalized_mode == "":
            return
        if normalized_mode not in {"active", "micro_sleep", "light_sleep", "deep_sleep"}:
            normalized_mode = "active"

        current_tx_power = float(getattr(state.cell, "tx_power_dbm", 30.0))
        if str(getattr(state.cell, "cell_sleep_mode", "active")) == "active":
            state.cell.nominal_tx_power_dbm = float(getattr(state.cell, "nominal_tx_power_dbm", current_tx_power))
            state.cell.nominal_tx_power_dbm = float(current_tx_power)

        state.cell.cell_sleep_mode = normalized_mode
        nominal_tx_power = float(getattr(state.cell, "nominal_tx_power_dbm", current_tx_power))
        if normalized_mode == "active":
            if getattr(state.cell, "power_cap_dbm", None) is not None and float(state.cell.power_cap_dbm) <= 0.0:
                state.cell.power_cap_dbm = None
            restored_tx_power = nominal_tx_power
            if state.cell.power_cap_dbm is not None:
                restored_tx_power = min(restored_tx_power, float(state.cell.power_cap_dbm))
            state.cell.tx_power_dbm = float(restored_tx_power)
        elif normalized_mode == "micro_sleep":
            state.cell.tx_power_dbm = float(min(nominal_tx_power, max(12.0, nominal_tx_power - 3.0)))
        elif normalized_mode == "light_sleep":
            state.cell.tx_power_dbm = float(min(nominal_tx_power, max(6.0, nominal_tx_power - 8.0)))
        else:
            state.cell.tx_power_dbm = 0.0
        debug_print(
            f"Applied cell_sleep_mode={normalized_mode} on {self._slot_cell_text(state)}. "
            f"effective_tx_power_dbm={state.cell.tx_power_dbm:.2f}."
        )

    def _apply_direct_scheduler_mode(self, state: Any, scheduler_mode: str) -> None:
        mode = str(scheduler_mode).strip()
        if mode == "":
            return

        state.scheduler.scheduler_mode = mode
        mapped_hint = self._SCHEDULER_MODE_TO_HINT.get(mode)
        if mapped_hint is not None:
            state.scheduler.scheduler_policy_hint = mapped_hint
        if mode != "latency_aware":
            state.scheduler.latency_bias = None
        if mode != "slice_aware":
            state.scheduler.slice_weight_adjustment = {}
        debug_print(
            f"Applied scheduler_mode={mode} on {self._slot_cell_text(state)}."
        )

    def apply_action(self, state: GlobalState, action: Action) -> list[str]:
        """
        Applies one executable action to the current global state.
        INPUT
        state: The global simulation state.
        action: The executable action object.
        OUTPUT
        None. The state is updated in place.
        EXAMPLE OUTPUT
        The current codebook, beam profile, transmit power, or scheduler
        mode may be updated.
        """
        effective_from_slot = getattr(action, "effective_from_slot", None)
        effective_until_slot = getattr(action, "effective_until_slot", None)

        if effective_from_slot is not None and int(state.slot_id) < int(effective_from_slot):
            return []

        if effective_until_slot is not None and int(state.slot_id) > int(effective_until_slot):
            return []

        resolved_target_cell_ids: list[str] = []
        for target_cell_id, target_state in self._resolve_target_states(state, action):
            resolved_target_cell_ids.append(str(target_cell_id))
            if action.scheduler_policy_hint is not None:
                self._apply_scheduler_policy_hint(target_state, action.scheduler_policy_hint)

            if action.latency_bias is not None:
                self._apply_latency_bias(target_state, action.latency_bias)

            if action.slice_weight_adjustment:
                self._apply_slice_weight_adjustment(target_state, action.slice_weight_adjustment)

            if action.beam_candidate_preference is not None:
                self._apply_beam_candidate_preference(target_state, action.beam_candidate_preference)

            if action.power_cap_dbm is not None:
                self._apply_power_cap_dbm(target_state, action.power_cap_dbm)

            if action.cell_sleep_mode is not None:
                self._apply_cell_sleep_mode(target_state, action.cell_sleep_mode)

            if action.traffic_steering_preference is not None:
                self._apply_traffic_steering_preference(target_state, action.traffic_steering_preference)

            if action.mobility_bias_db is not None:
                self._apply_mobility_bias_db(target_state, action.mobility_bias_db)

            if action.handover_cost_db is not None:
                self._apply_handover_cost_db(target_state, action.handover_cost_db)

            if action.handover_interruption_slots is not None:
                self._apply_handover_interruption_slots(
                    target_state,
                    action.handover_interruption_slots,
                )

            if action.steering_hysteresis_db is not None:
                self._apply_steering_hysteresis_db(target_state, action.steering_hysteresis_db)

            if action.measurement_event_trigger is not None:
                self._apply_measurement_event_trigger(target_state, action.measurement_event_trigger)

            if action.measurement_event_offset_db is not None:
                self._apply_measurement_event_offset_db(
                    target_state,
                    action.measurement_event_offset_db,
                )

            if action.codebook_id is not None:
                target_state.cell.current_codebook_id = action.codebook_id
                debug_print(
                    f"Applied codebook_id={action.codebook_id} on {self._slot_cell_text(target_state)}."
                )

            if action.beam_profile_id is not None:
                target_state.cell.current_beam_profile_id = action.beam_profile_id
                debug_print(
                    f"Applied beam_profile_id={action.beam_profile_id} on {self._slot_cell_text(target_state)}."
                )

            if action.tx_power_dbm is not None:
                requested_tx_power_dbm = float(action.tx_power_dbm)
                if target_state.cell.power_cap_dbm is not None:
                    requested_tx_power_dbm = min(
                        requested_tx_power_dbm,
                        float(target_state.cell.power_cap_dbm),
                    )
                target_state.cell.tx_power_dbm = requested_tx_power_dbm
                if str(getattr(target_state.cell, "cell_sleep_mode", "active")) == "active":
                    target_state.cell.nominal_tx_power_dbm = requested_tx_power_dbm
                debug_print(
                    f"Applied tx_power_dbm={requested_tx_power_dbm:.2f} on {self._slot_cell_text(target_state)}."
                )

            if action.scheduler_mode is not None:
                self._apply_direct_scheduler_mode(target_state, action.scheduler_mode)

        return resolved_target_cell_ids
