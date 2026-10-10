"""ran/command_consumer.py

Consumes one-shot CommandAction objects and applies their effects to the
current simulator state for the ongoing slot.
"""

from __future__ import annotations

from core.debug import debug_print
from models.action import CommandAction


class CommandConsumer:
    """Applies one-shot commands into slot-local scheduler context."""

    def apply_commands(self, state, commands: list[CommandAction]) -> None:
        # Reset per-slot command context.
        if hasattr(getattr(state, "scheduler", None), "priority_boost_by_ue"):
            state.scheduler.priority_boost_by_ue = {}

        if not commands:
            return

        slot_id = int(getattr(state, "slot_id", -1))

        for command in commands:
            command_type = str(getattr(command, "command_type", ""))

            valid_from_slot = getattr(command, "valid_from_slot", None)
            valid_until_slot = getattr(command, "valid_until_slot", None)

            if valid_from_slot is not None and slot_id < int(valid_from_slot):
                continue
            if valid_until_slot is not None and slot_id > int(valid_until_slot):
                continue

            if command_type == "boost_ue_priority_once":
                ue_id = getattr(command, "target_ue_id", None)
                if ue_id is None:
                    continue

                delta = float(getattr(command, "delta_priority", 0.0))
                if not hasattr(getattr(state, "scheduler", None), "priority_boost_by_ue"):
                    continue

                current = float(state.scheduler.priority_boost_by_ue.get(str(ue_id), 0.0))
                state.scheduler.priority_boost_by_ue[str(ue_id)] = max(current, delta)

                debug_print(
                    f"Command consumed at slot {slot_id}: boost_ue_priority_once "
                    f"ue_id={ue_id}, delta_priority={delta:.2f}."
                )
                continue

            debug_print(
                f"Unknown command ignored at slot {slot_id}: command_type={command_type}."
            )
