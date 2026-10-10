"""Slow-timescale active intent lifecycle manager."""

from __future__ import annotations

from dataclasses import dataclass, field

from models.intent_orchestration import ActiveIntentRecord
from ric.non_rt.intent_orchestration_helpers import stable_intent_id


@dataclass
class ActiveIntentEvent:
    """One event-driven active intent update."""

    epoch_id: int
    intent_text: str
    change_reason: str = "operator_update"
    intent_id: str = ""
    source_name: str = "operator"
    summary: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


class ActiveIntentManager:
    """Maintains the piecewise-constant active intent across Non-RT epochs."""

    def __init__(
        self,
        *,
        initial_intent_text: str = "",
        initial_epoch: int = 1,
        initial_change_reason: str = "initial_intent",
        source_name: str = "operator",
        scheduled_events: list[ActiveIntentEvent] | None = None,
    ) -> None:
        self._source_name = str(source_name or "operator")
        self._events_by_epoch: dict[int, ActiveIntentEvent] = {}
        self._processed_epoch = 0
        self._changed_epochs: set[int] = set()
        self._active_intent: ActiveIntentRecord | None = None

        if str(initial_intent_text or "").strip() != "":
            self.update_intent(
                ActiveIntentEvent(
                    epoch_id=max(1, int(initial_epoch)),
                    intent_text=str(initial_intent_text).strip(),
                    change_reason=str(initial_change_reason or "initial_intent"),
                    source_name=self._source_name,
                )
            )
        for event in list(scheduled_events or []):
            self.update_intent(event)

    def update_intent(
        self,
        event_or_policy: ActiveIntentEvent | str,
        *,
        epoch_id: int | None = None,
        change_reason: str = "operator_update",
        source_name: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Register a future or immediate intent update event."""

        if isinstance(event_or_policy, ActiveIntentEvent):
            event = event_or_policy
        else:
            normalized_text = str(event_or_policy or "").strip()
            if normalized_text == "":
                return
            resolved_epoch = max(1, int(epoch_id if epoch_id is not None else (self._processed_epoch + 1)))
            event = ActiveIntentEvent(
                epoch_id=resolved_epoch,
                intent_text=normalized_text,
                change_reason=str(change_reason or "operator_update"),
                source_name=str(source_name or self._source_name),
                metadata=dict(metadata or {}),
            )
        normalized_epoch = max(1, int(event.epoch_id))
        normalized_text = str(event.intent_text or "").strip()
        if normalized_text == "":
            return
        event.intent_text = normalized_text
        event.epoch_id = normalized_epoch
        if str(event.intent_id or "").strip() == "":
            event.intent_id = stable_intent_id(normalized_text)
        if str(event.source_name or "").strip() == "":
            event.source_name = self._source_name
        self._events_by_epoch[normalized_epoch] = event

    def _advance_to_epoch(self, epoch_id: int) -> None:
        normalized_epoch = max(0, int(epoch_id))
        if normalized_epoch <= self._processed_epoch:
            return
        for current_epoch in range(self._processed_epoch + 1, normalized_epoch + 1):
            event = self._events_by_epoch.get(current_epoch)
            if event is None:
                continue
            self._active_intent = ActiveIntentRecord(
                intent_id=str(event.intent_id or stable_intent_id(event.intent_text)),
                intent_text=str(event.intent_text or ""),
                summary=str(event.summary or ""),
                effective_from_epoch=int(current_epoch),
                last_update_epoch=int(current_epoch),
                last_change_reason=str(event.change_reason or "operator_update"),
                source_name=str(event.source_name or self._source_name),
                metadata=dict(event.metadata or {}),
            )
            self._changed_epochs.add(int(current_epoch))
        self._processed_epoch = normalized_epoch

    def get_active_intent(self, epoch_id: int) -> ActiveIntentRecord | None:
        self._advance_to_epoch(epoch_id)
        return self._active_intent

    def is_intent_changed(self, epoch_id: int) -> bool:
        self._advance_to_epoch(epoch_id)
        return int(epoch_id) in self._changed_epochs

    def get_epoch_state(self, epoch_id: int) -> tuple[ActiveIntentRecord | None, bool]:
        self._advance_to_epoch(epoch_id)
        return self._active_intent, self.is_intent_changed(epoch_id)
