class SimulationClock:
    """
    Simple slot-based simulation clock.
    """

    def __init__(self, slot_duration_ms: float = 1.0) -> None:
        self.slot_duration_ms = slot_duration_ms
        self.current_slot = 0

    @property
    def current_time_ms(self) -> float:
        return self.current_slot * self.slot_duration_ms

    def tick(self) -> None:
        self.current_slot += 1