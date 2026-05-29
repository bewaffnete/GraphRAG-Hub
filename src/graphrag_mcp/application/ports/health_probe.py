"""Health probe port."""

from typing import Protocol


class HealthProbe(Protocol):
    def check(self) -> dict[str, object]:
        """Return health metadata."""
