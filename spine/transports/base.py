"""Transport interface. Apps / coordinator only ever see this, never serial or sim details."""
from __future__ import annotations

from abc import ABC, abstractmethod


class Transport(ABC):
    name = "base"
    is_simulated = False

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def write(self, data: bytes) -> None: ...

    @abstractmethod
    def read(self, timeout: float = 0.1) -> bytes:
        """Return whatever bytes are available (b'' on timeout)."""

    def set_hint(self, material: str | None, zone: str | None) -> None:
        """Simulation-only convenience; real transports ignore it."""

    def simulate_tap(self, material: str, zone: str) -> bool:
        """Simulation-only; returns False on real hardware."""
        return False

    @property
    def description(self) -> str:
        return self.name
