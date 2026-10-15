"""SIMULATED transport: a VirtualNode on the other end of the same byte protocol."""
from __future__ import annotations

import queue

from ..sim.virtual_node import VirtualNode
from .base import Transport


class SimTransport(Transport):
    name = "SIMULATED"
    is_simulated = True

    def __init__(self, seed: int | None = None, auto_taps: bool = True, realtime: bool = True):
        self.node = VirtualNode(seed=seed, auto_taps=auto_taps, realtime=realtime)

    def open(self) -> None:
        self.node.start_thread()

    def close(self) -> None:
        self.node.stop_thread()

    def write(self, data: bytes) -> None:
        self.node.feed(data)

    def read(self, timeout: float = 0.1) -> bytes:
        try:
            data = self.node.out.get(timeout=timeout)
        except queue.Empty:
            return b""
        while True:                      # drain whatever else is ready
            try:
                data += self.node.out.get_nowait()
            except queue.Empty:
                return data

    def set_hint(self, material, zone) -> None:
        self.node.hint = (material, zone) if material and zone else None

    def simulate_tap(self, material: str, zone: str) -> bool:
        self.node.request_tap(material, zone)
        return True

    @property
    def description(self) -> str:
        return "SIMULATED virtual node (no hardware)"
