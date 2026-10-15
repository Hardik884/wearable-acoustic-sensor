"""Recorder app: collects labelled real taps -> data/real/<material>/<zone>/<n>.npz.

When the transport is SIMULATED the files go to data/simulated_recordings/ instead, so
simulated taps can never end up in the "real" training set by accident.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .. import dataset as D
from ..config import CFG, REAL_DIR, SIMREC_DIR
from ..coordinator import Coordinator, Listener


class RecorderApp(Listener):
    def __init__(self, coord: Coordinator, emit: Callable[[dict], None] | None = None):
        self.coord = coord
        self.emit = emit or (lambda m: None)
        self.root: Path = SIMREC_DIR if coord.transport.is_simulated else REAL_DIR
        self.source = "SIMULATED" if coord.transport.is_simulated else "REAL"
        self.active = False
        self.material: str | None = None
        self.zone: str | None = None
        self.target = 40
        self.session_count = 0
        self.user = "default"
        self.last_path: Path | None = None
        self.history: list[Path] = []

    def _dir(self, material: str, zone: str) -> Path:
        return self.root / material / zone

    def counts(self) -> dict:
        return D.count_dir(self.root, CFG["materials"], CFG["zones"])

    def state(self) -> dict:
        return dict(active=self.active, material=self.material, zone=self.zone, target=self.target,
                    session_count=self.session_count, counts=self.counts(), source=self.source,
                    root=str(self.root), materials=CFG["materials"], zones=CFG["zones"])

    def start(self, material: str, zone: str, target: int = 40, user: str = "default") -> None:
        if material not in CFG["materials"] or zone not in CFG["zones"]:
            raise ValueError("unknown material/zone")
        self.material, self.zone, self.target, self.user = material, zone, int(target), user
        self.session_count = 0
        self.active = True
        self.coord.sim_hint(material, zone)
        self.emit(dict(type="recorder", **self.state()))

    def stop(self) -> None:
        self.active = False
        self.coord.sim_hint(None, None)
        self.emit(dict(type="recorder", **self.state()))

    def delete_last(self) -> bool:
        while self.history:
            p = self.history.pop()
            if p.exists():
                p.unlink()
                self.session_count = max(0, self.session_count - 1)
                self.emit(dict(type="recorder", **self.state()))
                return True
        return False

    def on_data(self, ev) -> None:
        if not self.active or not self.material:
            return
        d = self._dir(self.material, self.zone)
        d.mkdir(parents=True, exist_ok=True)
        nums = [int(p.stem) for p in d.glob("*.npz") if p.stem.isdigit()]
        path = d / f"{(max(nums) + 1) if nums else 1}.npz"
        prof = self.coord.env.profile or {}
        D.save_event(path, ev.raw, self.material, self.zone, self.user,
                     "SIMULATED" if ev.simulated else "REAL", prof.get("baseline_db"))
        self.history.append(path)
        self.session_count += 1
        if self.session_count >= self.target:
            self.stop()
        else:
            self.emit(dict(type="recorder", saved=True, **self.state()))
