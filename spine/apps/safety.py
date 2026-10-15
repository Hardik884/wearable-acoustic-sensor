"""Safety app: user-editable rules -> alerts (stored in SQLite, pushed to the dashboard)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Callable

import yaml

from ..config import RULES_PATH
from ..coordinator import Listener
from .store import Store

DEFAULT_RULES = [
    dict(id="metal_wrist_danger", name="Metal on wrist", material="metal", zone="wrist", min_confidence=0.7,
         count=1, window_s=0, level="danger", message="Possible hazardous contact (metal on wrist)"),
    dict(id="metal_repeated", name="Repeated metal exposure", material="metal", zone="any",
         min_confidence=0.5, count=3, window_s=30, level="warning",
         message="Repeated exposure: 3+ metal contacts within 30 s"),
    dict(id="unknown_wrist", name="Unknown material on wrist", material="unknown", zone="wrist",
         min_confidence=0.0, count=1, window_s=0, level="info", message="Unrecognised contact on wrist"),
]
LEVELS = ("info", "warning", "danger")


def load_rules(path=RULES_PATH) -> list[dict]:
    if not path.exists():
        save_rules(DEFAULT_RULES, path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))["rules"]


def save_rules(rules: list[dict], path=RULES_PATH) -> None:
    path.write_text("# Safety rules. Edit here or in the dashboard (SAFETY tab).\n" +
                    yaml.safe_dump({"rules": rules}, sort_keys=False), encoding="utf-8")


def validate_rule(r: dict) -> dict:
    out = dict(id=str(r.get("id") or f"rule_{int(time.time() * 1000) % 10**8}"), name=str(r.get("name", "")),
               material=str(r.get("material", "any")), zone=str(r.get("zone", "any")),
               min_confidence=float(r.get("min_confidence", 0.0)), count=max(1, int(r.get("count", 1))),
               window_s=float(r.get("window_s", 0)), level=str(r.get("level", "warning")),
               message=str(r.get("message", "")))
    if out["level"] not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    return out


class SafetyApp(Listener):
    def __init__(self, store: Store, emit: Callable[[dict], None] | None = None, path=RULES_PATH):
        self.store, self.emit, self.path = store, emit or (lambda a: None), path
        self.rules = load_rules(path)
        self._hits: dict[str, deque] = defaultdict(deque)
        self._last_fire: dict[str, float] = {}

    def set_rules(self, rules: list[dict]) -> None:
        self.rules = [validate_rule(r) for r in rules]
        save_rules(self.rules, self.path)
        self._hits.clear()

    @staticmethod
    def matches(rule: dict, material: str, zone: str, m_conf: float) -> bool:
        if rule["material"] not in ("any", "*") and rule["material"] != material:
            return False
        if rule["zone"] not in ("any", "*") and rule["zone"] != zone:
            return False
        if material != "unknown" and m_conf < rule["min_confidence"]:
            return False
        return True

    def evaluate(self, material: str, zone: str, m_conf: float, now: float, source: str = "") -> list[dict]:
        fired = []
        for r in self.rules:
            if not self.matches(r, material, zone, m_conf):
                continue
            hits = self._hits[r["id"]]
            hits.append(now)
            if r["window_s"] > 0:
                while hits and now - hits[0] > r["window_s"]:
                    hits.popleft()
            if len(hits) < r["count"]:
                continue
            if now - self._last_fire.get(r["id"], 0) < 1.0:
                continue
            hits.clear()
            self._last_fire[r["id"]] = now
            alert = dict(ts=now, level=r["level"], rule=r["name"] or r["id"], message=r["message"],
                         material=material, zone=zone, source=source)
            alert["id"] = self.store.add_alert(alert)
            fired.append(alert)
            self.emit(alert)
        return fired

    # SPINE listener API
    def on_data(self, event) -> None:
        self.evaluate(event.final_material, event.final_zone, event.material_conf, event.ts,
                      "SIMULATED" if event.simulated else "REAL")
