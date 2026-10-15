"""SQLite storage for events and alerts (+ CSV export)."""
from __future__ import annotations

import csv
import io
import sqlite3
import threading
import time
from pathlib import Path

from ..config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, zone TEXT, material TEXT, zone_conf REAL,
  material_conf REAL, probe INTEGER, source TEXT, user TEXT, tdoa_ms REAL, ratio_db REAL,
  probs TEXT);
CREATE TABLE IF NOT EXISTS alerts(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, level TEXT, rule TEXT, message TEXT,
  material TEXT, zone TEXT, acked INTEGER DEFAULT 0, source TEXT);
"""


class Store:
    def __init__(self, path: Path | str = DB_PATH):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(SCHEMA)

    # ------------------------------------------------------------ events
    def add_event(self, e: dict, user: str = "default") -> int:
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO events(ts,zone,material,zone_conf,material_conf,probe,source,user,tdoa_ms,ratio_db,probs)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (e["ts"], e["zone"], e["material"], e["zone_conf"], e["material_conf"], int(e["probe_used"]),
                 "SIMULATED" if e["simulated"] else "REAL", user, e.get("tdoa_ms"), e.get("ratio_db"),
                 str(e.get("material_probs"))))
            self.db.commit()
            return cur.lastrowid

    def events(self, limit: int = 200, material: str | None = None, zone: str | None = None,
               since: float | None = None) -> list[dict]:
        q, a = "SELECT * FROM events WHERE 1=1", []
        if material:
            q += " AND material=?"; a.append(material)
        if zone:
            q += " AND zone=?"; a.append(zone)
        if since:
            q += " AND ts>=?"; a.append(since)
        q += " ORDER BY id DESC LIMIT ?"; a.append(limit)
        with self.lock:
            return [dict(r) for r in self.db.execute(q, a).fetchall()]

    def analytics(self, bins: int = 30) -> dict:
        with self.lock:
            rows = [dict(r) for r in self.db.execute("SELECT ts,zone,material FROM events ORDER BY ts").fetchall()]
        by_m, by_z, heat = {}, {}, {}
        for r in rows:
            by_m[r["material"]] = by_m.get(r["material"], 0) + 1
            by_z[r["zone"]] = by_z.get(r["zone"], 0) + 1
            k = f"{r['zone']}|{r['material']}"
            heat[k] = heat.get(k, 0) + 1
        timeline = []
        if rows:
            t0, t1 = rows[0]["ts"], max(rows[-1]["ts"], rows[0]["ts"] + 1)
            w = (t1 - t0) / bins
            counts = [0] * bins
            for r in rows:
                counts[min(int((r["ts"] - t0) / w), bins - 1)] += 1
            timeline = [dict(t=t0 + i * w, n=c) for i, c in enumerate(counts)]
        return dict(total=len(rows), by_material=by_m, by_zone=by_z, heatmap=heat, timeline=timeline)

    def clear_events(self) -> None:
        with self.lock:
            self.db.execute("DELETE FROM events")
            self.db.commit()

    # ------------------------------------------------------------ alerts
    def add_alert(self, a: dict) -> int:
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO alerts(ts,level,rule,message,material,zone,source) VALUES(?,?,?,?,?,?,?)",
                (a["ts"], a["level"], a["rule"], a["message"], a.get("material"), a.get("zone"), a.get("source", "")))
            self.db.commit()
            return cur.lastrowid

    def alerts(self, limit: int = 200) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))]

    def ack_alert(self, alert_id: int | None = None) -> None:
        with self.lock:
            if alert_id is None:
                self.db.execute("UPDATE alerts SET acked=1")
            else:
                self.db.execute("UPDATE alerts SET acked=1 WHERE id=?", (alert_id,))
            self.db.commit()

    # ------------------------------------------------------------ csv
    @staticmethod
    def to_csv(rows: list[dict]) -> str:
        if not rows:
            return ""
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            r = dict(r)
            if "ts" in r:
                r["time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"]))
            w.writerow({k: r.get(k) for k in w.fieldnames})
        return buf.getvalue()
