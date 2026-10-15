import time

import numpy as np

from spine.apps.recorder import RecorderApp
from spine.apps.safety import DEFAULT_RULES, SafetyApp
from spine.apps.store import Store
from spine.coordinator import Coordinator
from spine.transports.sim_transport import SimTransport


def make_safety(tmp_path):
    from spine.apps.safety import save_rules
    p = tmp_path / "rules.yaml"
    save_rules(DEFAULT_RULES, p)
    got = []
    return SafetyApp(Store(":memory:"), emit=got.append, path=p), got


def test_safety_default_rules(tmp_path):
    app, got = make_safety(tmp_path)
    t = 1000.0
    assert not app.evaluate("metal", "wrist", 0.5, t)            # below 0.7 -> no danger
    a = app.evaluate("metal", "wrist", 0.9, t + 2)
    assert [x["level"] for x in a] == ["danger"]
    assert app.evaluate("unknown", "wrist", 0.0, t + 4)[0]["level"] == "info"
    # 3 metal contacts within 30 s -> warning (contacts at t+10, +12, +14)
    out = []
    for dt in (10, 12, 14):
        out += app.evaluate("metal", "mid_forearm", 0.8, t + dt)
    assert [x["level"] for x in out] == ["warning"]
    assert len(app.store.alerts()) == len(got) == 3


def test_recorder_saves_and_deletes(tmp_path, monkeypatch):
    c = Coordinator(SimTransport(seed=1, auto_taps=False))
    rec = RecorderApp(c)
    rec.root = tmp_path
    c.add_listener(rec)
    c.start()
    try:
        assert c.connect_and_start(live=False)
        rec.start("wood", "wrist", target=3)
        for _ in range(3):
            c.transport.simulate_tap("wood", "wrist")
            time.sleep(0.5)
        end = time.time() + 5
        while rec.active and time.time() < end:
            time.sleep(0.1)
        files = sorted((tmp_path / "wood" / "wrist").glob("*.npz"))
        assert len(files) == 3 and not rec.active
        d = np.load(files[0])
        assert d["material"] == "wood" and d["ch"].shape[0] == 2 and str(d["source"]) == "SIMULATED"
        assert rec.delete_last() and len(list((tmp_path / "wood" / "wrist").glob("*.npz"))) == 2
    finally:
        c.close()


def test_recorder_sim_goes_to_separate_folder():
    from spine.config import REAL_DIR, SIMREC_DIR
    c = Coordinator(SimTransport(auto_taps=False))
    assert RecorderApp(c).root == SIMREC_DIR != REAL_DIR
