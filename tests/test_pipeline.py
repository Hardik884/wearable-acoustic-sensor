import time

from spine.coordinator import Coordinator, Listener
from spine.ml.train import ensure_model
from spine.transports.sim_transport import SimTransport


class Collect(Listener):
    def __init__(self):
        self.events, self.live, self.msgs, self.nodes = [], 0, [], []

    def on_data(self, ev): self.events.append(ev)
    def on_live(self, l): self.live += 1
    def on_service_message(self, m): self.msgs.append(m)
    def on_new_node(self, i): self.nodes.append(i)


def test_sim_end_to_end():
    c = Coordinator(SimTransport(seed=3, auto_taps=False))
    c.set_models(ensure_model(verbose=False))
    col = Collect()
    c.add_listener(col)
    c.start()
    try:
        info = c.connect_and_start(live=True)
        assert info and info["board_name"] == "SIMULATED" and info["rate_hz"] == 10000
        truth = [("metal", "wrist"), ("wood", "mid_forearm"), ("finger", "wrist"), ("plastic", "mid_forearm")]
        for m, z in truth:
            c.transport.simulate_tap(m, z)
            time.sleep(0.45)
        deadline = time.time() + 5
        while len(col.events) < len(truth) and time.time() < deadline:
            time.sleep(0.1)
        assert len(col.events) == len(truth)
        assert col.live > 5
        assert all(e.simulated for e in col.events)
        ok = sum(e.final_material == m and e.final_zone == z for e, (m, z) in zip(col.events, truth))
        assert ok >= 3, [(e.final_material, e.final_zone) for e in col.events]
        assert any(m["code_name"] == "ACK" for m in col.msgs)
    finally:
        c.close()
