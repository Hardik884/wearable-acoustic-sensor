import json

from spine import dataset as D
from spine.ml import model as model_mod
from spine.ml import train as T
from spine.sim.generator import EventSynth


def _patch(monkeypatch, tmp_path):
    synth = tmp_path / "synth.npz"
    D.generate_synthetic(per_class=25, seed=1, path=synth, verbose=False)
    monkeypatch.setattr(D, "SYNTH_FILE", synth)
    monkeypatch.setattr(T, "MODELS", tmp_path / "models")
    monkeypatch.setattr(model_mod, "MODELS", tmp_path / "models")
    (tmp_path / "models").mkdir()
    real = tmp_path / "real"
    monkeypatch.setattr(T, "REAL_DIR", real)
    s = EventSynth(seed=9)
    for m in ("finger", "metal"):
        for z in ("wrist", "mid_forearm"):
            for i in range(12):
                D.save_event(real / m / z / f"{i + 1}.npz", s.synth(m, z), m, z, "tester", "REAL")
    return real


def test_three_sources_report_their_source(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    for source in ("synthetic", "real", "mixed"):
        tag = T.train(source, "rf", verbose=False)
        meta = json.loads((tmp_path / "models" / tag / "meta.json").read_text())
        assert meta["source"] == source and source in tag
        assert "accuracy" in meta and meta["accuracy"]["zone"] > 0.8
        report = (tmp_path / "models" / tag / "report.md").read_text()
        assert meta["banner"] in report
        assert (tmp_path / "models" / tag / "confusion_material.png").exists()
    assert "SYNTHETIC" in T.BANNER["synthetic"]
    mb = model_mod.ModelBundle.latest()
    assert mb is not None and mb.meta["source"] == "mixed"


def test_ablation_present(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    tag = T.train("synthetic", "rf", verbose=False)
    meta = json.loads((tmp_path / "models" / tag / "meta.json").read_text())
    assert set(meta["results"]["ablation"]) == {"tap-only (passive)", "probe-only (active)", "combined (hybrid)"}
