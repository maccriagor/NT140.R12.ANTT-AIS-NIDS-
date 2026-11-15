"""End-to-end test that streams with unknown runs < 4 trigger no update (Plan Week 5 acceptance criterion 3 / P2)."""
import numpy as np
import torch

import pipeline as pl
import skeleton as sk


def test_short_unknown_runs_trigger_no_update():
    sk.set_seed(42)
    model = sk.build_model("mlp", 4)
    X_tr = np.random.randint(0, 256, size=(80, 1500), dtype=np.uint8)
    y_tr = np.repeat(np.arange(4, dtype=np.int64), 20)
    X_va = np.random.randint(0, 256, size=(40, 1500), dtype=np.uint8)
    y_va = np.repeat(np.arange(4, dtype=np.int64), 10)
    memory = pl.MemoryModule.build((X_tr, y_tr), (X_va, y_va), ild_min_per_class=5, vd_min_per_class=5, seed=42)

    # Craft a stream where unknown flags occur only in runs of length <= 3
    X_stream = np.random.randint(0, 256, size=(30, 1500), dtype=np.uint8)
    confs, _ = pl.predict_conf(model, X_stream)
    # Pick a threshold t=0.0 so nothing is flagged, or monkeypatch detect_unknown to return runs of length 3
    orig_detect = pl.detect_unknown
    try:
        def fake_detect(m, x, t, temperature=1.0):
            n = len(x)
            flags = np.zeros(n, dtype=bool)
            # Runs of length 3 separated by False
            flags[2:5] = True
            flags[10:13] = True
            flags[20:23] = True
            return np.full(n, 0.5, dtype=np.float32), np.zeros(n, dtype=np.int64), flags

        pl.detect_unknown = fake_detect
        class_names = ["BENIGN", "A", "B", "C"]
        model_out, log = pl.run_episode(
            model, memory, X_stream, t=0.9, class_names=class_names,
            cfg=pl.UpdateConfig(refine_epochs=1), min_run=4
        )
        assert not log.accepted
        assert log.kept_runs == 0
        assert log.discarded_runs == 3
        assert log.n_collected == 0
        assert len(class_names) == 4
        assert pl.n_outputs(model_out) == 4
        assert "too few collected packets, nothing learned" in log.notes
    finally:
        pl.detect_unknown = orig_detect
