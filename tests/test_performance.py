import time
from pathlib import Path

import numpy as np

from app.prediction import predict_fraud
from app.training import save_pipeline

P95_SINGLE_MS = 50.0   # seuil de démonstration (à justifier par le SLA métier dans le rapport)
MAX_MODEL_KB = 500


def test_single_prediction_latency_p95(model, splits):
    row = splits[2].head(1)
    predict_fraud(model, row)  # échauffement
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        predict_fraud(model, row)
        times.append((time.perf_counter() - t0) * 1000)
    assert np.percentile(times, 95) < P95_SINGLE_MS


def test_model_size(tmp_path: Path, model):
    path = tmp_path / "m.joblib"
    save_pipeline(model, 0.5, path)
    assert path.stat().st_size / 1024 < MAX_MODEL_KB


def test_no_secret_in_repo():
    root = Path(__file__).resolve().parents[1]
    bad = ("AKIA", "BEGIN PRIVATE KEY", "ghp_", "password=")
    for p in list(root.glob("app/*.py")) + list(root.glob("scripts/*.py")):
        text = p.read_text(encoding="utf-8")
        assert not any(tok in text for tok in bad), p
