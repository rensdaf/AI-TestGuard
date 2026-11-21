"""Crée la baseline FIGÉE (reports/baseline.json). À lancer UNE fois, puis versionner le fichier.
Ne jamais l'exécuter automatiquement en CI : cela masquerait les régressions."""
import json
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import sklearn

from app.evaluation import evaluate_model
from app.prediction import predict_fraud
from app.preprocessing import split_data
from app.synthetic import make_synthetic_df
from app.training import select_threshold, train_model

SEED, DATASET = 42, {"n": 20000, "fraud_rate": 0.01, "seed": 42}


def git_version():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "non-versionné"


df = make_synthetic_df(**DATASET)
X_tr, X_va, _, y_tr, y_va, _ = split_data(df, seed=SEED)
model = train_model(X_tr, y_tr, "logreg", seed=SEED)
thr = select_threshold(y_va, model.predict_proba(X_va)[:, 1], min_recall=0.80)
proba, labels = predict_fraud(model, X_va, thr)
baseline = {
    "code_version": git_version(),
    "model": {"name": "logreg", "params": model.named_steps["clf"].get_params()},
    "libraries": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                  "numpy": np.__version__, "pandas": pd.__version__},
    "seed": SEED,
    "threshold": thr,
    "dataset": DATASET,
    "dataset_stats": {"rows": len(df), "frauds": int(df["Class"].sum()),
                      "validation_rows": len(X_va), "validation_frauds": int(y_va.sum())},
    "metrics_validation": evaluate_model(y_va, labels, proba),
}
with open("reports/baseline.json", "w", encoding="utf-8") as f:
    json.dump(baseline, f, indent=2, default=str)
print("baseline écrite :", {k: round(v, 4) for k, v in baseline["metrics_validation"].items() if isinstance(v, float)})
sys.exit(0)
