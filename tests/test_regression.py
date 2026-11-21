"""Non-régression : comparaison à une baseline FIGÉE (reports/baseline.json), jamais recalculée ici."""
import json
from pathlib import Path


from app.evaluation import evaluate_model
from app.prediction import predict_fraud
from app.preprocessing import split_data
from app.synthetic import make_synthetic_df
from app.training import train_model

BASELINE = Path(__file__).resolve().parents[1] / "reports" / "baseline.json"
TOL = {"pr_auc": 0.03, "recall": 0.05, "fpr": 0.01}


def _run(model_name, params=None):
    b = json.loads(BASELINE.read_text())
    df = make_synthetic_df(**b["dataset"])
    X_tr, X_va, _, y_tr, y_va, _ = split_data(df, seed=b["seed"])
    model = train_model(X_tr, y_tr, model_name, seed=b["seed"], **(params or {}))
    # seuil de la baseline GELÉ pour comparer à décision identique
    proba, labels = predict_fraud(model, X_va, b["threshold"])
    return b, evaluate_model(y_va, labels, proba), model, X_va, y_va


def _check(b, m):
    ref = b["metrics_validation"]
    return {
        "pr_auc": ref["pr_auc"] - m["pr_auc"] <= TOL["pr_auc"],
        "recall": ref["recall"] - m["recall"] <= TOL["recall"],
        "fpr": m["fpr"] - ref["fpr"] <= TOL["fpr"],
    }


def test_baseline_file_complete():
    b = json.loads(BASELINE.read_text())
    for key in ["code_version", "libraries", "seed", "threshold", "dataset", "metrics_validation", "model"]:
        assert key in b


def test_current_version_within_tolerance():
    name = json.loads(BASELINE.read_text())["model"]["name"]
    b, m, *_ = _run(name)
    assert all(_check(b, m).values()), (name, m)


def test_degraded_model_is_detected():
    """Contrôle du contrôle : un modèle volontairement dégradé doit être rejeté."""
    b, m, *_ = _run("tree", {"max_depth": 1})
    assert not all(_check(b, m).values())
