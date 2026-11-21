import numpy as np
import pytest

from app.evaluation import evaluate_model
from app.prediction import predict_fraud
from app.training import select_threshold, train_model


def test_probabilities_are_valid(model, splits):  # T04
    _, _, X_test, _, _, _ = splits
    proba, labels = predict_fraud(model, X_test, 0.5)
    assert proba.shape == (len(X_test),)
    assert np.isfinite(proba).all() and ((proba >= 0) & (proba <= 1)).all()
    assert set(np.unique(labels)) <= {0, 1}


def test_threshold_monotonic(model, splits):
    _, _, X_test, _, _, _ = splits
    _, low = predict_fraud(model, X_test, 0.1)
    _, high = predict_fraud(model, X_test, 0.9)
    assert low.sum() >= high.sum()


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_invalid_threshold_rejected(model, splits, bad):
    with pytest.raises(ValueError):
        predict_fraud(model, splits[2], bad)


def test_invalid_input_rejected(model, splits):
    X = splits[2].copy().head(10)
    X.loc[X.index[0], "Amount"] = np.nan
    with pytest.raises(ValueError):
        predict_fraud(model, X)


def test_train_rejects_single_class(splits):
    X, y, = splits[0].head(100), splits[3].head(100) * 0
    with pytest.raises(ValueError, match="deux classes"):
        train_model(X, y)


def test_unknown_model_rejected(splits):
    with pytest.raises(ValueError):
        train_model(splits[0], splits[3], "inconnu")


def test_evaluate_model_known_values():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 1, 1, 0])
    proba = np.array([.1, .2, .6, .7, .9, .8, .55, .4])
    m = evaluate_model(y_true, y_pred, proba)
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (3, 2, 2, 1)
    assert np.isclose(m["precision"], 0.6) and np.isclose(m["recall"], 0.75)
    assert np.isclose(m["fpr"], 0.5) and np.isclose(m["fnr"], 0.25)


def test_accuracy_paradox():
    """Un modèle qui prédit toujours « normale » a une accuracy élevée et un rappel nul."""
    y_true = np.array([0] * 998 + [1] * 2)
    m = evaluate_model(y_true, np.zeros(1000, dtype=int), np.full(1000, 0.001) + np.arange(1000) * 1e-9)
    assert m["accuracy"] > 0.99 and m["recall"] == 0.0 and m["fnr"] == 1.0


def test_select_threshold_respects_min_recall(model, splits):
    _, X_val, _, _, y_val, _ = splits
    proba = model.predict_proba(X_val)[:, 1]
    t = select_threshold(y_val, proba, min_recall=0.8)
    recall = ((proba >= t) & (y_val == 1)).sum() / (y_val == 1).sum()
    assert recall >= 0.8

def test_select_threshold_fallback_to_f1(splits):
    import numpy as np
    y = np.array([0, 0, 0, 1, 1, 0, 1, 0])
    proba = np.array([.9, .8, .7, .1, .2, .6, .3, .5])  # classifieur très mauvais
    t = select_threshold(y, proba, min_recall=1.1)  # rappel inatteignable -> repli F1
    assert 0.0 <= t <= 1.0

def test_train_rejects_empty_or_mismatched(splits):
    with pytest.raises(ValueError, match="même longueur"):
        train_model(splits[0].head(10), splits[3].head(5))

def test_echec_volontaire():
    assert False, "Échec volontaire pour démontrer la détection par la CI"