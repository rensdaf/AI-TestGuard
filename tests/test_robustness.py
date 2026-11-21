import numpy as np
import pytest

from app.prediction import predict_fraud


def test_extreme_amount_gives_valid_probabilities(model, splits):
    X = splits[2].copy()
    X["Amount"] = 1e9
    proba, _ = predict_fraud(model, X)
    assert np.isfinite(proba).all() and ((proba >= 0) & (proba <= 1)).all()


def test_small_perturbation_is_a_bounded_sensitivity(model, splits, threshold):
    """On mesure la sensibilité (pas d'invariance exigée) : bruit multiplicatif 1 % sur Amount."""
    X = splits[2].copy()
    rng = np.random.default_rng(0)
    Xp = X.copy()
    Xp["Amount"] = X["Amount"] * (1 + rng.normal(0, 0.01, len(X)))
    p0, l0 = predict_fraud(model, X, threshold)
    p1, l1 = predict_fraud(model, Xp, threshold)
    assert np.abs(p1 - p0).mean() < 0.01
    assert (l0 != l1).mean() < 0.02


def test_flips_concentrate_near_boundary(model, splits, threshold):
    X = splits[2].copy()
    Xp = X.copy()
    Xp["V1"] = X["V1"] + 0.05
    p0, l0 = predict_fraud(model, X, threshold)
    _, l1 = predict_fraud(model, Xp, threshold)
    flipped = l0 != l1
    if flipped.any():
        assert np.abs(p0[flipped] - threshold).max() < 0.25


@pytest.mark.parametrize("col,bad", [("V3", np.nan), ("Amount", np.inf), ("V9", -np.inf)])
def test_non_finite_inputs_rejected(model, splits, col, bad):
    X = splits[2].head(20).copy()
    X.loc[X.index[2], col] = bad
    with pytest.raises(ValueError):
        predict_fraud(model, X)


def test_non_numeric_input_rejected(model, splits):
    X = splits[2].head(20).copy().astype({"V2": object})
    X.loc[X.index[0], "V2"] = "abc"
    with pytest.raises(TypeError):
        predict_fraud(model, X)


def test_invalid_schema_rejected(model, splits):
    with pytest.raises(ValueError):
        predict_fraud(model, splits[2].head(5).drop(columns=["V7"]))


def test_distribution_shift_alert_not_crash(model, splits, threshold):
    """Décalage de distribution : la prédiction reste valide ; le taux d'alertes bouge (signal de dérive)."""
    X = splits[2].copy()
    base = predict_fraud(model, X, threshold)[1].mean()
    X[[f"V{i}" for i in range(1, 15)]] += 1.5
    proba, labels = predict_fraud(model, X, threshold)
    assert np.isfinite(proba).all()
    assert labels.mean() != base
