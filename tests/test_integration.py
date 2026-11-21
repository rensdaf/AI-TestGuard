import numpy as np
import pytest

from app.evaluation import evaluate_model
from app.prediction import predict_fraud
from app.preprocessing import split_data
from app.training import load_pipeline, save_pipeline, select_threshold, train_model


def test_end_to_end_save_reload(tmp_path, df):
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(df, seed=42)
    model = train_model(X_tr, y_tr, "logreg", seed=42)
    thr = select_threshold(y_va, model.predict_proba(X_va)[:, 1])
    path = tmp_path / "pipeline.joblib"
    save_pipeline(model, thr, path, {"seed": 42})

    loaded = load_pipeline(path)
    p_before, l_before = predict_fraud(model, X_te, thr)
    p_after, l_after = predict_fraud(loaded["pipeline"], X_te, loaded["threshold"])
    np.testing.assert_allclose(p_before, p_after, atol=1e-12)
    assert (l_before == l_after).all()
    assert loaded["threshold"] == thr and loaded["metadata"]["seed"] == 42
    metrics = evaluate_model(y_te, l_after, p_after)
    assert metrics["pr_auc"] > 0.5
    assert metrics["tp"] + metrics["fn"] == int(y_te.sum())


def test_dimensions_in_out(model, splits):
    X_te = splits[2]
    assert model.named_steps["prep"].transform(X_te).shape == X_te.shape
    assert predict_fraud(model, X_te)[0].shape[0] == len(X_te)


def test_scaler_fitted_on_train_only(splits):
    """Anti-fuite : les paramètres du scaler proviennent du train, pas du test."""
    X_tr, _, X_te, y_tr, _, _ = splits
    model = train_model(X_tr, y_tr, "logreg", seed=42)
    scaler = model.named_steps["prep"].named_transformers_["scale"]
    np.testing.assert_allclose(scaler.center_, X_tr[["Time", "Amount"]].median().to_numpy())
    assert not np.allclose(scaler.center_, X_te[["Time", "Amount"]].median().to_numpy())


def test_no_non_finite_after_preprocessing(model, splits):
    out = model.named_steps["prep"].transform(splits[2])
    assert np.isfinite(out).all()


def test_missing_file_raises_explicitly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pipeline(tmp_path / "absent.joblib")


def test_reproducibility_same_seed(splits):
    X_tr, _, X_te, y_tr, _, _ = splits
    p1 = train_model(X_tr, y_tr, "tree", seed=3).predict_proba(X_te)
    p2 = train_model(X_tr, y_tr, "tree", seed=3).predict_proba(X_te)
    np.testing.assert_array_equal(p1, p2)  # T06
