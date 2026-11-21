import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from app.preprocessing import (FEATURE_COLUMNS, REQUIRED_COLUMNS, preprocess_data,
                               split_data, validate_data)


def test_standard_scaler():
    data = pd.DataFrame({"Amount": [100.0, 200.0, 300.0]})
    scaled = StandardScaler().fit_transform(data)
    assert np.isfinite(scaled).all()
    assert abs(scaled.mean()) < 1e-10
    assert np.isclose(scaled.std(), 1.0)


def test_valid_data_accepted(valid_df):  # T01
    assert validate_data(valid_df) is valid_df


@pytest.mark.parametrize("col", REQUIRED_COLUMNS)
def test_missing_column_rejected(valid_df, col):
    with pytest.raises(ValueError, match="manquantes"):
        validate_data(valid_df.drop(columns=[col]))


def test_target_not_required_for_inference(valid_df):
    validate_data(valid_df.drop(columns=["Class"]), require_target=False)


def test_duplicated_column_rejected(valid_df):
    dup = pd.concat([valid_df, valid_df[["Amount"]]], axis=1)
    with pytest.raises(ValueError, match="dupliqu"):
        validate_data(dup)


def test_empty_rejected(valid_df):
    with pytest.raises(ValueError, match="vide"):
        validate_data(valid_df.iloc[0:0])


def test_non_dataframe_rejected():
    with pytest.raises(TypeError):
        validate_data([[1, 2, 3]])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_rejected(valid_df, bad):  # T03
    valid_df.loc[valid_df.index[3], "V5"] = bad
    with pytest.raises(ValueError):
        validate_data(valid_df)


def test_wrong_type_rejected(valid_df):  # T02
    valid_df["Amount"] = valid_df["Amount"].astype(str)
    with pytest.raises(TypeError, match="Amount"):
        validate_data(valid_df)


@pytest.mark.parametrize("bad_target", [2, -1, 0.5])
def test_invalid_target_rejected(valid_df, bad_target):
    valid_df["Class"] = valid_df["Class"].astype(float)
    valid_df.loc[valid_df.index[0], "Class"] = bad_target
    with pytest.raises(ValueError, match="cible"):
        validate_data(valid_df)


def test_negative_amount_rejected(valid_df):
    valid_df.loc[valid_df.index[0], "Amount"] = -5.0
    with pytest.raises(ValueError, match="négatif"):
        validate_data(valid_df)


def test_single_observation_and_dimensions(valid_df):
    X, y = preprocess_data(valid_df.head(1))
    assert X.shape == (1, len(FEATURE_COLUMNS)) and len(y) == 1


def test_preprocess_dimensions_and_order(valid_df):
    shuffled = valid_df[list(reversed(valid_df.columns))]
    X, y = preprocess_data(shuffled)
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.shape[0] == y.shape[0] == len(valid_df)


def test_extra_column_ignored(valid_df):
    valid_df["extra"] = 1
    X, _ = preprocess_data(valid_df)
    assert "extra" not in X.columns


def test_split_is_disjoint_stratified_and_complete(df):
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(df, seed=1)
    assert len(X_tr) + len(X_va) + len(X_te) == len(df)
    rate = df["Class"].mean()
    for y in (y_tr, y_va, y_te):
        assert abs(y.mean() - rate) < 0.002
    assert abs(len(X_te) / len(df) - 0.2) < 0.01 and abs(len(X_va) / len(df) - 0.2) < 0.01


def test_split_reproducible_with_seed(df):
    a = split_data(df, seed=7)[0]
    b = split_data(df, seed=7)[0]
    pd.testing.assert_frame_equal(a, b)
