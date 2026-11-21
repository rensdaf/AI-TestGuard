import pytest

from app.preprocessing import split_data
from app.synthetic import make_synthetic_df
from app.training import select_threshold, train_model

SEED = 42


@pytest.fixture(scope="session")
def df():
    return make_synthetic_df(n=20000, fraud_rate=0.01, seed=SEED)


@pytest.fixture(scope="session")
def splits(df):
    return split_data(df, seed=SEED)


@pytest.fixture(scope="session")
def model(splits):
    X_train, _, _, y_train, _, _ = splits
    return train_model(X_train, y_train, "logreg", seed=SEED)


@pytest.fixture(scope="session")
def threshold(model, splits):
    _, X_val, _, _, y_val, _ = splits
    return select_threshold(y_val, model.predict_proba(X_val)[:, 1], min_recall=0.80)


@pytest.fixture()
def valid_df(df):
    return df.head(50).copy()
