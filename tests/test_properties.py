import pytest

pytest.importorskip("hypothesis")  # installé en CI via requirements.txt
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from app.prediction import predict_fraud  # noqa: E402


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(amount=st.floats(min_value=0, max_value=1e7, allow_nan=False, allow_infinity=False),
       v1=st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False))
def test_property_probability_always_in_unit_interval(model, splits, amount, v1):
    X = splits[2].head(1).copy()
    X["Amount"], X["V1"] = amount, v1
    proba, _ = predict_fraud(model, X)
    assert 0.0 <= proba[0] <= 1.0
