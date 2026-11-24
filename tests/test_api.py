"""End-to-end tests for every endpoint: happy paths, bad input and auth."""
import pytest

# ---------------------------------------------------------------- ops / info


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    body = client.get("/model/info").json()
    assert body["algorithm"] == "LogisticRegression"
    assert 0 < body["decision_threshold"] < 1
    assert body["test_metrics"]["roc_auc"] > 0.8  # guards against shipping a broken model


def test_request_id_header_is_returned(client):
    r = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"


# ---------------------------------------------------------------- predictions


def test_predict_high_risk(client, high_risk):
    body = client.post("/predict", json=high_risk).json()
    assert body["customer_id"] == "TEST-HIGH"
    assert body["will_churn"] is True
    assert body["risk_tier"] == "High"
    assert 1 <= len(body["top_risk_factors"]) <= 3


def test_predict_low_risk(client, low_risk):
    body = client.post("/predict", json=low_risk).json()
    assert body["will_churn"] is False
    assert body["risk_tier"] == "Low"
    assert body["churn_probability"] < 0.1


def test_contract_explains_high_risk(client, high_risk):
    factors = client.post("/predict", json=high_risk).json()["top_risk_factors"]
    assert any("Month-to-month" in f or "tenure" in f or "Fiber" in f for f in factors)


def test_longer_contract_lowers_risk(client, high_risk):
    """Sanity check on model behavior: same customer, two-year contract -> lower risk."""
    before = client.post("/predict", json=high_risk).json()["churn_probability"]
    high_risk["Contract"] = "Two year"
    after = client.post("/predict", json=high_risk).json()["churn_probability"]
    assert after < before


def test_predictions_are_deterministic(client, high_risk):
    a = client.post("/predict", json=high_risk).json()
    b = client.post("/predict", json=high_risk).json()
    assert a == b


def test_new_customer_with_zero_tenure(client, high_risk):
    """Edge case: brand-new customer (tenure 0, no charges yet) must not crash on divide-by-zero."""
    high_risk.update(tenure=0, TotalCharges=0)
    r = client.post("/predict", json=high_risk)
    assert r.status_code == 200
    assert 0 <= r.json()["churn_probability"] <= 1


def test_customer_id_is_optional(client, low_risk):
    del low_risk["customer_id"]
    r = client.post("/predict", json=low_risk)
    assert r.status_code == 200
    assert r.json()["customer_id"] is None


def test_batch(client, high_risk, low_risk):
    body = client.post("/predict/batch", json={"customers": [high_risk, low_risk]}).json()
    assert body["count"] == 2
    assert body["high_risk_count"] == 1
    assert [p["customer_id"] for p in body["predictions"]] == ["TEST-HIGH", "TEST-LOW"]


# ---------------------------------------------------------------- input validation (422)


@pytest.mark.parametrize("field,bad_value", [
    ("Contract", "Three year"),
    ("tenure", -1),
    ("tenure", 500),
    ("MonthlyCharges", 0),
    ("SeniorCitizen", 2),
    ("PaymentMethod", "Bitcoin"),
])
def test_rejects_invalid_values(client, high_risk, field, bad_value):
    high_risk[field] = bad_value
    r = client.post("/predict", json=high_risk)
    assert r.status_code == 422
    assert field in str(r.json()["detail"])


def test_rejects_missing_field(client, high_risk):
    del high_risk["Contract"]
    assert client.post("/predict", json=high_risk).status_code == 422


def test_rejects_inconsistent_internet_fields(client, high_risk):
    high_risk["InternetService"] = "No"  # but add-ons still say "No" instead of "No internet service"
    r = client.post("/predict", json=high_risk)
    assert r.status_code == 422
    assert "No internet service" in r.text


def test_rejects_inconsistent_phone_fields(client, high_risk):
    high_risk["PhoneService"] = "No"
    assert client.post("/predict", json=high_risk).status_code == 422


def test_batch_limits(client, low_risk):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422
    assert client.post("/predict/batch", json={"customers": [low_risk] * 501}).status_code == 422


# ---------------------------------------------------------------- auth


def test_api_key_required_when_configured(client, low_risk, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-key")
    assert client.post("/predict", json=low_risk).status_code == 401
    assert client.post("/predict", json=low_risk, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/predict", json=low_risk, headers={"X-API-Key": "secret-key"}).status_code == 200


def test_health_does_not_need_api_key(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-key")
    assert client.get("/health").status_code == 200
