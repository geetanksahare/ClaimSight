import pytest
from fastapi.testclient import TestClient

from claimsight.api.main import app
from claimsight.config import get_settings


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "models_loaded" in data
    assert "disclaimer" in data


def test_get_models(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "cost" in data["models"]
    assert "member" in data["models"]
    assert "fraud" in data["models"]


def test_predict_cost_endpoint(client):
    payload = {
        "age": 45,
        "sex": "male",
        "bmi": 28.0,
        "children": 1,
        "smoker": "no",
        "region": "northwest",
    }
    response = client.post("/v1/cost/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "lower" in data
    assert "upper" in data
    assert data["lower"] <= data["prediction"] <= data["upper"]


def test_explain_cost_endpoint(client):
    payload = {
        "age": 45,
        "sex": "male",
        "bmi": 28.0,
        "children": 1,
        "smoker": "no",
        "region": "northwest",
    }
    response = client.post("/v1/cost/explain?top_k=3", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "contributions" in data
    assert "summary_text" in data
    assert len(data["contributions"]) <= 3


def test_what_if_endpoint(client):
    payload = {
        "base": {
            "age": 45,
            "sex": "male",
            "bmi": 32.0,
            "children": 1,
            "smoker": "yes",
            "region": "northwest",
        },
        "changes": {
            "smoker": "no",
        },
    }
    response = client.post("/v1/cost/what-if", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "baseline" in data
    assert "new" in data
    assert "delta_abs" in data
    # Quitting smoking should decrease predicted charges
    assert data["delta_abs"] < 0


def test_score_single_member_endpoint(client):
    payload = {
        "member_id": "MEM_API_001",
        "age": 70,
        "sex": "female",
        "race": "1",
        "esrd": 0,
        "chronic_flags": {"SP_DIABETES": 1, "SP_CHF": 1},
        "prior_cost_ip": 1500.0,
        "prior_cost_op": 500.0,
        "prior_cost_car": 300.0,
        "ip_claims": 1,
        "op_claims": 2,
        "carrier_claims": 3,
        "ip_days": 2,
        "has_history": 1,
        "cost_change": 200.0,
    }
    response = client.post("/v1/members/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "expected_cost" in data
    assert "prob_high_cost" in data
    assert "risk_tier" in data


def test_score_batch_members_endpoint(client):
    payload = [
        {
            "member_id": f"MEM_{i:03d}",
            "age": 65 + i,
            "sex": "male",
            "race": "1",
            "esrd": 0,
            "chronic_flags": {},
            "prior_cost_ip": 0.0,
            "prior_cost_op": 100.0,
            "prior_cost_car": 50.0,
            "ip_claims": 0,
            "op_claims": 1,
            "carrier_claims": 1,
            "ip_days": 0,
            "has_history": 0,
            "cost_change": 0.0,
        }
        for i in range(5)
    ]
    response = client.post("/v1/members/score-batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5
    for item in data:
        assert "expected_cost" in item
        assert "risk_tier" in item


def test_batch_exceeds_max_limit(client, monkeypatch):
    from claimsight.api import main

    new_cfg = main.cfg.model_copy(
        update={"api": main.cfg.api.model_copy(update={"max_batch_rows": 3})}
    )
    monkeypatch.setattr(main, "cfg", new_cfg)

    payload = [
        {
            "member_id": f"MEM_{i:03d}",
            "age": 65,
            "sex": "male",
            "race": "1",
            "esrd": 0,
            "chronic_flags": {},
            "prior_cost_ip": 0.0,
            "prior_cost_op": 0.0,
            "prior_cost_car": 0.0,
            "ip_claims": 0,
            "op_claims": 0,
            "carrier_claims": 0,
            "ip_days": 0,
            "has_history": 0,
            "cost_change": 0.0,
        }
        for i in range(5)
    ]
    response = client.post("/v1/members/score-batch", json=payload)
    assert response.status_code == 413


def test_validation_error_handling(client):
    # Invalid age (-5) and invalid sex ("unknown")
    bad_payload = {
        "age": -5,
        "sex": "unknown",
        "bmi": 25.0,
        "children": 0,
        "smoker": "no",
        "region": "northeast",
    }
    response = client.post("/v1/cost/predict", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "validation_error"
    assert "details" in data


def test_api_key_authentication(monkeypatch):
    from claimsight.api import security

    settings = get_settings()
    # Configure an API key
    keyed_settings = settings.model_copy(update={"api_key": "secret-test-key"})
    monkeypatch.setattr(security, "get_settings", lambda: keyed_settings)

    with TestClient(app) as key_client:
        # Request without header should be 401
        res_no_key = key_client.get("/v1/models")
        assert res_no_key.status_code == 401

        # Request with bad key should be 401
        res_bad_key = key_client.get("/v1/models", headers={"X-API-Key": "wrong-key"})
        assert res_bad_key.status_code == 401

        # Request with correct key should be 200
        res_good_key = key_client.get("/v1/models", headers={"X-API-Key": "secret-test-key"})
        assert res_good_key.status_code == 200
