import numpy as np
from fastapi.testclient import TestClient

from src import api


class FakeModel:
    classes_ = np.array(["negative", "positive"])

    def predict(self, reviews):
        return np.array(["positive"])

    def predict_proba(self, reviews):
        return np.array([[0.1, 0.9]])


def test_health_loads_model(monkeypatch):
    monkeypatch.setattr(api, "load_model", lambda: FakeModel())
    response = TestClient(api.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_label_and_probability(monkeypatch):
    monkeypatch.setattr(api, "load_model", lambda: FakeModel())
    response = TestClient(api.app).post(
        "/predict", json={"review": "A wonderful movie"}
    )

    assert response.status_code == 200
    assert response.json() == {"prediction": "positive", "probability": 0.9}


def test_predict_rejects_whitespace_review():
    response = TestClient(api.app).post("/predict", json={"review": "   "})

    assert response.status_code == 422
