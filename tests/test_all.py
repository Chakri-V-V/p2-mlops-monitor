"""
tests/test_all.py — Pytest suite for P2 MLOps Monitor
Run: pytest tests/ -v
"""
import json, pickle
import pandas as pd
import numpy as np
import pytest
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def model():
    with open("model/model.pkl", "rb") as f:
        return pickle.load(f)

@pytest.fixture(scope="session")
def meta():
    with open("model/metadata.json") as f:
        return json.load(f)

@pytest.fixture(scope="session")
def ref_df():
    return pd.read_csv("model/ref_df.csv")

@pytest.fixture(scope="session")
def client():
    from api.app import app
    return TestClient(app)


SAMPLE = {
    "age": 45, "limit_bal": 150000.0, "pay_delay": 3,
    "bill_amt": 80000.0, "pay_amt": 5000.0, "education": 2,
}

LOW_RISK = {
    "age": 30, "limit_bal": 500000.0, "pay_delay": -1,
    "bill_amt": 10000.0, "pay_amt": 50000.0, "education": 1,
}


# ── Model tests ───────────────────────────────────────────────────
def test_model_roc_auc_above_threshold(model, ref_df, meta):
    from sklearn.metrics import roc_auc_score
    X = ref_df[meta["features"]]
    y = ref_df["default"]
    probs = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs)
    assert auc >= 0.80, f"AUC {auc:.4f} below 0.80"

def test_model_output_shape(model, meta):
    row = pd.DataFrame([SAMPLE])[meta["features"]]
    proba = model.predict_proba(row)
    assert proba.shape == (1, 2)
    assert 0 <= proba[0, 1] <= 1

def test_meta_keys(meta):
    for key in ["features", "roc_auc", "default_rate"]:
        assert key in meta


# ── API tests ─────────────────────────────────────────────────────
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_high_risk(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert "prob" in body
    assert "risk_level" in body
    assert 0 <= body["prob"] <= 1
    assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH")

def test_predict_low_risk(client):
    r = client.post("/predict", json=LOW_RISK)
    assert r.status_code == 200
    assert r.json()["risk_level"] == "LOW"

def test_predict_missing_field(client):
    bad = {k: v for k, v in SAMPLE.items() if k != "pay_delay"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422

def test_prediction_logged(client, tmp_path):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    from pathlib import Path
    log = Path("logs/predictions.csv")
    assert log.exists(), "Prediction log not created"
    df = pd.read_csv(log)
    assert len(df) >= 1
    assert "prob" in df.columns


# ── Monitoring tests ──────────────────────────────────────────────
def test_drift_report_exists():
    from pathlib import Path
    assert Path("reports/drift_metrics.json").exists(), "Run drift_check.py first"

def test_nannyml_report_exists():
    from pathlib import Path
    assert Path("reports/nannyml_summary.json").exists(), "Run performance_estimate.py first"

def test_drift_report_structure():
    with open("reports/drift_metrics.json") as f:
        d = json.load(f)
    assert "dataset_drift_detected" in d
    assert "number_of_drifted_columns" in d
    assert "drift_by_column" in d
