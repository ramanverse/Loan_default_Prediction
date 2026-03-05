"""
Integration tests for FastAPI endpoints.
"""
import pytest
import sys
import os
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "degraded"]


def test_model_info():
    """Test model info endpoint."""
    response = client.get("/model/info")
    # May return 404 if model not loaded, which is acceptable
    assert response.status_code in [200, 404]


def test_predict_endpoint_missing_data():
    """Test predict endpoint with missing required fields."""
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_invalid_data():
    """Test predict endpoint with invalid data."""
    invalid_data = {
        "amt_income_total": -100,  # Negative income
        "amt_credit": 100000,
        "amt_annuity": 5000,
        "age": 30,
        "employ_years": 5
    }
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error


def test_predict_batch_empty():
    """Test batch predict with empty data."""
    response = client.post("/predict/batch", json={"data": []})
    assert response.status_code in [200, 400]


def test_predict_batch_invalid_format():
    """Test batch predict with invalid format."""
    response = client.post("/predict/batch", json={"invalid": "data"})
    assert response.status_code in [422, 400]
