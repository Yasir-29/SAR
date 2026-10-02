import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"

def test_analyze_disaster_valid_request():
    payload = {
        "latitude": 11.0168,
        "longitude": 76.9558,
        "disaster_date": "2025-01-15",
        "before_days": 30,
        "after_days": 30,
        "cloud_cover_max": 30.0
    }
    response = client.post("/api/disaster/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "request_id" in data
    assert data["location"]["latitude"] == 11.0168
    assert data["location"]["longitude"] == 76.9558
    assert "before" in data
    assert "after" in data
    assert "comparison" in data
    assert "model" in data
    assert data["model"]["class"] in ["INTACT", "DAMAGED", "DESTROYED"]
    assert 0.0 <= data["model"]["confidence"] <= 1.0

def test_analyze_disaster_invalid_latitude():
    payload = {
        "latitude": 120.0,  # Invalid lat > 90
        "longitude": 76.9558,
        "disaster_date": "2025-01-15"
    }
    response = client.post("/api/disaster/analyze", json=payload)
    assert response.status_code == 422  # Validation error

def test_analyze_disaster_invalid_date():
    payload = {
        "latitude": 11.0168,
        "longitude": 76.9558,
        "disaster_date": "15-01-2025"  # Invalid format (not YYYY-MM-DD)
    }
    response = client.post("/api/disaster/analyze", json=payload)
    assert response.status_code == 422

def test_damage_predict_endpoint():
    payload = {
        "latitude": 11.0168,
        "longitude": 76.9558
    }
    response = client.post("/api/damage/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["INTACT", "DAMAGED", "DESTROYED"]
    assert "confidence" in data
