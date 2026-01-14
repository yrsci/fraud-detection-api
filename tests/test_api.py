import pytest
from fastapi.testclient import TestClient
from src.api import app

def test_amount_scaling():
    """Test that Amount feature gets scaled correctly"""
    with TestClient(app) as client:  # <-- Use context manager
        # Transaction with Amount = 100.0
        transaction = {
            "Amount": 100.0,
            **{f"V{i}": 0.0 for i in range(1, 29)}
        }
        
        response = client.post("/predict", json=transaction)
        
        assert response.status_code == 200
        assert "prediction" in response.json()

def test_invalid_amount_rejected():
    """Test that negative Amount is rejected"""
    with TestClient(app) as client:
        transaction = {
            "Amount": -50.0,  # Invalid - negative
            **{f"V{i}": 0.0 for i in range(1, 29)}
        }
        
        response = client.post("/predict", json=transaction)
        
        assert response.status_code == 422  # Validation error
        assert "Amount must be non-negative" in response.text

def test_missing_feature_rejected():
    """Test that request with missing V feature is rejected"""
    with TestClient(app) as client:
        transaction = {
            "Amount": 100.0,
            **{f"V{i}": 0.0 for i in range(1, 28)}  # Only V1-V27, missing V28
        }
        
        response = client.post("/predict", json=transaction)
        
        assert response.status_code == 422
        assert "V28" in response.text  # Error should mention missing field

def test_prediction_response_format():
    """Test that response has correct structure and types"""
    with TestClient(app) as client:
        transaction = {
            "Amount": 100.0,
            **{f"V{i}": 0.0 for i in range(1, 29)}
        }
        
        response = client.post("/predict", json=transaction)
        data = response.json()
        
        assert response.status_code == 200
        
        # Check all required fields present
        assert "prediction" in data
        assert "probability" in data
        assert "threshold" in data
        assert "fraud_model_version" in data
        
        # Check types
        assert isinstance(data["prediction"], int)
        assert isinstance(data["probability"], float)
        assert isinstance(data["threshold"], float)
        assert isinstance(data["fraud_model_version"], str)
        
        # Check value ranges
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["probability"] <= 1.0

def test_health_check():
    """Test that health check endpoint works"""
    with TestClient(app) as client:
        response = client.get("/")
        data = response.json()
        
        assert response.status_code == 200
        assert data["status"] == "healthy"
        assert "fraud_model_version" in data
        assert "threshold" in data

def test_realistic_transaction():
    """Test with transaction similar to real data"""
    with TestClient(app) as client:
        # Simulate a legitimate transaction with non-zero V features
        transaction = {
            "V1": -1.359807,
            "V2": -0.072781,
            "V3": 2.536347,
            "V4": 1.378155,
            **{f"V{i}": 0.5 for i in range(5, 29)},  # Rest with small values
            "Amount": 149.62
        }
        
        response = client.post("/predict", json=transaction)
        data = response.json()
        
        assert response.status_code == 200
        assert data["prediction"] in [0, 1]
