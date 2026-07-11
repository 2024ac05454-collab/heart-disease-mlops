from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_endpoint():
    payload = {
        "age": 50, "sex": 1, "cp": 1, "trestbps": 120, "chol": 200,
        "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.0,
        "slope": 1, "ca": 0, "thal": 2
    }
    response = client.post("/predict", json=payload)
    
    # Check if the API successfully responded
    assert response.status_code == 200
    
    # Check if the model successfully returned the expected keys
    data = response.json()
    assert "prediction" in data
    assert "confidence_score" in data