from api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_validation():

    response = client.post(
        "/predict",
        json={
            "job_description": "short",
            "resume_text": "short",
        },
    )

    assert response.status_code == 422
