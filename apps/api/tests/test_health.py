from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check_returns_healthy_status() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "traffic-vision-api",
        "environment": "development",
    }


def test_root_returns_api_information() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["health"] == "/api/health"
    assert response.json()["documentation"] == "/docs"