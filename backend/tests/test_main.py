
"""
NeuroSense - Backend API Tests
Run from the project root:
    python -m pytest backend/tests/test_main.py -v
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI application."""
    with TestClient(app) as test_client:
        yield test_client


def test_application_starts(client):
    """Check that the FastAPI application responds."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_health_endpoint(client):
    """Check backend health endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_system_status_endpoint(client):
    """Check system status endpoint."""
    response = client.get("/api/system/status")

    assert response.status_code == 200
    assert isinstance(response.json(), dict)


@pytest.mark.parametrize(
    ("path", "method"),
    [
        ("/health", "get"),
        ("/api/system/status", "get"),
        ("/api/sensors", "post"),
        ("/api/sensors/latest", "get"),
        ("/api/sensors/history", "get"),
        ("/api/device/status", "get"),
        ("/api/devices", "get"),
        ("/api/devices/pair", "post"),
    ],
)
def test_core_api_routes_are_registered(client, path, method):
    """Verify that the expected API routes exist in FastAPI."""
    response = client.get("/openapi.json")

    assert response.status_code == 200

    paths = response.json().get("paths", {})

    assert path in paths, f"API route is missing: {path}"
    assert method in paths[path], (
        f"HTTP method {method.upper()} is missing for {path}"
    )


def test_sensor_endpoint_rejects_empty_payload(client):
    """An empty sensor payload should fail validation."""
    response = client.post(
        "/api/sensors",
        json={},
        headers={"X-Demo-User": "neurosense-test-user"},
    )

    assert response.status_code == 422
