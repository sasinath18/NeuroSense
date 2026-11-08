
"""
NeuroSense - Backend API Tests
Run from the project root:
    python -m pytest backend/tests/test_main.py -v
"""

from datetime import datetime, timedelta, timezone

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


def test_valid_sensor_reading_is_accepted_and_stored(client):
    """POST a valid simulated sensor reading and verify the stored response."""
    response = client.post(
        "/api/sensors",
        json={
            "heart_rate_bpm": 76,
            "hrv_ms": 42,
            "gsr_raw": 1800,
            "skin_temperature_c": 33.2,
            "signal_quality": "good",
            "source": "simulated",
        },
        headers={"X-Demo-User": "api-test-sensor-post"},
    )

    assert response.status_code == 201

    body = response.json()
    assert body["success"] is True
    assert body["reading"]["user_id"] == "api-test-sensor-post"
    assert body["reading"]["device_id"] == "manual-preview"
    assert body["reading"]["heart_rate_bpm"] == 76
    assert body["reading"]["hrv_ms"] == 42
    assert body["reading"]["gsr_raw"] == 1800
    assert body["reading"]["skin_temperature_c"] == 33.2
    assert body["reading"]["signal_quality"] == "good"
    assert body["reading"]["source"] == "simulated"
    assert body["stress"]["status"] == "simulated_data"


def test_latest_sensor_reading_returns_saved_record(client):
    """GET the latest reading for the same development user after POST."""
    headers = {"X-Demo-User": "api-test-latest"}

    create_response = client.post(
        "/api/sensors",
        json={
            "heart_rate_bpm": 81,
            "hrv_ms": 38,
            "gsr_raw": 1920,
            "signal_quality": "moderate",
            "source": "simulated",
        },
        headers=headers,
    )

    assert create_response.status_code == 201
    created_id = create_response.json()["reading"]["id"]

    latest_response = client.get(
        "/api/sensors/latest",
        headers=headers,
    )

    assert latest_response.status_code == 200
    body = latest_response.json()
    assert body["data"] is not None
    assert body["data"]["id"] == created_id
    assert body["data"]["heart_rate_bpm"] == 81
    assert body["data"]["hrv_ms"] == 38
    assert body["data"]["gsr_raw"] == 1920
    assert body["data"]["source"] == "simulated"


def test_sensor_history_returns_saved_readings(client):
    """GET sensor history and verify the record created for this user."""
    headers = {"X-Demo-User": "api-test-history"}

    create_response = client.post(
        "/api/sensors",
        json={
            "heart_rate_bpm": 73,
            "hrv_ms": 45,
            "gsr_raw": 1650,
            "signal_quality": "good",
            "source": "simulated",
        },
        headers=headers,
    )

    assert create_response.status_code == 201
    created_id = create_response.json()["reading"]["id"]

    history_response = client.get(
        "/api/sensors/history?limit=10",
        headers=headers,
    )

    assert history_response.status_code == 200
    body = history_response.json()
    assert body["total"] >= 1
    assert any(
        item["id"] == created_id
        and item["heart_rate_bpm"] == 73
        and item["hrv_ms"] == 45
        and item["gsr_raw"] == 1650
        for item in body["readings"]
    )


def test_device_pairing_creates_device_and_returns_one_time_token(client):
    """Pair a device and verify a usable token is returned only in the response."""
    device_id = "test-device-pairing-01"
    response = client.post(
        "/api/devices/pair",
        json={
            "device_id": device_id,
            "device_name": "NeuroSense Test Wearable",
        },
        headers={"X-Demo-User": "api-test-device-pairing"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True
    assert body["device_id"] == device_id
    assert isinstance(body["device_token"], str)
    assert len(body["device_token"]) >= 24
    assert "shown only once" in body["warning"]


def test_device_list_returns_devices_for_authenticated_user(client):
    """Verify a newly paired device appears in that user's device list."""
    headers = {"X-Demo-User": "api-test-device-list"}
    device_id = "test-device-list-01"

    pair_response = client.post(
        "/api/devices/pair",
        json={
            "device_id": device_id,
            "device_name": "Device List Test",
        },
        headers=headers,
    )

    assert pair_response.status_code == 201

    list_response = client.get(
        "/api/devices",
        headers=headers,
    )

    assert list_response.status_code == 200
    devices = list_response.json()["devices"]
    matching = [
        item for item in devices
        if item["device_id"] == device_id
    ]

    assert len(matching) == 1
    assert matching[0]["device_name"] == "Device List Test"
    assert matching[0]["active"] == 1


def test_paired_device_can_upload_reading_with_valid_device_token(client):
    """A paired device must upload wearable data using X-Device-Token."""
    user_headers = {"X-Demo-User": "api-test-device-upload"}
    device_id = "test-device-upload-01"

    pair_response = client.post(
        "/api/devices/pair",
        json={
            "device_id": device_id,
            "device_name": "Device Upload Test",
        },
        headers=user_headers,
    )

    assert pair_response.status_code == 201
    device_token = pair_response.json()["device_token"]

    upload_response = client.post(
        f"/api/devices/{device_id}/readings",
        json={
            "heart_rate_bpm": 78,
            "hrv_ms": 44,
            "gsr_raw": 1770,
            "signal_quality": "good",
            "source": "wearable",
        },
        headers={"X-Device-Token": device_token},
    )

    assert upload_response.status_code == 201
    body = upload_response.json()
    assert body["success"] is True
    assert body["reading"]["device_id"] == device_id
    assert body["reading"]["source"] == "wearable"
    assert body["reading"]["heart_rate_bpm"] == 78
    assert body["reading"]["hrv_ms"] == 44
    assert body["reading"]["gsr_raw"] == 1770


def test_device_upload_rejects_missing_device_token(client):
    """The wearable readings endpoint must reject requests without a token."""
    pair_response = client.post(
        "/api/devices/pair",
        json={
            "device_id": "test-device-missing-token",
            "device_name": "Missing Token Test",
        },
        headers={"X-Demo-User": "api-test-missing-token"},
    )

    assert pair_response.status_code == 201

    response = client.post(
        "/api/devices/test-device-missing-token/readings",
        json={
            "heart_rate_bpm": 76,
            "signal_quality": "good",
            "source": "wearable",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "X-Device-Token header is required."


def test_device_upload_rejects_invalid_device_token(client):
    """A token with the wrong secret must not authorize device uploads."""
    device_id = "test-device-invalid-token"
    pair_response = client.post(
        "/api/devices/pair",
        json={
            "device_id": device_id,
            "device_name": "Invalid Token Test",
        },
        headers={"X-Demo-User": "api-test-invalid-token"},
    )

    assert pair_response.status_code == 201

    response = client.post(
        f"/api/devices/{device_id}/readings",
        json={
            "heart_rate_bpm": 76,
            "signal_quality": "good",
            "source": "wearable",
        },
        headers={"X-Device-Token": "x" * 48},
    )

    assert response.status_code == 401


def test_device_status_reports_recently_seen_device_as_connected(client):
    """A device that uploaded a reading within 60 seconds is connected."""
    user_headers = {"X-Demo-User": "api-test-device-status"}
    device_id = "test-device-status-01"

    pair_response = client.post(
        "/api/devices/pair",
        json={
            "device_id": device_id,
            "device_name": "Device Status Test",
        },
        headers=user_headers,
    )
    assert pair_response.status_code == 201

    token = pair_response.json()["device_token"]
    upload_response = client.post(
        f"/api/devices/{device_id}/readings",
        json={
            "heart_rate_bpm": 74,
            "signal_quality": "good",
            "source": "wearable",
        },
        headers={"X-Device-Token": token},
    )
    assert upload_response.status_code == 201

    status_response = client.get(
        "/api/device/status",
        headers=user_headers,
    )

    assert status_response.status_code == 200
    body = status_response.json()
    assert body["connected"] is True
    assert any(
        item["device_id"] == device_id and item["connected"] is True
        for item in body["devices"]
    )


def test_unpair_device_marks_device_inactive(client):
    """DELETE device endpoint should deactivate an owned paired device."""
    user_headers = {"X-Demo-User": "api-test-unpair"}
    device_id = "test-device-unpair-01"

    pair_response = client.post(
        "/api/devices/pair",
        json={
            "device_id": device_id,
            "device_name": "Unpair Test",
        },
        headers=user_headers,
    )
    assert pair_response.status_code == 201

    delete_response = client.delete(
        f"/api/devices/{device_id}",
        headers=user_headers,
    )

    assert delete_response.status_code == 200
    assert delete_response.json()["success"] is True
    assert delete_response.json()["active"] is False

    list_response = client.get("/api/devices", headers=user_headers)
    assert list_response.status_code == 200
    matching = [
        item for item in list_response.json()["devices"]
        if item["device_id"] == device_id
    ]
    assert len(matching) == 1
    assert matching[0]["active"] == 0


def test_baseline_calibration_rejects_insufficient_readings(client):
    """Baseline calibration requires at least ten usable wearable readings."""
    response = client.post(
        "/api/baseline/calibrate?lookback_minutes=10",
        headers={"X-Demo-User": "api-test-baseline-insufficient"},
    )

    assert response.status_code == 422
    assert "Need at least 10" in response.json()["detail"]


def test_baseline_calibration_and_retrieval(client):
    """Ten usable wearable readings should create a retrievable baseline."""
    headers = {"X-Demo-User": "api-test-baseline-success"}

    for index in range(10):
        reading_response = client.post(
            "/api/sensors",
            json={
                "heart_rate_bpm": 60 + index,
                "hrv_ms": 30 + index,
                "gsr_raw": 1000 + index,
                "signal_quality": "good",
                "source": "wearable",
            },
            headers=headers,
        )
        assert reading_response.status_code == 201

    calibration_response = client.post(
        "/api/baseline/calibrate?lookback_minutes=10",
        headers=headers,
    )

    assert calibration_response.status_code == 200
    calibration = calibration_response.json()
    assert calibration["success"] is True
    assert calibration["baseline"]["sample_count"] == 10
    assert calibration["baseline"]["heart_rate_bpm"] == 64.5
    assert calibration["baseline"]["hrv_ms"] == 34.5
    assert calibration["baseline"]["gsr_raw"] == 1004.5

    get_response = client.get("/api/baseline", headers=headers)

    assert get_response.status_code == 200
    baseline_body = get_response.json()
    assert baseline_body["available"] is True
    assert baseline_body["baseline"]["sample_count"] == 10
    assert baseline_body["baseline"]["heart_rate_bpm"] == 64.5


def test_haptic_setting_is_saved_and_retrieved(client):
    """Haptic preference should persist through the settings API."""
    headers = {"X-Demo-User": "api-test-haptic-settings"}

    initial = client.get("/api/settings", headers=headers)
    assert initial.status_code == 200
    assert initial.json()["haptic_enabled"] is False

    update = client.put(
        "/api/settings/haptic",
        json={"haptic_enabled": True},
        headers=headers,
    )
    assert update.status_code == 200
    assert update.json()["success"] is True
    assert update.json()["haptic_enabled"] is True

    saved = client.get("/api/settings", headers=headers)
    assert saved.status_code == 200
    assert saved.json()["haptic_enabled"] is True


def test_haptic_command_can_be_created_polled_and_acknowledged(client):
    """Verify the backend haptic command queue lifecycle end to end."""
    user_headers = {"X-Demo-User": "api-test-haptic-flow"}
    device_id = "test-device-haptic-flow"

    pair = client.post(
        "/api/devices/pair",
        json={
            "device_id": device_id,
            "device_name": "Haptic Flow Test",
        },
        headers=user_headers,
    )
    assert pair.status_code == 201
    token = pair.json()["device_token"]

    setting = client.put(
        "/api/settings/haptic",
        json={"haptic_enabled": True},
        headers=user_headers,
    )
    assert setting.status_code == 200

    create = client.post(
        "/api/haptic/commands",
        json={"device_id": device_id, "duration_ms": 500},
        headers=user_headers,
    )
    assert create.status_code == 201
    command = create.json()
    assert command["success"] is True
    assert command["status"] == "pending"
    assert command["duration_ms"] == 500

    poll = client.get(
        f"/api/devices/{device_id}/commands",
        headers={"X-Device-Token": token},
    )
    assert poll.status_code == 200
    commands = poll.json()["commands"]
    assert any(item["command_id"] == command["command_id"] for item in commands)

    acknowledge = client.post(
        f"/api/devices/{device_id}/commands/{command['command_id']}/ack",
        headers={"X-Device-Token": token},
    )
    assert acknowledge.status_code == 200
    assert acknowledge.json()["success"] is True
    assert acknowledge.json()["status"] == "acknowledged"


def test_haptic_command_is_blocked_when_preference_is_off(client):
    """The backend must not queue haptic commands while the setting is OFF."""
    headers = {"X-Demo-User": "api-test-haptic-disabled"}
    device_id = "test-device-haptic-disabled"

    pair = client.post(
        "/api/devices/pair",
        json={
            "device_id": device_id,
            "device_name": "Haptic Disabled Test",
        },
        headers=headers,
    )
    assert pair.status_code == 201

    response = client.post(
        "/api/haptic/commands",
        json={"device_id": device_id, "duration_ms": 500},
        headers=headers,
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Haptic feedback is OFF in user settings."


def test_recovery_session_is_saved_and_returned_in_history(client):
    """A completed recovery session should be saved and listed in history."""
    headers = {"X-Demo-User": "api-test-recovery-history"}
    started = datetime.now(timezone.utc) - timedelta(seconds=90)
    completed = started + timedelta(seconds=60)

    create = client.post(
        "/api/recovery/sessions",
        json={
            "activity": "breathing",
            "started_at": started.isoformat(),
            "completed_at": completed.isoformat(),
            "duration_seconds": 60,
            "completed": True,
        },
        headers=headers,
    )

    assert create.status_code == 201
    body = create.json()
    assert body["success"] is True
    assert body["session"]["activity"] == "breathing"
    assert body["session"]["duration_seconds"] == 60

    history = client.get("/api/recovery/history", headers=headers)
    assert history.status_code == 200
    assert history.json()["total"] == 1
    assert history.json()["sessions"][0]["session_id"] == body["session"]["session_id"]


def test_dashboard_overview_returns_empty_user_state(client):
    """Dashboard overview should return a clean waiting state for a new user."""
    response = client.get(
        "/api/dashboard/overview",
        headers={"X-Demo-User": "api-test-dashboard-empty"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["device_connected"] is False
    assert body["latest_reading"] is None
    assert body["stress"]["status"] == "waiting_for_wearable_data"
    assert body["stress"]["stress_score"] is None
    assert body["haptic_enabled"] is False


def test_latest_stress_waits_until_wearable_data_exists(client):
    """Stress endpoint should not invent a score before wearable data arrives."""
    response = client.get(
        "/api/stress/latest",
        headers={"X-Demo-User": "api-test-stress-empty"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "waiting_for_data"
    assert body["stress_score"] is None


def test_summary_and_high_stress_history_endpoints_return_empty_lists(client):
    """Summary and high-stress event history should return empty results for a new user."""
    headers = {"X-Demo-User": "api-test-empty-histories"}

    summaries = client.get("/api/summaries", headers=headers)
    assert summaries.status_code == 200
    assert summaries.json()["total"] == 0
    assert summaries.json()["summaries"] == []

    events = client.get("/api/events/high-stress", headers=headers)
    assert events.status_code == 200
    assert events.json()["total"] == 0
    assert events.json()["events"] == []


def test_face_scan_status_enforces_preview_only_privacy_state(client):
    """Face scan status must disclose that analysis is not integrated."""
    response = client.get("/api/face-scan/status")

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["status"] == "analysis_not_integrated"
    assert body["image_storage"] is False
    assert "OFF by default" in body["camera_policy"]


def test_sensor_data_is_isolated_between_users(client):
    """One development user's sensor readings must not appear for another user."""
    user_a = {"X-Demo-User": "api-test-user-a"}
    user_b = {"X-Demo-User": "api-test-user-b"}

    create = client.post(
        "/api/sensors",
        json={
            "heart_rate_bpm": 80,
            "hrv_ms": 40,
            "gsr_raw": 1750,
            "signal_quality": "good",
            "source": "simulated",
        },
        headers=user_a,
    )
    assert create.status_code == 201

    user_a_latest = client.get("/api/sensors/latest", headers=user_a)
    assert user_a_latest.status_code == 200
    assert user_a_latest.json()["data"] is not None

    user_b_latest = client.get("/api/sensors/latest", headers=user_b)
    assert user_b_latest.status_code == 200
    assert user_b_latest.json()["data"] is None

    user_b_history = client.get("/api/sensors/history", headers=user_b)
    assert user_b_history.status_code == 200
    assert user_b_history.json()["total"] == 0


def test_invalid_demo_user_identifier_is_rejected(client):
    """Development user identifiers must not contain spaces or special characters."""
    response = client.get(
        "/api/sensors/latest",
        headers={"X-Demo-User": "invalid user!"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid development user ID."


def test_device_cannot_be_paired_to_a_second_user(client):
    """An already-owned device ID must not be reassigned to another account."""
    device_id = "test-device-cross-user-01"

    first = client.post(
        "/api/devices/pair",
        json={"device_id": device_id, "device_name": "Owner A Device"},
        headers={"X-Demo-User": "api-test-device-owner-a"},
    )
    assert first.status_code == 201

    second = client.post(
        "/api/devices/pair",
        json={"device_id": device_id, "device_name": "Owner B Attempt"},
        headers={"X-Demo-User": "api-test-device-owner-b"},
    )

    assert second.status_code == 409


def test_user_cannot_unpair_another_users_device(client):
    """A user must not be able to deactivate a device owned by another user."""
    device_id = "test-device-cross-unpair-01"

    first = client.post(
        "/api/devices/pair",
        json={"device_id": device_id, "device_name": "Owner Device"},
        headers={"X-Demo-User": "api-test-unpair-owner"},
    )
    assert first.status_code == 201

    second = client.delete(
        f"/api/devices/{device_id}",
        headers={"X-Demo-User": "api-test-unpair-other"},
    )

    assert second.status_code == 404

