"""
NeuroSense - Backend API Tests

Run from the project root:
    python -m pytest backend/tests/test_main.py -v

Tests use an isolated temporary SQLite database. They do not modify
the developer's local NeuroSense database or require real hardware.
"""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from backend import main as api


class FakeFirestoreSnapshot:
    def __init__(self, document_id, payload):
        self.id = document_id
        self._payload = payload

    def to_dict(self):
        return dict(self._payload)


class FakeFirestoreCollection:
    def __init__(self, documents, operations):
        self.documents = documents
        self.operations = operations

    def collection(self, name):
        self.operations.append(("collection", name))
        return self

    def document(self, name):
        self.operations.append(("document", name))
        return self

    def order_by(self, field, direction=None):
        self.operations.append(("order_by", field, direction))
        return self

    def limit(self, count):
        self.operations.append(("limit", count))
        return self

    def stream(self):
        self.operations.append(("stream",))
        return self.documents


class FakeFirestoreClient:
    def __init__(self, documents):
        self.documents = documents
        self.operations = []

    def collection(self, name):
        self.operations.append(("collection", name))
        return FakeFirestoreCollection(
            self.documents,
            self.operations
        )


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Start the API with an isolated test database and no cloud services."""
    monkeypatch.setattr(api, "DB_PATH", tmp_path / "neurosense_test.db")
    monkeypatch.setattr(api, "APP_ENV", "development")
    monkeypatch.setattr(api, "FIREBASE_APP", None)
    monkeypatch.setattr(api, "FIRESTORE_CLIENT", None)
    monkeypatch.setattr(api, "STRESS_MODEL", None)

    # Keep API tests independent from external Firebase credentials
    # and from loading a locally trained model into the test process.
    monkeypatch.setattr(api, "initialize_firebase", lambda: None)
    monkeypatch.setattr(api, "initialize_ml_model", lambda: None)

    with TestClient(api.app) as test_client:
        yield test_client


def test_application_starts(client):
    """Check that the FastAPI application responds."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_health_endpoint(client):
    """Check backend and isolated database health."""
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "healthy"
    assert body["database"] == "healthy"


def test_system_status_endpoint(client):
    """Check system status endpoint."""
    response = client.get("/api/system/status")

    assert response.status_code == 200
    body = response.json()

    assert body["backend"] == "online"
    assert body["database"] == "sqlite"


def test_firestore_collection_reader_accepts_user_subcollection_path(monkeypatch):
    """Firestore collection paths have odd segment counts: collection/document/collection."""
    payload = {
        "id": "cloud-reading-1",
        "timestamp": "2026-11-08T10:00:00+00:00",
        "source": "wearable",
        "heart_rate_bpm": 74.0,
    }
    fake_document = FakeFirestoreSnapshot("cloud-reading-1", payload)
    fake_client = FakeFirestoreClient([fake_document])
    monkeypatch.setattr(api, "FIRESTORE_CLIENT", fake_client)

    rows = api.firestore_read_collection(
        ("users", "firebase-user-1", "sensor_readings"),
        limit=25,
    )

    assert rows == [{**payload, "document_id": "cloud-reading-1"}]
    assert fake_client.operations[:3] == [
        ("collection", "users"),
        ("document", "firebase-user-1"),
        ("collection", "sensor_readings"),
    ]
    assert ("limit", 25) in fake_client.operations
    assert ("stream",) in fake_client.operations


def test_sensor_history_restores_firestore_backup_when_sqlite_is_empty(
    client,
    monkeypatch,
):
    """Sensor history should return a user's Firestore backup after an empty SQLite store."""
    user_id = "firestore-backup-history-user"
    cloud_reading = {
        "id": "firestore-reading-1",
        "timestamp": "2026-11-08T10:00:00+00:00",
        "received_at": "2026-11-08T10:00:01+00:00",
        "source": "wearable",
        "signal_quality": "good",
        "heart_rate_bpm": 73.0,
        "hrv_ms": 41.0,
        "gsr_raw": 1800,
        "skin_temperature_c": 33.1,
        "ibi_ms": [810.0, 820.0],
    }
    fake_document = FakeFirestoreSnapshot(
        "firestore-reading-1",
        cloud_reading,
    )
    fake_client = FakeFirestoreClient([fake_document])
    monkeypatch.setattr(api, "FIRESTORE_CLIENT", fake_client)

    response = client.get(
        "/api/sensors/history?limit=10",
        headers={"X-Demo-User": user_id},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["readings"][0]["id"] == "firestore-reading-1"
    assert body["readings"][0]["heart_rate_bpm"] == 73.0
    assert ("document", user_id) in fake_client.operations
    assert ("collection", "sensor_readings") in fake_client.operations


def test_public_firebase_config_is_unavailable_without_configuration(client):
    """Public config must not expose unconfigured or private Firebase credentials."""
    response = client.get("/api/public-config")

    assert response.status_code == 200
    body = response.json()

    assert body["firebase_web_configured"] is False
    assert body["firebase_auth_backend_ready"] is False
    assert body["firebase_sign_in_ready"] is False
    assert body["firebase"] is None
    assert "FIREBASE_SERVICE_ACCOUNT_JSON" not in response.text


def test_public_firebase_config_exposes_only_browser_fields_when_ready(client, monkeypatch):
    """Browser config is public, but Admin service-account credentials never are."""
    monkeypatch.setattr(
        api,
        "FIREBASE_WEB_CONFIG",
        {
            "apiKey": "public-web-key",
            "authDomain": "neurosense-test.firebaseapp.com",
            "projectId": "neurosense-test",
            "appId": "1:123:web:abc",
            "messagingSenderId": "123",
        },
    )
    monkeypatch.setattr(api, "FIREBASE_APP", object())

    response = client.get("/api/public-config")

    assert response.status_code == 200
    body = response.json()
    assert body["firebase_web_configured"] is True
    assert body["firebase_auth_backend_ready"] is True
    assert body["firebase_sign_in_ready"] is True
    assert body["firebase"]["apiKey"] == "public-web-key"
    assert set(body["firebase"]) == {
        "apiKey", "authDomain", "projectId", "appId", "messagingSenderId"
    }
    assert "serviceAccount" not in response.text
    assert "private_key" not in response.text


@pytest.mark.parametrize(
    ("path", "method"),
    [
        ("/", "get"),
        ("/health", "get"),
        ("/api/system/status", "get"),
        ("/api/public-config", "get"),
        ("/api/devices/pair", "post"),
        ("/api/devices", "get"),
        ("/api/devices/{device_id}", "delete"),
        ("/api/sensors", "post"),
        ("/api/devices/{device_id}/readings", "post"),
        ("/api/sensors/latest", "get"),
        ("/api/sensors/history", "get"),
        ("/api/device/status", "get"),
        ("/api/baseline/calibrate", "post"),
        ("/api/baseline", "get"),
        ("/api/stress/latest", "get"),
        ("/api/dashboard/overview", "get"),
        ("/api/settings", "get"),
        ("/api/settings/haptic", "put"),
        ("/api/haptic/commands", "post"),
        ("/api/devices/{device_id}/commands", "get"),
        ("/api/devices/{device_id}/commands/{command_id}/ack", "post"),
        ("/api/recovery/sessions", "post"),
        ("/api/recovery/history", "get"),
        ("/api/summaries", "get"),
        ("/api/events/high-stress", "get"),
        ("/api/face-scan/status", "get"),
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
    """An empty sensor payload should fail Pydantic validation."""
    response = client.post(
        "/api/sensors",
        json={},
        headers={"X-Demo-User": "neurosense-test-user"},
    )

    assert response.status_code == 422


def test_sensor_post_accepts_valid_simulated_reading_and_persists_it(client):
    """API Test 1: valid sensor POST returns 201 and stores the reading."""
    headers = {"X-Demo-User": "neurosense-api-test-user"}
    payload = {
        "heart_rate_bpm": 76,
        "hrv_ms": 42,
        "gsr_raw": 1850,
        "skin_temperature_c": 33.2,
        "signal_quality": "moderate",
        "source": "simulated",
    }

    # The isolated database should start empty for this test.
    before = client.get("/api/sensors/latest", headers=headers)
    assert before.status_code == 200
    assert before.json()["data"] is None

    response = client.post(
        "/api/sensors",
        json=payload,
        headers=headers,
    )

    assert response.status_code == 201
    body = response.json()

    assert body["success"] is True
    assert body["reading"]["heart_rate_bpm"] == 76
    assert body["reading"]["hrv_ms"] == 42
    assert body["reading"]["gsr_raw"] == 1850
    assert body["reading"]["skin_temperature_c"] == 33.2
    assert body["reading"]["source"] == "simulated"
    assert body["stress"]["status"] == "simulated_data"

    # Confirm the same reading can be retrieved from SQLite.
    latest = client.get("/api/sensors/latest", headers=headers)
    assert latest.status_code == 200
    saved = latest.json()["data"]

    assert saved is not None
    assert saved["id"] == body["reading"]["id"]
    assert saved["heart_rate_bpm"] == 76
    assert saved["gsr_raw"] == 1850

    history = client.get("/api/sensors/history", headers=headers)
    assert history.status_code == 200
    assert history.json()["total"] == 1
    assert history.json()["readings"][0]["id"] == saved["id"]


def test_recovery_session_rejects_completion_before_start(client):
    """Recovery sessions must not complete before their start time."""
    response = client.post(
        "/api/recovery/sessions",
        json={
            "activity": "breathing",
            "started_at": "2026-11-08T09:00:00+00:00",
            "completed_at": "2026-11-08T08:59:00+00:00",
            "duration_seconds": 60,
            "completed": True,
        },
        headers={"X-Demo-User": "api-test-recovery-invalid-time"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "completed_at cannot be earlier than started_at."


def test_recovery_session_returns_before_after_comparison(client):
    """Recovery API should compare wearable readings around the session."""
    headers = {"X-Demo-User": "api-test-recovery-comparison"}
    now = datetime.now(timezone.utc)
    started = now - timedelta(seconds=90)
    completed = now - timedelta(seconds=30)
    before_time = started - timedelta(seconds=15)
    after_time = completed + timedelta(seconds=15)

    before = client.post(
        "/api/sensors",
        json={
            "timestamp": before_time.isoformat(),
            "heart_rate_bpm": 82,
            "hrv_ms": 32,
            "gsr_raw": 1900,
            "skin_temperature_c": 33.2,
            "signal_quality": "good",
            "source": "wearable",
        },
        headers=headers,
    )
    after = client.post(
        "/api/sensors",
        json={
            "timestamp": after_time.isoformat(),
            "heart_rate_bpm": 75,
            "hrv_ms": 40,
            "gsr_raw": 1600,
            "skin_temperature_c": 33.0,
            "signal_quality": "good",
            "source": "wearable",
        },
        headers=headers,
    )
    assert before.status_code == 201
    assert after.status_code == 201

    response = client.post(
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

    assert response.status_code == 201
    comparison = response.json()["comparison"]
    assert comparison is not None
    assert comparison["heart_rate_bpm"]["before"] == 82
    assert comparison["heart_rate_bpm"]["after"] == 75
    assert comparison["heart_rate_bpm"]["change"] == -7
    assert comparison["hrv_ms"]["change"] == 8
    assert comparison["gsr_raw"]["change"] == -300


def test_due_summary_generation_is_visible_in_summary_history(client):
    """A completed three-hour window should appear through the summaries API."""
    headers = {"X-Demo-User": "api-test-three-hour-summary"}
    old_timestamp = datetime.now(timezone.utc) - timedelta(hours=7)

    reading = client.post(
        "/api/sensors",
        json={
            "timestamp": old_timestamp.isoformat(),
            "heart_rate_bpm": 80,
            "hrv_ms": 36,
            "gsr_raw": 1700,
            "skin_temperature_c": 33.1,
            "signal_quality": "good",
            "source": "wearable",
        },
        headers=headers,
    )
    assert reading.status_code == 201

    created_count = api.create_due_summaries()
    assert created_count >= 1

    response = client.get("/api/summaries", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1
    assert body["summaries"][0]["reading_count"] == 1
    assert body["summaries"][0]["reliable_reading_count"] == 1
    assert body["summaries"][0]["avg_heart_rate"] == 80
    assert body["summaries"][0]["avg_hrv"] == 36
    assert body["summaries"][0]["avg_gsr"] == 1700




def test_sensor_post_rejects_out_of_range_heart_rate(client):
    """API Test 2a: reject physiological values outside the schema range."""
    response = client.post(
        "/api/sensors",
        json={
            "heart_rate_bpm": 221,
            "source": "simulated",
            "signal_quality": "good",
        },
        headers={"X-Demo-User": "neurosense-validation-user"},
    )

    assert response.status_code == 422


def test_sensor_post_rejects_timestamp_too_far_in_future(client):
    """API Test 2b: reject readings more than five minutes ahead."""
    from datetime import datetime, timedelta, timezone

    future_timestamp = (
        datetime.now(timezone.utc) + timedelta(minutes=10)
    ).isoformat()

    response = client.post(
        "/api/sensors",
        json={
            "timestamp": future_timestamp,
            "heart_rate_bpm": 72,
            "source": "simulated",
            "signal_quality": "good",
        },
        headers={"X-Demo-User": "neurosense-validation-user"},
    )

    assert response.status_code == 422
    assert "future" in response.json()["detail"].lower()


def test_sensor_post_rejects_reading_older_than_24_hours(client):
    """API Test 2c: reject stale sensor readings."""
    from datetime import datetime, timedelta, timezone

    old_timestamp = (
        datetime.now(timezone.utc) - timedelta(hours=25)
    ).isoformat()

    response = client.post(
        "/api/sensors",
        json={
            "timestamp": old_timestamp,
            "heart_rate_bpm": 72,
            "source": "simulated",
            "signal_quality": "good",
        },
        headers={"X-Demo-User": "neurosense-validation-user"},
    )

    assert response.status_code == 422
    assert "24-hour" in response.json()["detail"]


def test_sensor_readings_are_isolated_between_demo_users(client):
    """API Test 2d: one demo user must not see another user's readings."""
    alice_headers = {"X-Demo-User": "neurosense-alice"}
    bob_headers = {"X-Demo-User": "neurosense-bob"}

    created = client.post(
        "/api/sensors",
        json={
            "heart_rate_bpm": 74,
            "hrv_ms": 40,
            "gsr_raw": 1700,
            "source": "simulated",
            "signal_quality": "moderate",
        },
        headers=alice_headers,
    )

    assert created.status_code == 201

    alice_latest = client.get(
        "/api/sensors/latest",
        headers=alice_headers,
    )
    bob_latest = client.get(
        "/api/sensors/latest",
        headers=bob_headers,
    )

    assert alice_latest.status_code == 200
    assert alice_latest.json()["data"] is not None
    assert bob_latest.status_code == 200
    assert bob_latest.json()["data"] is None



def pair_test_device(client, user_id, device_id="neurosense-test-esp32"):
    """Pair a test device and return its one-time token without logging it."""
    response = client.post(
        "/api/devices/pair",
        json={
            "device_id": device_id,
            "device_name": "NeuroSense Test Wearable",
        },
        headers={"X-Demo-User": user_id},
    )
    assert response.status_code == 201
    return response.json()


def test_device_pairing_returns_one_time_token_and_lists_device(client):
    """API Test 3a: pairing creates an active device and returns a token."""
    user_id = "neurosense-device-pair-user"
    paired = pair_test_device(client, user_id)

    assert paired["success"] is True
    assert paired["device_id"] == "neurosense-test-esp32"
    assert isinstance(paired["device_token"], str)
    assert len(paired["device_token"]) >= 24

    listed = client.get(
        "/api/devices",
        headers={"X-Demo-User": user_id},
    )
    assert listed.status_code == 200
    devices = listed.json()["devices"]

    assert len(devices) == 1
    assert devices[0]["device_id"] == paired["device_id"]
    assert devices[0]["active"] == 1
    # The list endpoint must never return the device token or its hash.
    assert "device_token" not in devices[0]
    assert "token_hash" not in devices[0]


def test_device_reading_requires_valid_x_device_token(client):
    """API Test 3b: missing/invalid token is rejected; valid token is accepted."""
    user_id = "neurosense-device-auth-user"
    device_id = "neurosense-test-esp32"
    paired = pair_test_device(client, user_id, device_id)

    payload = {
        "heart_rate_bpm": 78,
        "hrv_ms": 38,
        "gsr_raw": 1760,
        "skin_temperature_c": 33.1,
        "signal_quality": "good",
        "source": "wearable",
    }
    path = f"/api/devices/{device_id}/readings"

    missing_token = client.post(path, json=payload)
    assert missing_token.status_code == 401

    wrong_token = client.post(
        path,
        json=payload,
        headers={"X-Device-Token": "not-a-valid-device-token-000"},
    )
    assert wrong_token.status_code == 401

    # The backend requires X-Device-Token, not Authorization.
    wrong_header = client.post(
        path,
        json=payload,
        headers={"Authorization": paired["device_token"]},
    )
    assert wrong_header.status_code == 401

    accepted = client.post(
        path,
        json=payload,
        headers={"X-Device-Token": paired["device_token"]},
    )
    assert accepted.status_code == 201
    body = accepted.json()

    assert body["success"] is True
    assert body["reading"]["device_id"] == device_id
    assert body["reading"]["source"] == "wearable"

    status_response = client.get(
        "/api/device/status",
        headers={"X-Demo-User": user_id},
    )
    assert status_response.status_code == 200
    assert status_response.json()["connected"] is True


def test_device_cannot_be_paired_to_two_different_users(client):
    """API Test 3c: a device already owned by one user cannot be reassigned."""
    device_id = "neurosense-shared-test-device"

    first = pair_test_device(
        client,
        "neurosense-device-owner-a",
        device_id,
    )
    assert first["success"] is True

    second = client.post(
        "/api/devices/pair",
        json={
            "device_id": device_id,
            "device_name": "Attempted second owner",
        },
        headers={"X-Demo-User": "neurosense-device-owner-b"},
    )

    assert second.status_code == 409


def test_unpaired_device_token_can_no_longer_upload_readings(client):
    """API Test 3d: unpairing deactivates the device token."""
    user_id = "neurosense-device-unpair-user"
    device_id = "neurosense-unpair-test-device"
    paired = pair_test_device(client, user_id, device_id)

    removed = client.delete(
        f"/api/devices/{device_id}",
        headers={"X-Demo-User": user_id},
    )
    assert removed.status_code == 200
    assert removed.json()["active"] is False

    upload = client.post(
        f"/api/devices/{device_id}/readings",
        json={
            "heart_rate_bpm": 75,
            "gsr_raw": 1650,
            "source": "wearable",
            "signal_quality": "good",
        },
        headers={"X-Device-Token": paired["device_token"]},
    )
    assert upload.status_code == 401

    device_status = client.get(
        "/api/device/status",
        headers={"X-Demo-User": user_id},
    )
    assert device_status.status_code == 200
    assert device_status.json()["connected"] is False



def post_wearable_reading(
    client,
    user_id,
    heart_rate_bpm=72,
    hrv_ms=40,
    gsr_raw=1600,
    skin_temperature_c=33.0,
    signal_quality="good",
    timestamp=None,
):
    """Create one clearly labelled wearable test reading."""
    from datetime import datetime, timezone

    payload = {
        "heart_rate_bpm": heart_rate_bpm,
        "hrv_ms": hrv_ms,
        "gsr_raw": gsr_raw,
        "skin_temperature_c": skin_temperature_c,
        "signal_quality": signal_quality,
        "source": "wearable",
    }
    if timestamp is not None:
        payload["timestamp"] = timestamp
    else:
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()

    return client.post(
        "/api/sensors",
        json=payload,
        headers={"X-Demo-User": user_id},
    )


def add_ten_wearable_readings(client, user_id):
    """Provide the minimum 10 recent reliable samples for baseline tests."""
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    responses = []

    for index in range(10):
        timestamp = (now - timedelta(seconds=index)).isoformat()
        response = post_wearable_reading(
            client,
            user_id,
            heart_rate_bpm=70 + index,
            hrv_ms=40 + index,
            gsr_raw=1500 + (index * 10),
            skin_temperature_c=33.0 + (index * 0.1),
            timestamp=timestamp,
        )
        responses.append(response)

    assert all(response.status_code == 201 for response in responses)


def test_baseline_calibration_rejects_fewer_than_ten_readings(client):
    """API Test 4a: baseline requires at least 10 reliable wearable readings."""
    user_id = "neurosense-baseline-insufficient"
    post_wearable_reading(client, user_id)

    response = client.post(
        "/api/baseline/calibrate",
        headers={"X-Demo-User": user_id},
    )

    assert response.status_code == 422
    assert "at least 10" in response.json()["detail"].lower()


def test_baseline_calibration_saves_and_returns_personal_baseline(client):
    """API Test 4b: ten recent readings produce a retrievable baseline."""
    user_id = "neurosense-baseline-success"
    add_ten_wearable_readings(client, user_id)

    calibrated = client.post(
        "/api/baseline/calibrate",
        headers={"X-Demo-User": user_id},
    )

    assert calibrated.status_code == 200
    assert calibrated.json()["success"] is True
    baseline = calibrated.json()["baseline"]
    assert baseline["sample_count"] == 10
    assert baseline["heart_rate_bpm"] is not None
    assert baseline["hrv_ms"] is not None
    assert baseline["gsr_raw"] is not None

    fetched = client.get(
        "/api/baseline",
        headers={"X-Demo-User": user_id},
    )
    assert fetched.status_code == 200
    assert fetched.json()["available"] is True
    assert fetched.json()["baseline"]["sample_count"] == 10


def test_stress_prediction_api_uses_baseline_and_records_high_stress_event(
    client,
    monkeypatch,
):
    """API Test 4c: exercise stress API contract with deterministic mock inference."""
    user_id = "neurosense-stress-api-user"
    add_ten_wearable_readings(client, user_id)

    calibrated = client.post(
        "/api/baseline/calibrate",
        headers={"X-Demo-User": user_id},
    )
    assert calibrated.status_code == 200

    class DeterministicStressModel:
        classes_ = ["baseline", "stress"]

        def predict_proba(self, matrix):
            assert matrix.shape == (1, 8)
            return [[0.20, 0.80]]

    monkeypatch.setattr(
        api,
        "STRESS_MODEL",
        DeterministicStressModel(),
    )

    reading = post_wearable_reading(
        client,
        user_id,
        heart_rate_bpm=92,
        hrv_ms=24,
        gsr_raw=2300,
        skin_temperature_c=33.5,
    )
    assert reading.status_code == 201

    stress = client.get(
        "/api/stress/latest",
        headers={"X-Demo-User": user_id},
    )
    assert stress.status_code == 200
    body = stress.json()

    assert body["status"] == "prediction_available"
    assert body["stress_score"] == 80.0
    assert body["stress_probability"] == 0.8
    assert body["model_validated"] is False

    events = client.get(
        "/api/events/high-stress",
        headers={"X-Demo-User": user_id},
    )
    assert events.status_code == 200
    assert events.json()["total"] == 1
    assert events.json()["events"][0]["stress_probability"] == 0.8



def test_haptic_setting_can_be_enabled_disabled_and_retrieved(client):
    """API Test 5a: haptic preference is saved per user."""
    user_id = "neurosense-haptic-settings-user"
    headers = {"X-Demo-User": user_id}

    initial = client.get("/api/settings", headers=headers)
    assert initial.status_code == 200
    assert initial.json()["haptic_enabled"] is False

    enabled = client.put(
        "/api/settings/haptic",
        json={"haptic_enabled": True},
        headers=headers,
    )
    assert enabled.status_code == 200
    assert enabled.json()["haptic_enabled"] is True
    assert "cloud_sync" in enabled.json()

    fetched_enabled = client.get("/api/settings", headers=headers)
    assert fetched_enabled.status_code == 200
    assert fetched_enabled.json()["haptic_enabled"] is True

    disabled = client.put(
        "/api/settings/haptic",
        json={"haptic_enabled": False},
        headers=headers,
    )
    assert disabled.status_code == 200
    assert disabled.json()["haptic_enabled"] is False

    fetched_disabled = client.get("/api/settings", headers=headers)
    assert fetched_disabled.status_code == 200
    assert fetched_disabled.json()["haptic_enabled"] is False


def test_haptic_command_is_rejected_when_preference_is_off(client):
    """API Test 5b: the server must not queue haptics while setting is OFF."""
    user_id = "neurosense-haptic-off-user"
    device_id = "neurosense-haptic-off-device"
    pair_test_device(client, user_id, device_id)

    response = client.post(
        "/api/haptic/commands",
        json={
            "device_id": device_id,
            "duration_ms": 500,
        },
        headers={"X-Demo-User": user_id},
    )

    assert response.status_code == 409
    assert "OFF" in response.json()["detail"]


def test_haptic_command_can_be_polled_and_acknowledged(client):
    """API Test 5c: enabled haptic command follows pending-delivered-acknowledged."""
    user_id = "neurosense-haptic-command-user"
    device_id = "neurosense-haptic-command-device"
    paired = pair_test_device(client, user_id, device_id)
    token = paired["device_token"]

    enabled = client.put(
        "/api/settings/haptic",
        json={"haptic_enabled": True},
        headers={"X-Demo-User": user_id},
    )
    assert enabled.status_code == 200
    assert enabled.json()["haptic_enabled"] is True

    queued = client.post(
        "/api/haptic/commands",
        json={
            "device_id": device_id,
            "duration_ms": 650,
        },
        headers={"X-Demo-User": user_id},
    )
    assert queued.status_code == 201
    queued_body = queued.json()

    assert queued_body["success"] is True
    assert queued_body["status"] == "pending"
    assert queued_body["duration_ms"] == 650

    command_path = f"/api/devices/{device_id}/commands"

    unauthorized = client.get(command_path)
    assert unauthorized.status_code == 401

    polled = client.get(
        command_path,
        headers={"X-Device-Token": token},
    )
    assert polled.status_code == 200
    commands = polled.json()["commands"]
    assert len(commands) == 1
    assert commands[0]["command_id"] == queued_body["command_id"]
    assert commands[0]["duration_ms"] == 650

    ack = client.post(
        f"{command_path}/{queued_body['command_id']}/ack",
        headers={"X-Device-Token": token},
    )
    assert ack.status_code == 200
    assert ack.json()["success"] is True
    assert ack.json()["status"] == "acknowledged"

    polled_again = client.get(
        command_path,
        headers={"X-Device-Token": token},
    )
    assert polled_again.status_code == 200
    assert polled_again.json()["commands"] == []


def test_haptic_command_rejects_duration_outside_safe_api_range(client):
    """API Test 5d: haptic command duration is limited by request validation."""
    user_id = "neurosense-haptic-duration-user"
    device_id = "neurosense-haptic-duration-device"
    pair_test_device(client, user_id, device_id)

    client.put(
        "/api/settings/haptic",
        json={"haptic_enabled": True},
        headers={"X-Demo-User": user_id},
    )

    response = client.post(
        "/api/haptic/commands",
        json={
            "device_id": device_id,
            "duration_ms": 50,
        },
        headers={"X-Demo-User": user_id},
    )

    assert response.status_code == 422


def test_dashboard_overview_returns_latest_device_reading(client):
    """Dashboard API should surface a recently uploaded wearable reading."""
    user_id = "neurosense-dashboard-integration-user"
    device_id = "neurosense-dashboard-esp32"
    paired = pair_test_device(client, user_id, device_id)

    upload = client.post(
        f"/api/devices/{device_id}/readings",
        json={
            "heart_rate_bpm": 79,
            "hrv_ms": 41,
            "gsr_raw": 1810,
            "skin_temperature_c": 33.1,
            "signal_quality": "good",
            "source": "wearable",
        },
        headers={"X-Device-Token": paired["device_token"]},
    )
    assert upload.status_code == 201

    response = client.get(
        "/api/dashboard/overview",
        headers={"X-Demo-User": user_id},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["device_connected"] is True
    assert body["latest_reading"]["device_id"] == device_id
    assert body["latest_reading"]["heart_rate_bpm"] == 79
    assert body["latest_reading"]["hrv_ms"] == 41
    assert body["stress"]["status"] == "model_not_available"
    assert body["stress"]["stress_score"] is None
    assert body["haptic_enabled"] is False


def test_haptic_setting_can_be_enabled_and_disabled(client):
    """Haptic preference should persist both ON and OFF states."""
    headers = {"X-Demo-User": "neurosense-haptic-settings-user"}

    initial = client.get("/api/settings", headers=headers)
    assert initial.status_code == 200
    assert initial.json()["haptic_enabled"] is False

    enabled = client.put(
        "/api/settings/haptic",
        json={"haptic_enabled": True},
        headers=headers,
    )
    assert enabled.status_code == 200
    assert enabled.json()["haptic_enabled"] is True

    fetched_enabled = client.get("/api/settings", headers=headers)
    assert fetched_enabled.status_code == 200
    assert fetched_enabled.json()["haptic_enabled"] is True

    disabled = client.put(
        "/api/settings/haptic",
        json={"haptic_enabled": False},
        headers=headers,
    )
    assert disabled.status_code == 200
    assert disabled.json()["haptic_enabled"] is False

    fetched_disabled = client.get("/api/settings", headers=headers)
    assert fetched_disabled.status_code == 200
    assert fetched_disabled.json()["haptic_enabled"] is False


def test_haptic_command_is_created_polled_and_acknowledged(client):
    """Verify the complete backend command queue lifecycle for a paired device."""
    user_id = "neurosense-haptic-command-user"
    device_id = "neurosense-haptic-esp32"
    paired = pair_test_device(client, user_id, device_id)
    user_headers = {"X-Demo-User": user_id}

    enabled = client.put(
        "/api/settings/haptic",
        json={"haptic_enabled": True},
        headers=user_headers,
    )
    assert enabled.status_code == 200

    created = client.post(
        "/api/haptic/commands",
        json={"device_id": device_id, "duration_ms": 600},
        headers=user_headers,
    )
    assert created.status_code == 201
    command = created.json()
    assert command["success"] is True
    assert command["status"] == "pending"
    assert command["duration_ms"] == 600

    polled = client.get(
        f"/api/devices/{device_id}/commands",
        headers={"X-Device-Token": paired["device_token"]},
    )
    assert polled.status_code == 200
    assert any(
        item["command_id"] == command["command_id"]
        and item["duration_ms"] == 600
        for item in polled.json()["commands"]
    )

    acknowledged = client.post(
        f"/api/devices/{device_id}/commands/{command['command_id']}/ack",
        headers={"X-Device-Token": paired["device_token"]},
    )
    assert acknowledged.status_code == 200
    assert acknowledged.json()["success"] is True
    assert acknowledged.json()["status"] == "acknowledged"


def test_haptic_command_is_rejected_when_preference_is_off(client):
    """No haptic command should be queued when the user preference is OFF."""
    user_id = "neurosense-haptic-disabled-user"
    device_id = "neurosense-haptic-disabled-esp32"
    pair_test_device(client, user_id, device_id)

    response = client.post(
        "/api/haptic/commands",
        json={"device_id": device_id, "duration_ms": 500},
        headers={"X-Demo-User": user_id},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Haptic feedback is OFF in user settings."


def test_recovery_history_returns_saved_completed_session(client):
    """Completed recovery session should appear in the user's history endpoint."""
    user_id = "neurosense-recovery-history-user"
    headers = {"X-Demo-User": user_id}
    now = datetime.now(timezone.utc)
    started = now - timedelta(seconds=90)
    completed = now - timedelta(seconds=30)

    created = client.post(
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
    assert created.status_code == 201

    history = client.get("/api/recovery/history", headers=headers)
    assert history.status_code == 200
    body = history.json()
    assert body["total"] == 1
    assert body["sessions"][0]["session_id"] == created.json()["session"]["session_id"]
    assert body["sessions"][0]["activity"] == "breathing"
    assert body["sessions"][0]["completed"] == 1


def test_recovery_history_adds_after_reading_received_after_session_save(client):
    """History should find the first wearable reading received after a session completes."""
    headers = {"X-Demo-User": "api-test-recovery-late-after"}
    now = datetime.now(timezone.utc)
    started = now - timedelta(seconds=120)
    completed = now - timedelta(seconds=60)

    before_time = started - timedelta(seconds=10)
    after_time = completed + timedelta(seconds=10)

    before = client.post(
        "/api/sensors",
        json={
            "timestamp": before_time.isoformat(),
            "heart_rate_bpm": 84,
            "hrv_ms": 30,
            "gsr_raw": 2000,
            "skin_temperature_c": 33.4,
            "signal_quality": "good",
            "source": "wearable",
        },
        headers=headers,
    )
    assert before.status_code == 201

    created = client.post(
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
    assert created.status_code == 201
    assert created.json()["comparison"] is None

    after = client.post(
        "/api/sensors",
        json={
            "timestamp": after_time.isoformat(),
            "heart_rate_bpm": 76,
            "hrv_ms": 39,
            "gsr_raw": 1650,
            "skin_temperature_c": 33.1,
            "signal_quality": "good",
            "source": "wearable",
        },
        headers=headers,
    )
    assert after.status_code == 201

    history = client.get("/api/recovery/history", headers=headers)
    assert history.status_code == 200

    comparison = history.json()["sessions"][0]["comparison"]
    assert comparison["heart_rate_bpm"]["before"] == 84
    assert comparison["heart_rate_bpm"]["after"] == 76
    assert comparison["heart_rate_bpm"]["change"] == -8
    assert comparison["hrv_ms"]["change"] == 9
    assert comparison["gsr_raw"]["change"] == -350


def test_face_scan_status_reports_browser_local_landmark_privacy(client):
    """Face Scan API discloses local landmark processing without stress inference."""
    response = client.get("/api/face-scan/status")

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["status"] == "browser_local_landmark_detection_only"
    assert body["processing"] == "on_device_browser"
    assert body["raw_frames_uploaded"] is False
    assert body["image_storage"] is False
    assert body["emotion_inference"] is False
    assert body["stress_inference"] is False
    assert "OFF by default" in body["camera_policy"]




def test_dashboard_overview_has_safe_empty_state(client):
    """API Test 6a: dashboard does not invent readings when no sensor data exists."""
    user_id = "neurosense-dashboard-empty-user"
    response = client.get(
        "/api/dashboard/overview",
        headers={"X-Demo-User": user_id},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["device_connected"] is False
    assert body["latest_reading"] is None
    assert body["stress"]["status"] == "waiting_for_wearable_data"
    assert body["stress"]["stress_score"] is None
    assert body["haptic_enabled"] is False


def test_recovery_session_is_saved_and_returned_in_history(client):
    """API Test 6b: recovery session POST is visible through recovery history GET."""
    from datetime import datetime, timedelta, timezone

    user_id = "neurosense-recovery-history-user"
    now = datetime.now(timezone.utc)
    started = now - timedelta(seconds=75)

    saved = client.post(
        "/api/recovery/sessions",
        json={
            "activity": "breathing",
            "started_at": started.isoformat(),
            "completed_at": now.isoformat(),
            "duration_seconds": 75,
            "completed": True,
        },
        headers={"X-Demo-User": user_id},
    )

    assert saved.status_code == 201
    session_id = saved.json()["session"]["session_id"]

    history = client.get(
        "/api/recovery/history",
        headers={"X-Demo-User": user_id},
    )
    assert history.status_code == 200
    assert history.json()["total"] == 1
    assert history.json()["sessions"][0]["session_id"] == session_id
    assert history.json()["sessions"][0]["activity"] == "breathing"


def test_face_scan_status_enforces_local_landmark_privacy_contract(client):
    """API Test 6c: backend reports local landmarks and does not store images."""
    response = client.get("/api/face-scan/status")

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["status"] == "browser_local_landmark_detection_only"
    assert body["processing"] == "on_device_browser"
    assert body["raw_frames_uploaded"] is False
    assert body["image_storage"] is False
    assert body["emotion_inference"] is False
    assert body["stress_inference"] is False
    assert "OFF by default" in body["camera_policy"]


def test_invalid_demo_user_id_is_rejected(client):
    """API Test 6d: development user IDs cannot contain spaces or unsafe characters."""
    response = client.get(
        "/api/settings",
        headers={"X-Demo-User": "invalid user id"},
    )

    assert response.status_code == 400


def test_haptic_preferences_are_isolated_between_users(client):
    """API Test 6e: one user's haptic preference must not alter another's."""
    user_a = {"X-Demo-User": "neurosense-haptic-user-a"}
    user_b = {"X-Demo-User": "neurosense-haptic-user-b"}

    enabled = client.put(
        "/api/settings/haptic",
        json={"haptic_enabled": True},
        headers=user_a,
    )
    assert enabled.status_code == 200

    a_settings = client.get("/api/settings", headers=user_a)
    b_settings = client.get("/api/settings", headers=user_b)

    assert a_settings.status_code == 200
    assert b_settings.status_code == 200
    assert a_settings.json()["haptic_enabled"] is True
    assert b_settings.json()["haptic_enabled"] is False


def test_sensor_post_rejects_too_few_ibi_intervals(client):
    """API Test 6f: HRV input must include at least three IBI intervals."""
    response = client.post(
        "/api/sensors",
        json={
            "heart_rate_bpm": 75,
            "ibi_ms": [800, 810],
            "source": "simulated",
            "signal_quality": "good",
        },
        headers={"X-Demo-User": "neurosense-ibi-validation-user"},
    )

    assert response.status_code == 422


def test_haptic_command_rejects_unpaired_device(client):
    """API Test 6g: haptic commands can only target an active paired device."""
    user_id = "neurosense-haptic-unpaired-user"

    enabled = client.put(
        "/api/settings/haptic",
        json={"haptic_enabled": True},
        headers={"X-Demo-User": user_id},
    )
    assert enabled.status_code == 200

    response = client.post(
        "/api/haptic/commands",
        json={
            "device_id": "neurosense-never-paired",
            "duration_ms": 500,
        },
        headers={"X-Demo-User": user_id},
    )

    assert response.status_code == 404


def test_sensor_latest_returns_the_last_saved_reading(client):
    """GET /api/sensors/latest should return the user's most recent saved reading."""
    headers = {"X-Demo-User": "neurosense-sensor-latest-user"}

    first = client.post(
        "/api/sensors",
        json={
            "heart_rate_bpm": 70,
            "hrv_ms": 45,
            "gsr_raw": 1500,
            "signal_quality": "good",
            "source": "simulated",
        },
        headers=headers,
    )
    second = client.post(
        "/api/sensors",
        json={
            "heart_rate_bpm": 84,
            "hrv_ms": 35,
            "gsr_raw": 1900,
            "signal_quality": "moderate",
            "source": "simulated",
        },
        headers=headers,
    )
    assert first.status_code == 201
    assert second.status_code == 201

    latest = client.get("/api/sensors/latest", headers=headers)
    assert latest.status_code == 200
    body = latest.json()
    assert body["data"] is not None
    assert body["data"]["id"] == second.json()["reading"]["id"]
    assert body["data"]["heart_rate_bpm"] == 84
    assert body["data"]["hrv_ms"] == 35
    assert body["data"]["gsr_raw"] == 1900


def test_sensor_history_returns_saved_records_only_for_current_user(client):
    """GET /api/sensors/history should list saved readings for one user only."""
    user_a = {"X-Demo-User": "neurosense-sensor-history-a"}
    user_b = {"X-Demo-User": "neurosense-sensor-history-b"}

    first = client.post(
        "/api/sensors",
        json={
            "heart_rate_bpm": 71,
            "gsr_raw": 1510,
            "source": "simulated",
            "signal_quality": "good",
        },
        headers=user_a,
    )
    second = client.post(
        "/api/sensors",
        json={
            "heart_rate_bpm": 82,
            "gsr_raw": 1860,
            "source": "simulated",
            "signal_quality": "moderate",
        },
        headers=user_a,
    )
    other_user = client.post(
        "/api/sensors",
        json={
            "heart_rate_bpm": 90,
            "gsr_raw": 2100,
            "source": "simulated",
            "signal_quality": "good",
        },
        headers=user_b,
    )
    assert first.status_code == 201
    assert second.status_code == 201
    assert other_user.status_code == 201

    history_a = client.get(
        "/api/sensors/history?limit=10",
        headers=user_a,
    )
    assert history_a.status_code == 200
    body_a = history_a.json()
    assert body_a["total"] == 2
    returned_ids = {item["id"] for item in body_a["readings"]}
    assert returned_ids == {
        first.json()["reading"]["id"],
        second.json()["reading"]["id"],
    }

    history_b = client.get(
        "/api/sensors/history?limit=10",
        headers=user_b,
    )
    assert history_b.status_code == 200
    body_b = history_b.json()
    assert body_b["total"] == 1
    assert body_b["readings"][0]["id"] == other_user.json()["reading"]["id"]


def test_sensor_history_rejects_limit_above_maximum(client):
    """History pagination must enforce the API maximum limit."""
    response = client.get(
        "/api/sensors/history?limit=501",
        headers={"X-Demo-User": "neurosense-sensor-history-limit"},
    )

    assert response.status_code == 422




def test_demo_authentication_is_disabled_outside_development(client, monkeypatch):
    """API Test 7a: X-Demo-User must not authenticate production requests."""
    monkeypatch.setattr(api, "APP_ENV", "production")

    response = client.get(
        "/api/settings",
        headers={"X-Demo-User": "neurosense-production-user"},
    )

    assert response.status_code == 503
    assert "Firebase authentication" in response.json()["detail"]


def test_firebase_bearer_token_uses_verified_uid(client, monkeypatch):
    """API Test 7b: Firebase Bearer auth maps verified claims to the user ID."""
    fake_firebase_app = object()
    monkeypatch.setattr(api, "FIREBASE_APP", fake_firebase_app)

    def fake_verify_id_token(token, check_revoked, app):
        assert token == "unit-test-firebase-id-token"
        assert check_revoked is True
        assert app is fake_firebase_app
        return {"uid": "firebase-user-from-claims"}

    monkeypatch.setattr(
        api.firebase_auth,
        "verify_id_token",
        fake_verify_id_token,
    )

    response = client.get(
        "/api/settings",
        headers={
            "Authorization": "Bearer unit-test-firebase-id-token",
            "X-Demo-User": "must-not-override-firebase-uid",
        },
    )

    assert response.status_code == 200
    assert response.json()["haptic_enabled"] is False


def test_invalid_firebase_bearer_token_is_rejected(client, monkeypatch):
    """API Test 7c: failed Firebase token verification returns 401."""
    monkeypatch.setattr(api, "FIREBASE_APP", object())

    def reject_token(*args, **kwargs):
        raise ValueError("Invalid test token")

    monkeypatch.setattr(
        api.firebase_auth,
        "verify_id_token",
        reject_token,
    )

    response = client.get(
        "/api/settings",
        headers={"Authorization": "Bearer invalid-unit-test-token"},
    )

    assert response.status_code == 401
    assert "Invalid or expired" in response.json()["detail"]


def test_device_token_is_stored_as_sha256_digest_not_plaintext(client):
    """API Test 7d: device pairing must persist only a token digest."""
    user_id = "neurosense-device-token-storage-user"
    paired = pair_test_device(
        client,
        user_id,
        "neurosense-token-hash-device",
    )
    token = paired["device_token"]

    with api.db_connection() as db:
        row = db.execute(
            "SELECT token_hash FROM devices WHERE device_id = ?",
            ("neurosense-token-hash-device",),
        ).fetchone()

    assert row is not None
    stored_digest = row["token_hash"]
    assert stored_digest != token
    assert len(stored_digest) == 64


def test_firebase_auth_rejects_missing_bearer_token(client, monkeypatch):
    """When Firebase is configured, demo headers must not bypass authentication."""
    monkeypatch.setattr(api, "FIREBASE_APP", object())

    response = client.get(
        "/api/settings",
        headers={"X-Demo-User": "should-not-bypass-firebase"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Firebase Bearer token required."


def test_firebase_auth_accepts_verified_bearer_token(client, monkeypatch):
    """A verified Firebase ID token should resolve the Firebase UID."""
    fake_app = object()
    monkeypatch.setattr(api, "FIREBASE_APP", fake_app)

    def fake_verify(token, check_revoked, app):
        assert token == "unit-test-firebase-token"
        assert check_revoked is True
        assert app is fake_app
        return {"uid": "firebase-test-user-123"}

    monkeypatch.setattr(api.firebase_auth, "verify_id_token", fake_verify)

    response = client.get(
        "/api/settings",
        headers={"Authorization": "Bearer unit-test-firebase-token"},
    )

    assert response.status_code == 200
    assert response.json()["haptic_enabled"] is False


@pytest.mark.parametrize(
    "authorization",
    [
        "unit-test-token-without-scheme",
        "Basic unit-test-token",
        "Bearer",
        "Bearer token extra-part",
    ],
)
def test_firebase_auth_rejects_malformed_authorization_header(
    client,
    monkeypatch,
    authorization,
):
    """Firebase mode only accepts a correctly formatted Bearer header."""
    monkeypatch.setattr(api, "FIREBASE_APP", object())

    response = client.get(
        "/api/settings",
        headers={"Authorization": authorization},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == (
        "Use Authorization: Bearer <Firebase ID token>."
    )


def test_firebase_auth_rejects_invalid_or_expired_token(client, monkeypatch):
    """Invalid Firebase ID tokens must return 401."""
    monkeypatch.setattr(api, "FIREBASE_APP", object())

    def reject_token(*args, **kwargs):
        raise ValueError("invalid token")

    monkeypatch.setattr(api.firebase_auth, "verify_id_token", reject_token)

    response = client.get(
        "/api/settings",
        headers={"Authorization": "Bearer invalid-firebase-token"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or expired Firebase ID token."


def test_production_mode_fails_closed_without_firebase(client, monkeypatch):
    """Production mode must not silently fall back to development demo auth."""
    monkeypatch.setattr(api, "FIREBASE_APP", None)
    monkeypatch.setattr(api, "APP_ENV", "production")

    response = client.get(
        "/api/settings",
        headers={"X-Demo-User": "production-demo-user"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Firebase authentication is not configured."


def test_stress_api_returns_no_score_when_model_is_unavailable(client, monkeypatch):
    """Stress API must not fabricate a score when there is no loaded model."""
    monkeypatch.setattr(api, "STRESS_MODEL", None)
    user_id = "neurosense-stress-model-missing"
    reading = client.post(
        "/api/sensors",
        json={
            "heart_rate_bpm": 76,
            "hrv_ms": 42,
            "gsr_raw": 1700,
            "signal_quality": "good",
            "source": "wearable",
        },
        headers={"X-Demo-User": user_id},
    )
    assert reading.status_code == 201

    response = client.get(
        "/api/stress/latest",
        headers={"X-Demo-User": user_id},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "model_not_available"
    assert response.json()["stress_score"] is None


def test_stress_api_requires_baseline_before_prediction(client, monkeypatch):
    """A loaded model cannot infer stress until a personal baseline exists."""
    monkeypatch.setattr(api, "STRESS_MODEL", object())
    user_id = "neurosense-stress-baseline-required"
    reading = client.post(
        "/api/sensors",
        json={
            "heart_rate_bpm": 76,
            "hrv_ms": 42,
            "gsr_raw": 1700,
            "signal_quality": "good",
            "source": "wearable",
        },
        headers={"X-Demo-User": user_id},
    )
    assert reading.status_code == 201

    response = client.get(
        "/api/stress/latest",
        headers={"X-Demo-User": user_id},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "baseline_required"
    assert response.json()["stress_score"] is None


def test_stress_api_excludes_poor_quality_readings(client, monkeypatch):
    """Poor-quality wearable readings must not be passed to model inference."""
    monkeypatch.setattr(api, "STRESS_MODEL", object())
    user_id = "neurosense-stress-poor-quality"
    reading = client.post(
        "/api/sensors",
        json={
            "heart_rate_bpm": 76,
            "hrv_ms": 42,
            "gsr_raw": 1700,
            "signal_quality": "poor",
            "source": "wearable",
        },
        headers={"X-Demo-User": user_id},
    )
    assert reading.status_code == 201

    response = client.get(
        "/api/stress/latest",
        headers={"X-Demo-User": user_id},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "poor_signal_quality"
    assert response.json()["stress_score"] is None




def test_root_endpoint_reports_neurosense_identity(client):
    """API Test 8a: the root endpoint returns the expected service identity."""
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["project"] == "NeuroSense"
    assert body["version"] == "3.0.0"
    assert body["status"] == "online"


def test_three_hour_summary_generation_is_idempotent(client):
    """API Test 8b: the same three-hour summary window is not duplicated."""
    from datetime import datetime, timedelta, timezone

    user_id = "neurosense-summary-idempotency-user"
    timestamp = (
        datetime.now(timezone.utc) - timedelta(hours=7)
    ).isoformat()

    reading = post_wearable_reading(
        client,
        user_id,
        heart_rate_bpm=77,
        hrv_ms=41,
        gsr_raw=1680,
        timestamp=timestamp,
    )
    assert reading.status_code == 201

    first_creation = api.create_due_summaries()
    second_creation = api.create_due_summaries()

    assert first_creation == 1
    assert second_creation == 0

    history = client.get(
        "/api/summaries",
        headers={"X-Demo-User": user_id},
    )
    assert history.status_code == 200
    assert history.json()["total"] == 1


def test_paired_device_cloud_backup_never_stores_plaintext_token(client, monkeypatch):
    """Device ownership is backed up globally and per-user using only a token hash."""
    writes = []
    monkeypatch.setattr(
        api,
        "firestore_write",
        lambda path, document_id, payload: writes.append((path, document_id, dict(payload))) or "synced",
    )
    user_id = "neurosense-cloud-device-user"
    paired = pair_test_device(client, user_id, "neurosense-cloud-backup-device")

    assert paired["cloud_sync"] == {"global": "synced", "user": "synced"}
    assert len(writes) == 2
    assert writes[0][0] == ("devices",)
    assert writes[1][0] == ("users", user_id, "devices")
    stored = writes[0][2]
    assert len(stored["token_hash"]) == 64
    assert "device_token" not in stored
    assert paired["device_token"] not in repr(stored)


def test_cloud_device_records_restore_after_local_database_reset(client, monkeypatch):
    """A user's paired device metadata can be restored from Firestore."""
    user_id = "neurosense-cloud-device-restore-user"
    cloud_device = {
        "device_id": "neurosense-restored-device",
        "user_id": user_id,
        "device_name": "Restored ESP32",
        "token_hash": "a" * 64,
        "created_at": api.iso(),
        "last_seen": None,
        "active": True,
    }
    monkeypatch.setattr(
        api,
        "firestore_read_collection",
        lambda path, **kwargs: [dict(cloud_device)]
        if path == ("users", user_id, "devices") else [],
    )
    response = client.get(
        "/api/devices",
        headers={"X-Demo-User": user_id},
    )
    assert response.status_code == 200
    assert response.json()["devices"][0]["device_id"] == cloud_device["device_id"]
    assert response.json()["devices"][0]["device_name"] == "Restored ESP32"


def test_haptic_queue_is_not_delivered_after_user_disables_preference(client):
    """Turning haptics OFF after queueing prevents a not-yet-delivered pulse."""
    user_id = "neurosense-haptic-opt-out-user"
    device_id = "neurosense-haptic-opt-out-device"
    paired = pair_test_device(client, user_id, device_id)
    headers = {"X-Demo-User": user_id}

    enabled = client.put(
        "/api/settings/haptic",
        json={"haptic_enabled": True},
        headers=headers,
    )
    assert enabled.status_code == 200

    queued = client.post(
        "/api/haptic/commands",
        json={"device_id": device_id, "duration_ms": 500},
        headers=headers,
    )
    assert queued.status_code == 201

    disabled = client.put(
        "/api/settings/haptic",
        json={"haptic_enabled": False},
        headers=headers,
    )
    assert disabled.status_code == 200

    polled = client.get(
        "/api/devices/{}/commands".format(device_id),
        headers={"X-Device-Token": paired["device_token"]},
    )
    assert polled.status_code == 200
    assert polled.json()["commands"] == []


def test_model_features_preserve_missing_sensor_values_for_imputation():
    """Unavailable optional sensor channels must not become fake zero values."""
    reading = {
        "heart_rate_bpm": 78.0,
        "hrv_ms": None,
        "gsr_raw": None,
        "skin_temperature_c": None,
    }
    baseline = {
        "heart_rate_bpm": 72.0,
        "hrv_ms": None,
        "gsr_raw": None,
        "skin_temperature_c": None,
    }

    features = api.build_model_features(reading, baseline)

    assert features["heart_rate_bpm"] == 78.0
    assert features["heart_rate_delta"] == 6.0
    assert api.np.isnan(features["hrv_ms"])
    assert api.np.isnan(features["gsr_raw"])
    assert api.np.isnan(features["skin_temperature_c"])
    assert api.np.isnan(features["hrv_delta"])
    assert api.np.isnan(features["gsr_delta"])
    assert api.np.isnan(features["skin_temperature_delta"])


def test_model_features_preserve_valid_zero_measurement():
    """A supplied numeric zero is not confused with a missing value."""
    features = api.build_model_features(
        {
            "heart_rate_bpm": 72.0,
            "hrv_ms": 0.0,
            "gsr_raw": 0,
            "skin_temperature_c": None,
        },
        {
            "heart_rate_bpm": 72.0,
            "hrv_ms": 10.0,
            "gsr_raw": 100,
            "skin_temperature_c": None,
        },
    )

    assert features["hrv_ms"] == 0.0
    assert features["gsr_raw"] == 0.0
    assert features["hrv_delta"] == -10.0
    assert features["gsr_delta"] == -100.0
    assert api.np.isnan(features["skin_temperature_c"])


def test_haptic_command_status_reports_pending_delivered_and_acknowledged(client):
    """A user can inspect only their own haptic command lifecycle."""
    user_id = "neurosense-haptic-status-user"
    other_user = "neurosense-haptic-status-other"
    device_id = "neurosense-haptic-status-device"
    paired = pair_test_device(client, user_id, device_id)
    user_headers = {"X-Demo-User": user_id}

    client.put(
        "/api/settings/haptic",
        json={"haptic_enabled": True},
        headers=user_headers,
    )
    created = client.post(
        "/api/haptic/commands",
        json={"device_id": device_id, "duration_ms": 1000},
        headers=user_headers,
    )
    assert created.status_code == 201
    command_id = created.json()["command_id"]

    own_pending = client.get(
        "/api/haptic/commands/" + command_id,
        headers=user_headers,
    )
    assert own_pending.status_code == 200
    assert own_pending.json()["status"] == "pending"

    other_access = client.get(
        "/api/haptic/commands/" + command_id,
        headers={"X-Demo-User": other_user},
    )
    assert other_access.status_code == 404

    polled = client.get(
        "/api/devices/" + device_id + "/commands",
        headers={"X-Device-Token": paired["device_token"]},
    )
    assert polled.status_code == 200
    assert len(polled.json()["commands"]) == 1

    delivered = client.get(
        "/api/haptic/commands/" + command_id,
        headers=user_headers,
    )
    assert delivered.status_code == 200
    assert delivered.json()["status"] == "delivered"

    ack = client.post(
        "/api/devices/" + device_id + "/commands/" + command_id + "/ack",
        headers={"X-Device-Token": paired["device_token"]},
    )
    assert ack.status_code == 200

    acknowledged = client.get(
        "/api/haptic/commands/" + command_id,
        headers=user_headers,
    )
    assert acknowledged.status_code == 200
    assert acknowledged.json()["status"] == "acknowledged"


def test_haptic_command_older_than_sixty_seconds_is_not_delivered(client):
    """Stale queued haptic commands expire rather than firing much later."""
    user_id = "neurosense-haptic-expiry-user"
    device_id = "neurosense-haptic-expiry-device"
    paired = pair_test_device(client, user_id, device_id)
    headers = {"X-Demo-User": user_id}
    client.put(
        "/api/settings/haptic",
        json={"haptic_enabled": True},
        headers=headers,
    )
    created = client.post(
        "/api/haptic/commands",
        json={"device_id": device_id, "duration_ms": 1000},
        headers=headers,
    )
    command_id = created.json()["command_id"]

    with api.db_connection() as db:
        db.execute(
            "UPDATE haptic_commands SET created_at = ? WHERE command_id = ?",
            ("2000-01-01T00:00:00+00:00", command_id),
        )

    polled = client.get(
        "/api/devices/" + device_id + "/commands",
        headers={"X-Device-Token": paired["device_token"]},
    )
    assert polled.status_code == 200
    assert polled.json()["commands"] == []

    status = client.get(
        "/api/haptic/commands/" + command_id,
        headers=headers,
    )
    assert status.status_code == 200
    assert status.json()["status"] == "expired"

def test_pending_haptic_command_restores_from_firestore_after_sqlite_reset(client, monkeypatch):
    """A queued command is restored from cloud backup after an ephemeral DB reset."""
    user_id = "neurosense-haptic-cloud-restore-user"
    device_id = "neurosense-haptic-cloud-restore-device"
    paired = pair_test_device(client, user_id, device_id)
    headers = {"X-Demo-User": user_id}

    enabled = client.put(
        "/api/settings/haptic",
        json={"haptic_enabled": True},
        headers=headers,
    )
    assert enabled.status_code == 200

    queued = client.post(
        "/api/haptic/commands",
        json={"device_id": device_id, "duration_ms": 700},
        headers=headers,
    )
    assert queued.status_code == 201
    command_id = queued.json()["command_id"]

    with api.db_connection() as db:
        command_row = db.execute(
            "SELECT * FROM haptic_commands WHERE command_id = ?",
            (command_id,),
        ).fetchone()
        cloud_command = dict(command_row)
        db.execute(
            "DELETE FROM haptic_commands WHERE command_id = ?",
            (command_id,),
        )

    monkeypatch.setattr(
        api,
        "firestore_read_collection",
        lambda path, **kwargs: [dict(cloud_command)]
        if path == ("users", user_id, "haptic_commands") else [],
    )
    monkeypatch.setattr(api, "firestore_write", lambda *args, **kwargs: "synced")

    polled = client.get(
        "/api/devices/{}/commands".format(device_id),
        headers={"X-Device-Token": paired["device_token"]},
    )
    assert polled.status_code == 200
    assert polled.json()["commands"] == [
        {
            "command_id": command_id,
            "duration_ms": 700,
            "created_at": cloud_command["created_at"],
        }
    ]

    status = client.get(
        "/api/haptic/commands/" + command_id,
        headers=headers,
    )
    assert status.status_code == 200
    assert status.json()["status"] == "delivered"


def test_monitoring_interval_preferences_support_units_and_are_per_user(client):
    """Monitoring interval accepts seconds/minutes/hours and stays account-specific."""
    user_a = {"X-Demo-User": "neurosense-monitor-interval-a"}
    user_b = {"X-Demo-User": "neurosense-monitor-interval-b"}

    initial = client.get("/api/settings", headers=user_a)
    assert initial.status_code == 200
    assert initial.json()["monitoring_interval_seconds"] == 600
    assert initial.json()["monitoring_interval_value"] == 10
    assert initial.json()["monitoring_interval_unit"] == "minutes"

    saved_seconds = client.put(
        "/api/settings/monitoring-interval",
        json={"interval_value": 30, "interval_unit": "seconds"},
        headers=user_a,
    )
    assert saved_seconds.status_code == 200
    assert saved_seconds.json()["monitoring_interval_seconds"] == 30

    saved_hours = client.put(
        "/api/settings/monitoring-interval",
        json={"interval_value": 2, "interval_unit": "hours"},
        headers=user_a,
    )
    assert saved_hours.status_code == 200
    assert saved_hours.json()["monitoring_interval_seconds"] == 7200

    fetched_a = client.get("/api/settings", headers=user_a)
    fetched_b = client.get("/api/settings", headers=user_b)
    assert fetched_a.json()["monitoring_interval_value"] == 2
    assert fetched_a.json()["monitoring_interval_unit"] == "hours"
    assert fetched_b.json()["monitoring_interval_seconds"] == 600

    too_short = client.put(
        "/api/settings/monitoring-interval",
        json={"interval_value": 2, "interval_unit": "seconds"},
        headers=user_a,
    )
    assert too_short.status_code == 422
