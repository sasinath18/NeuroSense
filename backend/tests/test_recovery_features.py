"""Tests for the nine-phase NeuroSense recovery and personalization features."""

from datetime import datetime, timedelta, timezone
import json

import pytest
from fastapi.testclient import TestClient

from backend import main as api


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DB_PATH", tmp_path / "neurosense_features.db")
    monkeypatch.setattr(api, "MUSIC_STORAGE_DIR", tmp_path / "music")
    monkeypatch.setattr(api, "APP_ENV", "development")
    monkeypatch.setattr(api, "FIREBASE_APP", None)
    monkeypatch.setattr(api, "FIRESTORE_CLIENT", None)
    monkeypatch.setattr(api, "STRESS_MODEL", None)
    monkeypatch.setattr(api, "FIREBASE_STORAGE_BUCKET", "")
    monkeypatch.setattr(api, "initialize_firebase", lambda: None)
    monkeypatch.setattr(api, "initialize_ml_model", lambda: None)
    previous_overrides = dict(api.app.dependency_overrides)
    api.app.dependency_overrides.clear()
    try:
        with TestClient(api.app) as test_client:
            yield test_client
    finally:
        api.app.dependency_overrides.clear()
        api.app.dependency_overrides.update(previous_overrides)


def auth(user_id="phase-test-user"):
    return {"X-Demo-User": user_id}


class FakeStorageBlob:
    def __init__(self, bucket, path):
        self.bucket = bucket
        self.path = path

    def upload_from_string(self, content, content_type=None):
        self.bucket.objects[self.path] = (bytes(content), content_type)

    def download_as_bytes(self):
        return self.bucket.objects[self.path][0]

    def exists(self):
        return self.path in self.bucket.objects

    def delete(self):
        self.bucket.objects.pop(self.path, None)


class FakeStorageBucket:
    def __init__(self):
        self.objects = {}

    def blob(self, path):
        return FakeStorageBlob(self, path)


def test_firebase_cli_config_uses_deny_by_default_rules_for_client_sdks():
    """Firestore and Storage browser clients must remain denied; backend uses Admin SDK."""
    project_config_path = api.BASE_DIR / "firebase.json"
    assert project_config_path.is_file()
    project_config = json.loads(project_config_path.read_text(encoding="utf-8"))
    assert project_config["firestore"]["rules"] == "firestore.rules"
    assert project_config["storage"]["rules"] == "storage.rules"

    expected_rules = {
        "firestore.rules": (
            "service cloud.firestore {",
            "match /databases/{database}/documents {",
            "match /{document=**} {",
        ),
        "storage.rules": (
            "service firebase.storage {",
            "match /b/{bucket}/o {",
            "match /{allPaths=**} {",
        ),
    }

    for filename, required_fragments in expected_rules.items():
        rules_path = api.BASE_DIR / filename
        assert rules_path.is_file()
        rules = rules_path.read_text(encoding="utf-8")
        assert "rules_version = '2';" in rules
        for fragment in required_fragments:
            assert fragment in rules
        allow_statements = [
            line.strip()
            for line in rules.splitlines()
            if line.strip().startswith("allow ")
        ]
        assert allow_statements == ["allow read, write: if false;"]


def test_firebase_rules_deploy_workflow_is_manual_and_project_scoped():
    """A normal source push must never auto-publish production Firebase rules."""
    workflow_path = (
        api.BASE_DIR / ".github" / "workflows" / "deploy-firebase-rules.yml"
    )
    assert workflow_path.is_file()
    workflow = workflow_path.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "push:" not in workflow
    assert "pull_request:" not in workflow
    assert "if: github.ref == 'refs/heads/main'" in workflow
    assert "project_id: neurosense-d7f69" in workflow
    assert "--project neurosense-d7f69" in workflow
    assert "--only firestore:rules,storage" in workflow
    assert "NEUROSENSE_FIREBASE_RULES_DEPLOY_CREDENTIALS_JSON" in workflow
    assert "token_format: access_token" in workflow
    assert "serviceusage.googleapis.com/v1/projects/neurosense-d7f69/services/$api" in workflow
    assert "roles/serviceusage.serviceUsageViewer" not in workflow

    readme = (api.BASE_DIR / "README.md").read_text(encoding="utf-8")
    assert "roles/firebaserules.admin" in readme
    assert "roles/serviceusage.serviceUsageViewer" in readme
    assert "firebaserules.googleapis.com" in readme
    assert "firestore.googleapis.com" in readme
    assert "firebasestorage.googleapis.com" in readme
    assert "30-day recovery period" in readme


def test_music_mode_does_not_change_haptic_preference(client):
    enabled = client.put(
        "/api/settings/haptic",
        json={"haptic_enabled": True},
        headers=auth(),
    )
    assert enabled.status_code == 200

    music = client.put(
        "/api/settings/recovery",
        json={
            "music_mode": "automatic",
            "warning_sound_enabled": True,
            "notifications_enabled": False,
        },
        headers=auth(),
    )
    assert music.status_code == 200

    settings = client.get("/api/settings", headers=auth()).json()
    assert settings["music_mode"] == "automatic"
    assert settings["haptic_enabled"] is True
    assert settings["warning_sound_enabled"] is True
    assert settings["notifications_enabled"] is False

    # Changing haptic after music preferences must preserve music/alert settings.
    haptic_off = client.put(
        "/api/settings/haptic",
        json={"haptic_enabled": False},
        headers=auth(),
    )
    assert haptic_off.status_code == 200
    settings_after_haptic = client.get("/api/settings", headers=auth()).json()
    assert settings_after_haptic["haptic_enabled"] is False
    assert settings_after_haptic["music_mode"] == "automatic"
    assert settings_after_haptic["warning_sound_enabled"] is True


def test_music_library_upload_default_and_playback_are_user_scoped(client):
    first = client.post(
        "/api/music/upload",
        files={"file": ("calm.mp3", b"fake-mp3-audio", "audio/mpeg")},
        headers=auth("music-user-one"),
    )
    assert first.status_code == 201
    track_id = first.json()["track"]["track_id"]

    listed = client.get("/api/music", headers=auth("music-user-one"))
    assert listed.status_code == 200
    assert listed.json()["tracks"][0]["file_name"] == "calm.mp3"

    selected = client.put(
        f"/api/music/{track_id}/default",
        headers=auth("music-user-one"),
    )
    assert selected.status_code == 200

    audio = client.get(
        f"/api/music/{track_id}/audio",
        headers=auth("music-user-one"),
    )
    assert audio.status_code == 200
    assert audio.content == b"fake-mp3-audio"

    other_user = client.get(
        f"/api/music/{track_id}/audio",
        headers=auth("music-user-two"),
    )
    assert other_user.status_code == 404


def test_music_library_delete_is_user_scoped_and_removes_audio_file(client):
    uploaded = client.post(
        "/api/music/upload",
        files={"file": ("replace-me.mp3", b"old-song-content", "audio/mpeg")},
        headers=auth("music-delete-owner"),
    )
    assert uploaded.status_code == 201
    track_id = uploaded.json()["track"]["track_id"]

    with api.db_connection() as db:
        row = db.execute(
            "SELECT storage_path FROM music_tracks WHERE track_id = ? AND user_id = ?",
            (track_id, "music-delete-owner"),
        ).fetchone()
    assert row is not None
    audio_path = api.Path(row["storage_path"])
    assert audio_path.is_file()

    forbidden = client.delete(
        f"/api/music/{track_id}",
        headers=auth("different-user"),
    )
    assert forbidden.status_code == 404
    assert audio_path.is_file()

    deleted = client.delete(
        f"/api/music/{track_id}",
        headers=auth("music-delete-owner"),
    )
    assert deleted.status_code == 200
    assert deleted.json()["deleted_track_id"] == track_id
    assert not audio_path.exists()

    remaining = client.get("/api/music", headers=auth("music-delete-owner"))
    assert remaining.status_code == 200
    assert all(item["track_id"] != track_id for item in remaining.json()["tracks"])

    missing_audio = client.get(
        f"/api/music/{track_id}/audio",
        headers=auth("music-delete-owner"),
    )
    assert missing_audio.status_code == 404


def test_music_library_allows_ten_saved_tracks_and_rejects_the_eleventh(client):
    user_headers = auth("ten-song-limit-user")
    for index in range(10):
        response = client.post(
            "/api/music/upload",
            files={
                "file": (
                    f"track-{index + 1}.mp3",
                    f"audio-content-{index + 1}".encode(),
                    "audio/mpeg",
                )
            },
            headers=user_headers,
        )
        assert response.status_code == 201

    library = client.get("/api/music", headers=user_headers)
    assert library.status_code == 200
    assert len(library.json()["tracks"]) == 10
    assert library.json()["library_limit"] == 10
    assert library.json()["remaining_slots"] == 0

    rejected = client.post(
        "/api/music/upload",
        files={"file": ("track-11.mp3", b"eleventh-track", "audio/mpeg")},
        headers=user_headers,
    )
    assert rejected.status_code == 409
    assert "10/10" in rejected.json()["detail"]

    after_rejection = client.get("/api/music", headers=user_headers)
    assert len(after_rejection.json()["tracks"]) == 10
    assert after_rejection.json()["remaining_slots"] == 0


def test_blink_summaries_are_saved_per_google_uid(client):
    saved = client.post(
        "/api/face/blink-summary",
        json={"blink_count": 36, "duration_seconds": 120},
        headers=auth("blink-user-one"),
    )
    assert saved.status_code == 201
    assert saved.json()["summary"]["blinks_per_minute"] == 18.0

    own_history = client.get(
        "/api/face/blink-history",
        headers=auth("blink-user-one"),
    )
    other_history = client.get(
        "/api/face/blink-history",
        headers=auth("blink-user-two"),
    )
    assert own_history.json()["total"] == 1
    assert other_history.json()["total"] == 0


def test_personalization_profile_uses_distinct_days_and_is_preliminary(client):
    now = datetime.now(timezone.utc)
    # Insert historical rows directly: the public sensor-ingest API correctly
    # rejects device packets older than 24 hours.
    with api.db_connection() as db:
        for day_offset in (1, 2):
            for index in range(5):
                timestamp = (now - timedelta(days=day_offset, minutes=index)).isoformat()
                db.execute(
                    """
                    INSERT INTO sensor_readings (
                        id, user_id, device_id, timestamp, received_at,
                        heart_rate_bpm, hrv_ms, gsr_raw, skin_temperature_c,
                        signal_quality, source, ibi_ms_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'good', 'wearable', NULL)
                    """,
                    (
                        f"learning-{day_offset}-{index}", "learning-user",
                        "test-device", timestamp, now.isoformat(),
                        70 + index, 40 + index, 1700 + index, 33.0,
                    ),
                )

    profile = client.get(
        "/api/personalization/profile",
        headers=auth("learning-user"),
    )
    assert profile.status_code == 200
    assert profile.json()["days_observed"] == 2
    assert profile.json()["ready_for_preliminary_profile"] is True
    assert profile.json()["recommended_profile_ready"] is False


def test_production_music_upload_refuses_ephemeral_storage(client, monkeypatch):
    """Production must not accept music that would vanish with Render's filesystem."""
    monkeypatch.setattr(api, "APP_ENV", "production")
    monkeypatch.setattr(api, "FIREBASE_APP", None)
    monkeypatch.setattr(api, "FIREBASE_STORAGE_BUCKET", "")
    client.app.dependency_overrides[api.current_user_id] = lambda: "production-music-user"

    response = client.post(
        "/api/music/upload",
        files={"file": ("calm.mp3", b"test-audio", "audio/mpeg")},
    )

    assert response.status_code == 503
    assert "Durable music storage is not configured" in response.json()["detail"]
    with api.db_connection() as db:
        assert db.execute("SELECT COUNT(*) FROM music_tracks").fetchone()[0] == 0


def test_firebase_storage_music_upload_download_delete_round_trip(client, monkeypatch):
    """Cloud audio and its per-UID Firestore metadata must both persist."""
    bucket = FakeStorageBucket()
    firebase_app = object()
    writes = []
    monkeypatch.setattr(api, "APP_ENV", "production")
    monkeypatch.setattr(api, "FIREBASE_APP", firebase_app)
    monkeypatch.setattr(api, "FIREBASE_STORAGE_BUCKET", "neurosense-test.firebasestorage.app")
    monkeypatch.setattr(
        api.firebase_storage,
        "bucket",
        lambda name, app: bucket if name == api.FIREBASE_STORAGE_BUCKET and app is firebase_app else None,
    )
    monkeypatch.setattr(
        api,
        "firestore_write",
        lambda path, document_id, payload: writes.append((path, document_id, dict(payload))) or "synced",
    )
    monkeypatch.setattr(api, "firestore_read_collection", lambda *args, **kwargs: [])
    monkeypatch.setattr(api, "firestore_delete_document", lambda *args, **kwargs: "synced")
    client.app.dependency_overrides[api.current_user_id] = lambda: "firebase-music-owner"

    uploaded = client.post(
        "/api/music/upload",
        files={"file": ("calm.mp3", b"firebase-audio-bytes", "audio/mpeg")},
    )
    assert uploaded.status_code == 201
    track = uploaded.json()["track"]
    assert track["storage_backend"] == "firebase_storage"
    assert len(bucket.objects) == 1
    assert len(writes) == 1
    assert writes[0][0] == ("users", "firebase-music-owner", "music_tracks")
    assert writes[0][2]["storage_path"] in bucket.objects

    playback = client.get(f"/api/music/{track['track_id']}/audio")
    assert playback.status_code == 200
    assert playback.content == b"firebase-audio-bytes"

    client.app.dependency_overrides[api.current_user_id] = lambda: "different-music-user"
    forbidden = client.get(f"/api/music/{track['track_id']}/audio")
    assert forbidden.status_code == 404

    client.app.dependency_overrides[api.current_user_id] = lambda: "firebase-music-owner"
    deleted = client.delete(f"/api/music/{track['track_id']}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted_track_id"] == track["track_id"]
    assert bucket.objects == {}


def test_firebase_storage_upload_rolls_back_when_firestore_metadata_fails(client, monkeypatch):
    """An audio blob must be removed when its durable UID-scoped metadata write fails."""
    bucket = FakeStorageBucket()
    monkeypatch.setattr(api, "APP_ENV", "production")
    monkeypatch.setattr(api, "FIREBASE_APP", object())
    monkeypatch.setattr(api, "FIREBASE_STORAGE_BUCKET", "neurosense-test.firebasestorage.app")
    monkeypatch.setattr(api.firebase_storage, "bucket", lambda name, app: bucket)
    monkeypatch.setattr(api, "firestore_write", lambda *args, **kwargs: "failed")
    client.app.dependency_overrides[api.current_user_id] = lambda: "firebase-metadata-failure-user"

    response = client.post(
        "/api/music/upload",
        files={"file": ("calm.mp3", b"test-audio", "audio/mpeg")},
    )

    assert response.status_code == 503
    assert "metadata could not be backed up" in response.json()["detail"]
    assert bucket.objects == {}
    with api.db_connection() as db:
        assert db.execute("SELECT COUNT(*) FROM music_tracks").fetchone()[0] == 0


def test_public_firebase_config_reports_storage_readiness_without_bucket_name(client, monkeypatch):
    """Public configuration may expose a readiness boolean but never bucket or Admin secrets."""
    bucket = FakeStorageBucket()
    firebase_app = object()
    monkeypatch.setattr(api, "FIREBASE_APP", firebase_app)
    monkeypatch.setattr(api, "FIREBASE_STORAGE_BUCKET", "private-bucket-name")
    monkeypatch.setattr(api.firebase_storage, "bucket", lambda name, app: bucket)

    response = client.get("/api/public-config")

    assert response.status_code == 200
    body = response.json()
    assert body["firebase_storage_configured"] is True
    assert "private-bucket-name" not in response.text
    assert "FIREBASE_SERVICE_ACCOUNT_JSON" not in response.text



def test_firebase_music_metadata_and_playback_restore_after_sqlite_reset(
    client,
    monkeypatch,
):
    """A refreshed library must recover its Firestore metadata and Storage audio."""
    user_id = "firebase-music-refresh-user"
    firebase_app = object()
    bucket = FakeStorageBucket()
    cloud_documents = {}

    monkeypatch.setattr(api, "APP_ENV", "production")
    monkeypatch.setattr(api, "FIREBASE_APP", firebase_app)
    monkeypatch.setattr(api, "FIREBASE_STORAGE_BUCKET", "neurosense-test.firebasestorage.app")
    monkeypatch.setattr(
        api.firebase_storage,
        "bucket",
        lambda name, app: bucket if name == api.FIREBASE_STORAGE_BUCKET and app is firebase_app else None,
    )

    def write_cloud_document(path, document_id, payload):
        if path == ("users", user_id, "music_tracks"):
            cloud_documents[document_id] = dict(payload)
        return "synced"

    def read_cloud_collection(path, **kwargs):
        if path == ("users", user_id, "music_tracks"):
            return [dict(item) for item in cloud_documents.values()]
        return []

    monkeypatch.setattr(api, "firestore_write", write_cloud_document)
    monkeypatch.setattr(api, "firestore_read_collection", read_cloud_collection)
    client.app.dependency_overrides[api.current_user_id] = lambda: user_id

    uploaded = client.post(
        "/api/music/upload",
        files={"file": ("after-refresh.mp3", b"persistent-firebase-audio", "audio/mpeg")},
    )
    assert uploaded.status_code == 201
    track_id = uploaded.json()["track"]["track_id"]
    assert track_id in cloud_documents
    storage_path = cloud_documents[track_id]["storage_path"]
    assert bucket.objects[storage_path][0] == b"persistent-firebase-audio"

    # Simulate Render's ephemeral SQLite metadata disappearing on restart.
    with api.db_connection() as db:
        db.execute(
            "DELETE FROM music_tracks WHERE track_id = ? AND user_id = ?",
            (track_id, user_id),
        )
        assert db.execute(
            "SELECT COUNT(*) FROM music_tracks WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0] == 0

    refreshed_library = client.get("/api/music")
    assert refreshed_library.status_code == 200
    recovered = next(
        item for item in refreshed_library.json()["tracks"]
        if item["track_id"] == track_id
    )
    assert recovered["storage_backend"] == "firebase_storage"

    with api.db_connection() as db:
        restored = db.execute(
            "SELECT storage_path FROM music_tracks WHERE track_id = ? AND user_id = ?",
            (track_id, user_id),
        ).fetchone()
    assert restored is not None
    assert restored["storage_path"] == storage_path

    playback = client.get(f"/api/music/{track_id}/audio")
    assert playback.status_code == 200
    assert playback.content == b"persistent-firebase-audio"



def test_firebase_music_delete_after_sqlite_reset_removes_cloud_metadata_and_audio(
    client,
    monkeypatch,
):
    """Delete must find and remove a user's cloud-only song after an SQLite reset."""
    user_id = "firebase-music-delete-reset-user"
    firebase_app = object()
    bucket = FakeStorageBucket()
    cloud_documents = {}
    delete_paths = []

    monkeypatch.setattr(api, "APP_ENV", "production")
    monkeypatch.setattr(api, "FIREBASE_APP", firebase_app)
    monkeypatch.setattr(api, "FIREBASE_STORAGE_BUCKET", "neurosense-test.firebasestorage.app")
    monkeypatch.setattr(
        api.firebase_storage,
        "bucket",
        lambda name, app: bucket if name == api.FIREBASE_STORAGE_BUCKET and app is firebase_app else None,
    )
    monkeypatch.setattr(api, "firestore_read_collection", lambda path, **kwargs: (
        [dict(item) for item in cloud_documents.values()]
        if path == ("users", user_id, "music_tracks") else []
    ))

    def write_cloud_document(path, document_id, payload):
        if path == ("users", user_id, "music_tracks"):
            cloud_documents[document_id] = dict(payload)
        return "synced"

    def read_cloud_document(path, document_id):
        if path == ("users", user_id, "music_tracks"):
            return dict(cloud_documents[document_id]) if document_id in cloud_documents else None
        return None

    def delete_cloud_document(path, document_id):
        delete_paths.append((path, document_id))
        cloud_documents.pop(document_id, None)
        return "synced"

    monkeypatch.setattr(api, "firestore_write", write_cloud_document)
    monkeypatch.setattr(api, "firestore_read_document", read_cloud_document)
    monkeypatch.setattr(api, "firestore_delete_document", delete_cloud_document)
    client.app.dependency_overrides[api.current_user_id] = lambda: user_id

    uploaded = client.post(
        "/api/music/upload",
        files={"file": ("remove-after-reset.mp3", b"delete-cloud-audio", "audio/mpeg")},
    )
    assert uploaded.status_code == 201
    track_id = uploaded.json()["track"]["track_id"]
    storage_path = cloud_documents[track_id]["storage_path"]
    assert storage_path in bucket.objects

    # Simulate a Render restart that removed only the ephemeral SQLite metadata.
    with api.db_connection() as db:
        db.execute(
            "DELETE FROM music_tracks WHERE track_id = ? AND user_id = ?",
            (track_id, user_id),
        )

    # A different authenticated user must not see or delete a cloud-only track.
    client.app.dependency_overrides[api.current_user_id] = lambda: "different-cloud-music-user"
    forbidden = client.delete(f"/api/music/{track_id}")
    assert forbidden.status_code == 404
    assert track_id in cloud_documents
    assert storage_path in bucket.objects

    client.app.dependency_overrides[api.current_user_id] = lambda: user_id
    deleted = client.delete(f"/api/music/{track_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted_track_id"] == track_id
    assert deleted.json()["cloud_sync"] == "synced"
    assert delete_paths == [
        (("users", user_id, "music_tracks"), track_id)
    ]
    assert track_id not in cloud_documents
    assert bucket.objects == {}

    remaining = client.get("/api/music")
    assert remaining.status_code == 200
    assert all(item["track_id"] != track_id for item in remaining.json()["tracks"])



def test_cloud_music_delete_keeps_firestore_record_when_storage_delete_fails(
    client,
    monkeypatch,
):
    """A failed blob delete must leave cloud metadata available so deletion can be retried."""
    user_id = "firebase-music-storage-delete-retry-user"
    firebase_app = object()
    bucket = FakeStorageBucket()
    cloud_documents = {}
    delete_calls = []
    original_delete = FakeStorageBlob.delete

    monkeypatch.setattr(api, "APP_ENV", "production")
    monkeypatch.setattr(api, "FIREBASE_APP", firebase_app)
    monkeypatch.setattr(api, "FIREBASE_STORAGE_BUCKET", "neurosense-test.firebasestorage.app")
    monkeypatch.setattr(
        api.firebase_storage,
        "bucket",
        lambda name, app: bucket if name == api.FIREBASE_STORAGE_BUCKET and app is firebase_app else None,
    )
    monkeypatch.setattr(api, "firestore_read_collection", lambda path, **kwargs: [])
    monkeypatch.setattr(
        api,
        "firestore_write",
        lambda path, document_id, payload: cloud_documents.setdefault(
            document_id, dict(payload)
        ) and "synced",
    )
    monkeypatch.setattr(
        api,
        "firestore_read_document",
        lambda path, document_id: (
            dict(cloud_documents[document_id])
            if path == ("users", user_id, "music_tracks") and document_id in cloud_documents
            else None
        ),
    )

    def delete_cloud_document(path, document_id):
        delete_calls.append((path, document_id))
        cloud_documents.pop(document_id, None)
        return "synced"

    monkeypatch.setattr(api, "firestore_delete_document", delete_cloud_document)
    client.app.dependency_overrides[api.current_user_id] = lambda: user_id

    uploaded = client.post(
        "/api/music/upload",
        files={"file": ("retry-storage.mp3", b"retry-storage-audio", "audio/mpeg")},
    )
    assert uploaded.status_code == 201
    track_id = uploaded.json()["track"]["track_id"]
    storage_path = cloud_documents[track_id]["storage_path"]

    # Model an ephemeral SQLite reset: deletion must now rely on Firestore.
    with api.db_connection() as db:
        db.execute("DELETE FROM music_tracks WHERE track_id = ?", (track_id,))

    def fail_blob_delete(self):
        raise OSError("simulated Firebase Storage delete failure")

    monkeypatch.setattr(FakeStorageBlob, "delete", fail_blob_delete)
    failed = client.delete(f"/api/music/{track_id}")
    assert failed.status_code == 503
    assert track_id in cloud_documents
    assert storage_path in bucket.objects
    assert delete_calls == []

    # Once Storage recovers, the same cloud-only item can be deleted cleanly.
    monkeypatch.setattr(FakeStorageBlob, "delete", original_delete)
    retried = client.delete(f"/api/music/{track_id}")
    assert retried.status_code == 200
    assert track_id not in cloud_documents
    assert bucket.objects == {}
    assert delete_calls == [
        (("users", user_id, "music_tracks"), track_id)
    ]


def test_cloud_music_delete_keeps_firestore_metadata_when_metadata_delete_fails(
    client,
    monkeypatch,
):
    """If Firestore fails after blob deletion, a retry can finish metadata cleanup."""
    user_id = "firebase-music-metadata-delete-retry-user"
    firebase_app = object()
    bucket = FakeStorageBucket()
    cloud_documents = {}
    delete_attempts = 0

    monkeypatch.setattr(api, "APP_ENV", "production")
    monkeypatch.setattr(api, "FIREBASE_APP", firebase_app)
    monkeypatch.setattr(api, "FIREBASE_STORAGE_BUCKET", "neurosense-test.firebasestorage.app")
    monkeypatch.setattr(
        api.firebase_storage,
        "bucket",
        lambda name, app: bucket if name == api.FIREBASE_STORAGE_BUCKET and app is firebase_app else None,
    )
    monkeypatch.setattr(api, "firestore_read_collection", lambda path, **kwargs: [])
    monkeypatch.setattr(
        api,
        "firestore_write",
        lambda path, document_id, payload: cloud_documents.setdefault(
            document_id, dict(payload)
        ) and "synced",
    )
    monkeypatch.setattr(
        api,
        "firestore_read_document",
        lambda path, document_id: (
            dict(cloud_documents[document_id])
            if path == ("users", user_id, "music_tracks") and document_id in cloud_documents
            else None
        ),
    )

    def flaky_cloud_delete(path, document_id):
        nonlocal delete_attempts
        delete_attempts += 1
        if delete_attempts == 1:
            return "failed"
        cloud_documents.pop(document_id, None)
        return "synced"

    monkeypatch.setattr(api, "firestore_delete_document", flaky_cloud_delete)
    client.app.dependency_overrides[api.current_user_id] = lambda: user_id

    uploaded = client.post(
        "/api/music/upload",
        files={"file": ("retry-metadata.mp3", b"retry-metadata-audio", "audio/mpeg")},
    )
    assert uploaded.status_code == 201
    track_id = uploaded.json()["track"]["track_id"]
    storage_path = cloud_documents[track_id]["storage_path"]

    with api.db_connection() as db:
        db.execute("DELETE FROM music_tracks WHERE track_id = ?", (track_id,))

    failed = client.delete(f"/api/music/{track_id}")
    assert failed.status_code == 503
    assert track_id in cloud_documents
    assert storage_path not in bucket.objects

    # The missing blob is treated as already deleted; retry removes metadata.
    retried = client.delete(f"/api/music/{track_id}")
    assert retried.status_code == 200
    assert track_id not in cloud_documents
    assert bucket.objects == {}
    assert delete_attempts == 2
