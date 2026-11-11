"""Tests for the nine-phase NeuroSense recovery and personalization features."""

from datetime import datetime, timedelta, timezone

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
    monkeypatch.setattr(api, "TWILIO_ACCOUNT_SID", "")
    monkeypatch.setattr(api, "TWILIO_AUTH_TOKEN", "")
    monkeypatch.setattr(api, "TWILIO_WHATSAPP_FROM", "")
    monkeypatch.setattr(api, "initialize_firebase", lambda: None)
    monkeypatch.setattr(api, "initialize_ml_model", lambda: None)
    with TestClient(api.app) as test_client:
        yield test_client


def auth(user_id="phase-test-user"):
    return {"X-Demo-User": user_id}


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


def test_emergency_contact_requires_explicit_enable_and_twilio_setup(client):
    saved = client.put(
        "/api/emergency-contact",
        json={"phone": "+919876543210", "enabled": True},
        headers=auth("contact-user"),
    )
    assert saved.status_code == 200
    assert saved.json()["enabled"] is True
    assert saved.json()["whatsapp_configured"] is False

    test_message = client.post(
        "/api/emergency-contact/test",
        headers=auth("contact-user"),
    )
    assert test_message.status_code == 503


def test_emergency_contact_rejects_non_e164_phone(client):
    response = client.put(
        "/api/emergency-contact",
        json={"phone": "9876543210", "enabled": True},
        headers=auth("invalid-contact-user"),
    )
    assert response.status_code == 422
