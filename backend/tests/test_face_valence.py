import numpy as np

from backend.face_valence import (
    FEATURE_NAMES,
    MIN_VALID_FRAMES,
    RAW_FEATURES,
    build_live_feature_frame,
    predict_from_openface_rows,
)


def make_rows(count=45):
    rows = []
    for index in range(count):
        row = {
            "timestamp": index * 0.1,
            "success": 1,
            "confidence": 0.9,
        }
        for feature_index, name in enumerate(RAW_FEATURES):
            row[name] = float(index + feature_index)
        rows.append(row)
    return rows


def test_schema_is_exactly_50_inputs_in_training_order():
    assert len(RAW_FEATURES) == 25
    assert len(FEATURE_NAMES) == 50
    assert FEATURE_NAMES[:25] == tuple(
        f"face__{name}__mean" for name in RAW_FEATURES
    )
    assert FEATURE_NAMES[25:] == tuple(
        f"face__{name}__std" for name in RAW_FEATURES
    )


def test_uses_only_first_4_8_seconds_success_and_confidence():
    rows = make_rows(50)
    rows[0]["success"] = 0
    rows[1]["confidence"] = 0.79
    rows[-1]["timestamp"] = 4.9

    features, valid_count, reason = build_live_feature_frame(rows)

    assert reason is None
    assert features is not None
    assert valid_count == 47
    assert features.shape == (1, 50)
    assert list(features.columns) == list(FEATURE_NAMES)
    assert all(dtype == np.dtype("float32") for dtype in features.dtypes)


def test_fewer_than_40_valid_frames_returns_no_prediction():
    rows = make_rows(MIN_VALID_FRAMES - 1)
    features, valid_count, reason = build_live_feature_frame(rows)

    assert features is None
    assert valid_count == MIN_VALID_FRAMES - 1
    assert "at least 40" in reason


def test_standard_deviation_uses_population_ddof_zero():
    rows = make_rows(40)
    features, valid_count, reason = build_live_feature_frame(rows)

    assert reason is None
    expected = np.std(
        np.arange(40, dtype=np.float32),
        ddof=0,
        dtype=np.float32,
    )
    assert features.iloc[0]["face__gaze_angle_x__std"] == expected


def test_prediction_is_not_attempted_when_valid_frames_are_insufficient():
    class ShouldNotPredict:
        def predict(self, _features):
            raise AssertionError("predict() must not be called")

    result = predict_from_openface_rows(
        ShouldNotPredict(),
        make_rows(MIN_VALID_FRAMES - 1),
    )

    assert result["status"] == "no_prediction"
    assert result["valid_frames"] == MIN_VALID_FRAMES - 1


def test_valid_rows_are_passed_to_model_as_50_float32_features():
    class CaptureModel:
        def predict(self, features):
            assert features.shape == (1, 50)
            assert list(features.columns) == list(FEATURE_NAMES)
            assert all(dtype == np.dtype("float32") for dtype in features.dtypes)
            return np.array([0.125], dtype=np.float64)

    result = predict_from_openface_rows(
        CaptureModel(),
        make_rows(45),
    )

    assert result["status"] == "prediction_available"
    assert result["felt_valence"] == 0.125
    assert result["valid_frames"] == 45
    assert result["features_used"] == 50
    assert result["numeric_type"] == "float32"





def test_predict_from_video_bytes_keeps_uploaded_video_inside_tempdir(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    from backend.face_valence import predict_from_video_bytes

    executable = tmp_path / "FeatureExtraction"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    openface_root = tmp_path / "OpenFace"
    openface_root.mkdir()
    observed = {}

    monkeypatch.setattr("backend.face_valence._probe_duration", lambda _path: 5.0)

    def fake_run(command, **kwargs):
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"fake-mp4")
            return subprocess.CompletedProcess(command, 0, "", "")
        else:
            out_dir = Path(command[command.index("-out_dir") + 1])
            csv_rows = [
                ",".join(["timestamp", "success", "confidence", *RAW_FEATURES]),
            ]
            for index in range(40):
                timestamp = index * 0.1
                csv_rows.append(
                    ",".join([str(timestamp), "1", "0.9"] + ["1.0"] * len(RAW_FEATURES))
                )
            (out_dir / "camera_capture.csv").write_text(
                "\n".join(csv_rows) + "\n",
                encoding="utf-8",
            )
            observed["cwd"] = kwargs["cwd"]
            observed["video_path"] = command[command.index("-f") + 1]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("backend.face_valence.subprocess.run", fake_run)

    class CaptureModel:
        def predict(self, features):
            assert features.shape == (1, 50)
            return np.array([0.25])

    result = predict_from_video_bytes(
        CaptureModel(), b"fake-webm", executable, openface_root
    )
    assert result["status"] == "prediction_available"
    assert result["felt_valence"] == 0.25
    assert Path(observed["cwd"]) == executable.parent
    assert not Path(observed["video_path"]).exists()

def test_model_download_is_not_published_when_sha256_is_wrong(tmp_path, monkeypatch):
    from backend.face_valence import load_face_valence_model

    model_path = tmp_path / "face_valence.joblib"
    payload = b"x" * 1_100_000

    class FakeGdown:
        @staticmethod
        def download(url, output, quiet, fuzzy):
            from pathlib import Path
            Path(output).write_bytes(payload)
            return output

    monkeypatch.setitem(__import__("sys").modules, "gdown", FakeGdown)
    import pytest

    with pytest.raises(RuntimeError, match="SHA-256"):
        load_face_valence_model(
            model_path,
            model_url="https://drive.google.com/uc?id=test",
            expected_sha256="0" * 64,
        )

    assert not model_path.exists()
    assert not model_path.with_suffix(model_path.suffix + ".download").exists()


def test_model_download_publishes_only_after_minimum_size_check(tmp_path, monkeypatch):
    from backend.face_valence import load_face_valence_model
    import pytest

    class FakeGdown:
        @staticmethod
        def download(url, output, quiet, fuzzy):
            from pathlib import Path
            Path(output).write_bytes(b"too small")
            return output

    monkeypatch.setitem(__import__("sys").modules, "gdown", FakeGdown)
    model_path = tmp_path / "face_valence.joblib"

    with pytest.raises(RuntimeError, match="unexpectedly small"):
        load_face_valence_model(
            model_path,
            model_url="https://drive.google.com/uc?id=test",
        )

    assert not model_path.exists()
    assert not model_path.with_suffix(model_path.suffix + ".download").exists()




def test_affec_artifact_is_verified_without_unpickling_in_main_process(tmp_path, monkeypatch):
    import hashlib
    from backend.face_valence import load_face_valence_model

    model_path = tmp_path / "face_valence.joblib"
    payload = b"safe-model-artifact" * 60_000
    model_path.write_bytes(payload)

    # A main-process joblib load would mix AFFEC sklearn 1.6.1 with the
    # wearable model's sklearn 1.9.1. The loader must only verify the file.
    import joblib
    monkeypatch.setattr(
        joblib,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("AFFEC must not be unpickled in the API process")
        ),
    )
    result = load_face_valence_model(
        model_path,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert result == model_path


def test_affec_worker_receives_exact_float32_feature_row(tmp_path, monkeypatch):
    import json
    import subprocess
    from backend.face_valence import _predict_with_affec_worker

    fake_python = tmp_path / "fake-affec-python"
    fake_python.write_text("test interpreter placeholder", encoding="utf-8")
    model_path = tmp_path / "face_valence.joblib"
    model_path.write_bytes(b"artifact")

    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["payload"] = json.loads(kwargs["input"])
        return subprocess.CompletedProcess(
            command, 0, '{"felt_valence":0.375}', ""
        )

    import backend.face_valence as face_valence
    monkeypatch.setenv("NEUROSENSE_AFFEC_PYTHON", str(fake_python))
    monkeypatch.setattr(face_valence.subprocess, "run", fake_run)

    frame, _, reason = build_live_feature_frame(make_rows(45))
    assert reason is None
    result = _predict_with_affec_worker(model_path, frame)

    assert result == 0.375
    assert observed["command"][0] == str(fake_python)
    assert len(observed["payload"]["feature_names"]) == 50
    assert len(observed["payload"]["features"]) == 50
    assert observed["payload"]["features"]["face__gaze_angle_x__mean"] == 22.0


def test_openface_timeout_is_reported_as_a_feature_extraction_timeout(tmp_path, monkeypatch):
    import subprocess
    import pytest
    from backend.face_valence import predict_from_video_bytes

    executable = tmp_path / "FeatureExtraction"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    openface_root = tmp_path / "OpenFace"
    openface_root.mkdir()

    def fake_probe(_path):
        return 5.0

    def fake_run(command, **kwargs):
        if command[0] == "ffmpeg":
            from pathlib import Path
            Path(command[-1]).write_bytes(b"fake-mp4")
            return subprocess.CompletedProcess(command, 0, "", "")
        raise subprocess.TimeoutExpired(command, timeout=120)

    monkeypatch.setattr("backend.face_valence._probe_duration", fake_probe)
    monkeypatch.setattr("backend.face_valence.subprocess.run", fake_run)

    with pytest.raises(TimeoutError, match="processing time limit"):
        predict_from_video_bytes(
            object(),
            b"fake-webm",
            executable,
            openface_root,
        )
