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



def test_model_download_is_not_published_when_sha256_is_wrong(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

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
