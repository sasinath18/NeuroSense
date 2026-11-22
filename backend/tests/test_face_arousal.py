from __future__ import annotations

import numpy as np

from backend.face_arousal import (
    FEATURE_NAMES,
    MIN_VALID_FRAMES,
    RAW_FEATURES,
    build_live_feature_frame,
    predict_from_openface_rows,
)


def make_rows(count: int = 48):
    rows = []
    for index in range(count):
        row = {
            "timestamp": index * 0.08,
            "success": 1,
            "confidence": 0.95,
        }
        for feature_index, name in enumerate(RAW_FEATURES):
            row[name] = float(index + feature_index) / 10.0
        rows.append(row)
    return rows


def test_training_schema_has_44_interleaved_mean_std_features():
    assert len(RAW_FEATURES) == 22
    assert len(FEATURE_NAMES) == 44
    assert FEATURE_NAMES[:4] == (
        RAW_FEATURES[0] + "__mean",
        RAW_FEATURES[0] + "__std",
        RAW_FEATURES[1] + "__mean",
        RAW_FEATURES[1] + "__std",
    )


def test_face_features_are_aggregated_only_from_valid_openface_rows():
    rows = make_rows()
    rows[0]["success"] = 0
    rows[1]["confidence"] = 0.79
    frame, valid_count, reason = build_live_feature_frame(rows)

    assert reason is None
    assert frame is not None
    assert valid_count == len(rows) - 2
    assert frame.shape == (1, 44)
    assert tuple(frame.columns) == FEATURE_NAMES
    assert all(np.issubdtype(dtype, np.number) for dtype in frame.dtypes)


def test_insufficient_frames_returns_no_prediction():
    frame, valid_count, reason = build_live_feature_frame(
        make_rows(MIN_VALID_FRAMES - 1)
    )

    assert frame is None
    assert valid_count == MIN_VALID_FRAMES - 1
    assert "at least 40" in reason


def test_prediction_uses_exact_saved_feature_order_and_reports_arousal_not_stress():
    class CaptureModel:
        def predict(self, frame):
            assert frame.shape == (1, 44)
            assert tuple(frame.columns) == FEATURE_NAMES
            return np.asarray([5.25])

    bundle = {
        "model": CaptureModel(),
        "feature_columns": list(FEATURE_NAMES),
        "target": "f_emotion_a",
    }

    result = predict_from_openface_rows(bundle, make_rows())

    assert result["status"] == "prediction_available"
    assert result["estimated_arousal"] == 5.25
    assert result["scale_min"] == 1
    assert result["scale_max"] == 9
    assert result["features_used"] == 44
    assert result["target"] == "f_emotion_a"
    assert result["stress_inference"] is False
    assert result["research_only"] is True


def test_prediction_does_not_call_model_when_frame_quality_is_insufficient():
    class MustNotPredict:
        def predict(self, _frame):
            raise AssertionError("Model must not run on insufficient frames.")

    bundle = {
        "model": MustNotPredict(),
        "feature_columns": list(FEATURE_NAMES),
        "target": "f_emotion_a",
    }

    result = predict_from_openface_rows(
        bundle,
        make_rows(MIN_VALID_FRAMES - 1),
    )

    assert result["status"] == "no_prediction"
    assert result["valid_frames"] == MIN_VALID_FRAMES - 1
