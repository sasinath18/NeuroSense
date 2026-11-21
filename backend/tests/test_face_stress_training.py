import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import ml_model.train_face_stress_model as face_stress
from ml_model.train_face_stress_model import (
    audit_dataset,
    summarize_audit,
    build_feature_windows,
    build_label_intervals,
    label_for_window,
    parse_stress_label,
    read_stress_labels,
    evaluate_loso,
    require_complete_loso_evaluation,
    run as run_empathicschool_training,
)

from ml_model.train_emowork_face_stress_model import (
    build_face_windows,
    discover_dataset_pairs,
    load_aligned_face_stress,
    select_face_feature_columns,
    timestamp_to_seconds,
    run as run_emowork_training,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", 0),
        ("1", 1),
        ("2", 2),
        ("no stress", 0),
        ("low stress", 1),
        ("high stress", 2),
        ("unknown", None),
        ("not-a-label", None),
    ],
)
def test_parse_documented_stress_labels(raw, expected):
    assert parse_stress_label(raw) == expected


def test_read_single_column_stress_csv_with_header_and_unknown(tmp_path: Path):
    path = tmp_path / "Stress_level.csv"
    path.write_text(
        "stress_level\n0\n1\nunknown\n2\n",
        encoding="utf-8",
    )

    assert read_stress_labels(path) == [0, 1, None, 2]


def test_read_single_column_stress_csv_without_header(tmp_path: Path):
    path = tmp_path / "Stress_level.csv"
    path.write_text("0\n1\n2\nunknown\n", encoding="utf-8")

    assert read_stress_labels(path) == [0, 1, 2, None]


def _make_fake_video_capture(monkeypatch, duration_seconds=540.0, fps=30.0):
    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, property_id):
            if property_id == face_stress.cv2.CAP_PROP_FPS:
                return fps
            if property_id == face_stress.cv2.CAP_PROP_FRAME_COUNT:
                return duration_seconds * fps
            return 0.0

        def release(self):
            return None

    monkeypatch.setattr(face_stress.cv2, "VideoCapture", lambda _path: FakeCapture())


def test_dataset_audit_reports_task_labels_and_loso_readiness(tmp_path: Path, monkeypatch):
    task_dir = tmp_path / "S7" / "T2"
    task_dir.mkdir(parents=True)
    (task_dir / "V.mp4").write_bytes(b"synthetic video placeholder")
    (task_dir / "Stress_level.csv").write_text(
        "stress_level\n0\n1\n2\n1\n0\n",
        encoding="utf-8",
    )
    _make_fake_video_capture(monkeypatch, duration_seconds=540.0)

    report = audit_dataset(tmp_path)

    assert report["audit_only"] is True
    assert report["paired_task_count"] == 1
    assert report["valid_task_count"] == 1
    assert report["participant_count"] == 1
    assert report["class_counts"] == {"0": 2, "1": 2, "2": 1}
    assert report["tasks"][0]["duration_seconds"] == pytest.approx(540.0)
    assert report["ready_for_exploratory_training"] is False
    assert any("at least 4" in warning for warning in report["warnings"])


def test_dataset_audit_reports_videos_when_every_task_label_is_missing(tmp_path: Path):
    task_dir = tmp_path / "S7" / "T2"
    task_dir.mkdir(parents=True)
    (task_dir / "V.mp4").write_bytes(b"synthetic video placeholder")

    report = audit_dataset(tmp_path)

    assert report["video_candidate_count"] == 1
    assert report["paired_task_count"] == 0
    assert report["valid_task_count"] == 0
    assert report["unpaired_video_task_count"] == 1
    assert report["unpaired_video_tasks"] == [{
        "participant_id": "S7",
        "task_id": "T2",
        "video_file": "V.mp4",
        "reason": "No task-level Stress_level.csv found.",
    }]
    assert report["ready_for_exploratory_training"] is False
    assert any("No video task has a matching Stress_level.csv" in item for item in report["warnings"])


def test_dataset_audit_flags_unpaired_video_and_invalid_interval_layout(tmp_path: Path, monkeypatch):
    valid_dir = tmp_path / "S7" / "T3"
    invalid_dir = tmp_path / "S8" / "T2"
    valid_dir.mkdir(parents=True)
    invalid_dir.mkdir(parents=True)
    (valid_dir / "V.mp4").write_bytes(b"synthetic video placeholder")
    (valid_dir / "Stress_level.csv").write_text(
        "stress_level\n0\n1\n2\n",
        encoding="utf-8",
    )
    (invalid_dir / "V.mp4").write_bytes(b"synthetic video placeholder")
    (invalid_dir / "Stress_level.csv").write_text(
        "stress_level\n0\n1\n",
        encoding="utf-8",
    )

    durations = {"S7": 300.0, "S8": 540.0}

    class FakeCapture:
        def __init__(self, path):
            self.duration = durations[Path(path).parent.parent.name]

        def isOpened(self):
            return True

        def get(self, property_id):
            if property_id == face_stress.cv2.CAP_PROP_FPS:
                return 30.0
            if property_id == face_stress.cv2.CAP_PROP_FRAME_COUNT:
                return self.duration * 30.0
            return 0.0

        def release(self):
            return None

    monkeypatch.setattr(face_stress.cv2, "VideoCapture", FakeCapture)

    report = audit_dataset(tmp_path)

    assert report["valid_task_count"] == 1
    assert report["invalid_task_count"] == 1
    assert report["tasks"][0]["status"] == "valid"
    assert report["tasks"][1]["status"] == "invalid"
    assert "Only shortened T3" in report["tasks"][1]["validation_error"]


def test_summary_audit_omits_participant_and_task_identifiers():
    report = {
        "dataset": "EmpathicSchool",
        "video_candidate_count": 2,
        "paired_task_count": 2,
        "valid_task_count": 1,
        "invalid_task_count": 1,
        "unpaired_video_task_count": 0,
        "participant_count": 1,
        "class_counts": {"0": 2, "1": 3, "2": 1},
        "class_count": 3,
        "ready_for_exploratory_training": False,
        "warnings": ["1 video/label pair failed validation."],
        "tasks": [{"participant_id": "S7", "task_id": "T2"}],
        "unpaired_video_tasks": [{"participant_id": "S8", "video_file": "V.mp4"}],
    }

    summary = summarize_audit(report)

    assert summary["participant_count"] == 1
    assert summary["class_counts"] == {"0": 2, "1": 3, "2": 1}
    assert summary["participant_identifiers_included"] is False
    assert summary["task_details_included"] is False
    assert "tasks" not in summary
    assert "unpaired_video_tasks" not in summary
    assert "S7" not in json.dumps(summary)
    assert "S8" not in json.dumps(summary)
    assert "V.mp4" not in json.dumps(summary)


def test_audit_summary_only_run_emits_aggregate_report_without_landmarker(
    tmp_path: Path, monkeypatch, capsys
):
    def unexpected_download(_path):
        pytest.fail("Summary-only audit must never download the MediaPipe asset")

    monkeypatch.setattr(face_stress, "ensure_landmarker_model", unexpected_download)
    args = SimpleNamespace(
        dataset_root=tmp_path,
        audit_only=False,
        audit_summary_only=True,
    )

    run_empathicschool_training(args)

    output = json.loads(capsys.readouterr().out)
    assert output["audit_only"] is True
    assert output["summary_only"] is True
    assert output["video_candidate_count"] == 0
    assert output["participant_count"] == 0
    assert "tasks" not in output
    assert "unpaired_video_tasks" not in output
    assert output["participant_identifiers_included"] is False


def test_audit_only_run_does_not_download_or_require_landmarker(tmp_path: Path, monkeypatch, capsys):
    task_dir = tmp_path / "S7" / "T2"
    task_dir.mkdir(parents=True)
    (task_dir / "V.mp4").write_bytes(b"synthetic video placeholder")
    (task_dir / "Stress_level.csv").write_text(
        "stress_level\n0\n1\n2\n1\n0\n",
        encoding="utf-8",
    )
    _make_fake_video_capture(monkeypatch, duration_seconds=540.0)
    monkeypatch.setattr(
        face_stress,
        "ensure_landmarker_model",
        lambda _path: (_ for _ in ()).throw(AssertionError("model download must not run")),
    )

    run_empathicschool_training(SimpleNamespace(dataset_root=tmp_path, audit_only=True))

    output = json.loads(capsys.readouterr().out)
    assert output["audit_only"] is True
    assert output["paired_task_count"] == 1


def test_five_interval_labels_use_published_task_timing():
    intervals = build_label_intervals([0, 1, 2, 1, 0], 540)

    assert intervals == [
        (0.0, 90.0, 0),
        (90.0, 210.0, 1),
        (210.0, 330.0, 2),
        (330.0, 450.0, 1),
        (450.0, 540.0, 0),
    ]


def test_five_interval_labels_reject_video_shorter_than_nine_minutes():
    with pytest.raises(ValueError, match="at least 540 seconds"):
        build_label_intervals([0, 1, 2, 1, 0], 539.99)


def test_five_interval_labels_leave_recording_tail_unlabelled():
    intervals = build_label_intervals([0, 1, 2, 1, 0], 600)

    assert intervals == [
        (0.0, 90.0, 0),
        (90.0, 210.0, 1),
        (210.0, 330.0, 2),
        (330.0, 450.0, 1),
        (450.0, 540.0, 0),
    ]
    assert label_for_window(540.0, 560.0, intervals) is None


def test_shortened_t3_labels_use_consecutive_two_minute_intervals():
    intervals = build_label_intervals([0, 2, 1], 300, task_id="T3")

    assert intervals == [
        (0.0, 120.0, 0),
        (120.0, 240.0, 2),
        (240.0, 300.0, 1),
    ]


def test_shortened_t3_leaves_unlabelled_tail_unassigned():
    intervals = build_label_intervals([0, 2], 420, task_id="T3")

    assert intervals == [
        (0.0, 120.0, 0),
        (120.0, 240.0, 2),
    ]


def test_incomplete_non_t3_labels_fail_closed():
    with pytest.raises(ValueError, match="Only shortened T3"):
        build_label_intervals([0, 2, 1], 300, task_id="T2")


def test_inconsistent_shortened_t3_duration_fails_closed():
    with pytest.raises(ValueError, match="too short"):
        build_label_intervals([0, 2, 1], 200, task_id="T3")


def test_feature_window_is_skipped_when_it_crosses_label_boundary():
    intervals = [(0.0, 20.0, 0), (20.0, 40.0, 2)]

    assert label_for_window(0.0, 10.0, intervals) == 0
    assert label_for_window(20.0, 30.0, intervals) == 2
    assert label_for_window(15.0, 25.0, intervals) is None


def test_blendshape_windows_are_grouped_and_labelled_without_wearable_features():
    samples = [
        (float(second), {"jawOpen": second / 300.0, "eyeBlinkLeft": 0.1})
        for second in range(300)
    ]

    rows = build_feature_windows(
        samples=samples,
        duration_seconds=300.0,
        labels=[0, 2, 1],
        subject_id="S7",
        task_id="T3",
        window_seconds=60.0,
        step_seconds=60.0,
        sample_fps=1.0,
    )

    assert [row["stress_label"] for row in rows] == [0, 0, 2, 2, 1]
    assert {row["participant_id"] for row in rows} == {"S7"}
    assert {row["task_id"] for row in rows} == {"T3"}
    assert "blendshape_jawOpen_mean" in rows[0]
    assert "heart_rate_bpm" not in rows[0]
    assert "gsr_raw" not in rows[0]


def test_training_preflight_stops_before_downloading_landmarker_when_dataset_not_ready(
    tmp_path: Path, monkeypatch
):
    task_dir = tmp_path / "S7" / "T2"
    task_dir.mkdir(parents=True)
    (task_dir / "V.mp4").write_bytes(b"synthetic video placeholder")
    (task_dir / "Stress_level.csv").write_text(
        "stress_level\n0\n1\n2\n1\n0\n",
        encoding="utf-8",
    )
    _make_fake_video_capture(monkeypatch, duration_seconds=540.0)

    def unexpected_download(_path):
        pytest.fail("MediaPipe must not download before the dataset passes preflight")

    monkeypatch.setattr(face_stress, "ensure_landmarker_model", unexpected_download)
    args = SimpleNamespace(
        dataset_root=tmp_path,
        audit_only=False,
        model_path=tmp_path / "model.joblib",
        report_path=tmp_path / "evaluation.json",
        output_csv=tmp_path / "windows.csv",
        landmarker_model=tmp_path / "face_landmarker.task",
        sample_fps=5.0,
        window_seconds=20.0,
        step_seconds=5.0,
    )

    with pytest.raises(ValueError, match="Dataset is not ready for exploratory training"):
        run_empathicschool_training(args)


def _emowork_video_frame(count: int = 150) -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame({
        "timestamp": [index / 30.0 for index in range(count)],
        "rotation_x": [0.1] * count,
        "rotation_y": [0.2] * count,
        "rotation_z": [0.3] * count,
        "translation_x": [1.0] * count,
        "translation_y": [2.0] * count,
        "translation_z": [3.0] * count,
        "total_translation": [0.5] * count,
        "blink": [0.0] * count,
        "gaze_x": [0.1] * count,
        "gaze_y": [0.2] * count,
        "left_lip_corner_x": [0.3] * count,
        "left_lip_corner_y": [0.4] * count,
        "right_lip_corner_x": [0.5] * count,
        "right_lip_corner_y": [0.6] * count,
        "error_type": [0] * count,
        "pupil_diameter": [3.0] * count,
        "gsr_raw": [1500] * count,
        "heart_rate_bpm": [72.0] * count,
        "audio_mfcc_1": [0.2] * count,
    })


def test_emowork_feature_selector_excludes_pupil_gsr_hr_and_audio():
    frame = _emowork_video_frame()
    error_column, selected = select_face_feature_columns(frame)

    assert error_column == "error_type"
    assert "rotation_x" in selected
    assert "gaze_x" in selected
    assert "left_lip_corner_x" in selected
    assert "right_lip_corner_y" in selected
    assert not any(
        forbidden in column.lower()
        for column in selected
        for forbidden in ("pupil", "gsr", "heart", "audio", "mfcc", "eda", "bvp")
    )


def test_emowork_timestamp_conversion_accepts_relative_seconds_and_epoch_ms():
    import pandas as pd

    seconds = timestamp_to_seconds(pd.Series([0.0, 1.0, 2.0]), "relative")
    milliseconds = timestamp_to_seconds(
        pd.Series([1_700_000_000_000, 1_700_000_001_000]),
        "epoch",
    )

    long_relative_seconds = timestamp_to_seconds(
        pd.Series([20_000.0, 20_000.1, 20_000.2]), "long relative seconds"
    )
    relative_milliseconds = timestamp_to_seconds(
        pd.Series([100_000.0, 100_100.0, 100_200.0]), "relative milliseconds"
    )

    assert seconds.tolist() == [0.0, 1.0, 2.0]
    assert milliseconds.iloc[1] - milliseconds.iloc[0] == pytest.approx(1.0)
    assert long_relative_seconds.iloc[-1] - long_relative_seconds.iloc[0] == pytest.approx(0.2)
    assert relative_milliseconds.iloc[-1] - relative_milliseconds.iloc[0] == pytest.approx(0.2)


def test_emowork_loader_aligns_face_only_frames_with_stress_labels(tmp_path: Path):
    import pandas as pd

    video = _emowork_video_frame()
    video.loc[10, "error_type"] = 2
    labels = pd.DataFrame({
        "timestamp": [index / 10.0 for index in range(50)],
        "stress": [float(5 + (index % 10)) for index in range(50)],
        "arousal": [8.0] * 50,
        "valence": [4.0] * 50,
    })
    video_path = tmp_path / "video.csv"
    label_path = tmp_path / "stress.csv"
    video.to_csv(video_path, index=False)
    labels.to_csv(label_path, index=False)

    aligned, face_columns = load_aligned_face_stress(
        video_path, label_path, label_tolerance_ms=60.0
    )

    # One invalid face row is filtered, and the final frame is outside the
    # 60 ms stress-label alignment tolerance.
    assert len(aligned) == 148
    assert "pupil_diameter" not in face_columns
    assert "gsr_raw" not in face_columns
    assert "heart_rate_bpm" not in face_columns
    assert "_stress_score" in aligned.columns
    assert aligned["_stress_score"].between(1, 20).all()


def test_emowork_windows_keep_subject_and_continuous_stress_target(tmp_path: Path):
    import pandas as pd

    video = _emowork_video_frame()
    labels = pd.DataFrame({
        "timestamp": [index / 10.0 for index in range(50)],
        "stress": [12.0] * 50,
    })
    video_path = tmp_path / "video.csv"
    label_path = tmp_path / "stress.csv"
    video.to_csv(video_path, index=False)
    labels.to_csv(label_path, index=False)

    aligned, face_columns = load_aligned_face_stress(
        video_path, label_path, label_tolerance_ms=60.0
    )
    rows = build_face_windows(
        aligned,
        face_columns,
        participant_id="1",
        condition="c1",
        window_seconds=5.0,
        step_seconds=2.5,
        min_frame_ratio=0.5,
    )

    assert len(rows) == 1
    assert rows[0]["participant_id"] == "1"
    assert rows[0]["condition"] == "c1"
    assert rows[0]["stress_score"] == pytest.approx(12.0)
    assert "rotation_x_mean" in rows[0]
    assert "pupil_diameter_mean" not in rows[0]


def test_emowork_dataset_pair_discovery_requires_matching_video_and_stress_csv(tmp_path: Path):
    label_dir = tmp_path / "LABEL" / "1" / "c1"
    video_dir = tmp_path / "CSV" / "VIDEO" / "1"
    label_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    label_file = label_dir / "stress.csv"
    video_file = video_dir / "c1.csv"
    label_file.write_text("timestamp,stress\n0,7\n", encoding="utf-8")
    video_file.write_text("timestamp,rotation_x\n0,0.1\n", encoding="utf-8")

    pairs = discover_dataset_pairs(tmp_path)

    assert pairs == [("1", "c1", video_file, label_file)]


def test_emowork_pair_discovery_does_not_confuse_condition_with_participant(tmp_path: Path):
    label_dir = tmp_path / "LABEL" / "1" / "c1"
    video_dir = tmp_path / "CSV" / "VIDEO" / "2"
    label_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    (label_dir / "stress.csv").write_text(
        "timestamp,stress\n0,7\n", encoding="utf-8"
    )
    (video_dir / "c1.csv").write_text(
        "timestamp,rotation_x\n0,0.1\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Expected one video-feature CSV"):
        discover_dataset_pairs(tmp_path)


def test_emowork_pair_discovery_accepts_labels_plural_directory(tmp_path: Path):
    label_dir = tmp_path / "LABELS" / "2" / "c2"
    video_dir = tmp_path / "CSV" / "VIDEO" / "2"
    label_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    label_file = label_dir / "stress.csv"
    video_file = video_dir / "c2.csv"
    label_file.write_text("timestamp,stress\n0,9\n", encoding="utf-8")
    video_file.write_text("timestamp,rotation_x\n0,0.2\n", encoding="utf-8")

    pairs = discover_dataset_pairs(tmp_path)

    assert pairs == [("2", "c2", video_file, label_file)]

def test_emowork_v3_csv_mirror_layout_is_supported(tmp_path: Path):
    """Match the official v3 CSV/LABEL_CSV and CSV/VIDEO_csv hierarchy."""
    label_dir = tmp_path / "CSV" / "LABEL_CSV" / "1" / "C1"
    video_dir = tmp_path / "CSV" / "VIDEO_csv" / "1"
    label_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    label_file = label_dir / "stress.csv"
    video_file = video_dir / "c1.csv"
    label_file.write_text(
        "Timestamp,stress\n1700000000000,7\n",
        encoding="utf-8",
    )
    video_file.write_text(
        "timestamp,rotation_x\n1700000000000,0.1\n",
        encoding="utf-8",
    )

    pairs = discover_dataset_pairs(tmp_path)

    assert pairs == [("1", "c1", video_file, label_file)]


def test_emowork_training_runs_end_to_end_on_isolated_synthetic_fixture(tmp_path: Path):
    """Smoke-test the entire trainer without using or exporting participant data."""
    import numpy as np
    import joblib

    dataset_root = tmp_path / "emowork"
    participants = 6
    epoch_start_ms = 1_700_000_000_000

    for participant_index in range(1, participants + 1):
        participant = str(participant_index)
        video_dir = dataset_root / "CSV" / "VIDEO" / participant
        label_dir = dataset_root / "LABELS" / participant / "c1"
        video_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)

        frame_count = 26 * 30
        frame_index = np.arange(frame_count, dtype=float)
        video = _emowork_video_frame(count=frame_count)
        video["timestamp"] = epoch_start_ms + participant_index * 10_000 + frame_index * (1000.0 / 30.0)
        video["rotation_x"] = 0.05 * np.sin(frame_index / 15.0) + participant_index * 0.01
        video["rotation_y"] = 0.04 * np.cos(frame_index / 18.0)
        video["translation_x"] = frame_index / frame_count
        video["blink"] = ((frame_index.astype(int) % 90) == 0).astype(float)
        video.to_csv(video_dir / "c1.csv", index=False)

        label_index = np.arange(261, dtype=float)
        stress = 5.0 + ((label_index.astype(int) // 20 + participant_index) % 12)
        labels = pd.DataFrame({
            "Timestamp": epoch_start_ms + participant_index * 10_000 + label_index * 100.0,
            "stress": stress,
            # These non-face labels must never enter the model feature matrix.
            "arousal": 8.0,
            "valence": 4.0,
        })
        labels.to_csv(label_dir / "stress.csv", index=False)

    args = SimpleNamespace(
        dataset_root=dataset_root,
        model_path=tmp_path / "model.joblib",
        report_path=tmp_path / "evaluation.json",
        output_csv=tmp_path / "windows.csv",
        window_seconds=5.0,
        step_seconds=2.5,
        label_tolerance_ms=250.0,
        min_frame_ratio=0.5,
    )

    run_emowork_training(args)

    assert args.model_path.is_file()
    assert args.report_path.is_file()
    assert args.output_csv.is_file()

    report = json.loads(args.report_path.read_text(encoding="utf-8"))
    assert report["dataset"] == "EmoWork"
    assert report["target"] == "self-reported stress score (1–20)"
    assert report["evaluation"]["validation"] == "Leave-One-Participant-Out (LOSO)"
    assert report["evaluation"]["participants"] == participants
    assert report["evaluation"]["evaluated_windows"] >= 50
    assert not any(
        forbidden in column.lower()
        for column in report["feature_columns"]
        for forbidden in ("pupil", "gsr", "heart", "audio", "arousal", "valence")
    )

    artifact = joblib.load(args.model_path)
    assert artifact["input_modality"] == "facial landmark-derived features only"
    assert artifact["target"] == "self-reported stress score (1–20)"
    assert len(artifact["feature_columns"]) == report["feature_count"]




def _balanced_empathicschool_loso_frame(participants=4):
    rows = []
    for participant_index in range(participants):
        participant = f"S{participant_index + 1}"
        for window_index in range(8):
            label = window_index % 2
            rows.append({
                "participant_id": participant,
                "task_id": "T2",
                "window_start_seconds": float(window_index * 20),
                "stress_label": label,
                "blendshape_jawOpen_mean": float(label + participant_index * 0.01),
                "blendshape_eyeBlinkLeft_std": float((window_index % 3) * 0.1 + participant_index * 0.02),
            })
    return pd.DataFrame(rows)


def test_empathicschool_loso_evaluates_every_window_with_participant_holdout():
    frame = _balanced_empathicschool_loso_frame()
    report = evaluate_loso(
        frame,
        ["blendshape_jawOpen_mean", "blendshape_eyeBlinkLeft_std"],
    )

    assert report["validation"] == "Leave-One-Subject-Out (LOSO)"
    assert report["participants"] == 4
    assert report["windows"] == len(frame)
    assert report["evaluated_windows"] == len(frame)
    assert report["not_evaluated_windows"] == 0
    assert len(report["folds"]) == 4
    assert sum(fold["test_windows"] for fold in report["folds"]) == len(frame)
    require_complete_loso_evaluation(report)


def test_empathicschool_loso_reports_partial_coverage_when_a_fold_has_one_training_class():
    rows = []
    for participant in ("S1", "S2", "S3", "S4"):
        for window_index in range(6):
            label = 0 if participant == "S1" else 1
            rows.append({
                "participant_id": participant,
                "task_id": "T2",
                "window_start_seconds": float(window_index * 20),
                "stress_label": label,
                "blendshape_jawOpen_mean": float(label + window_index * 0.01),
                "blendshape_eyeBlinkLeft_std": float(window_index * 0.02),
            })
    frame = pd.DataFrame(rows)

    report = evaluate_loso(
        frame,
        ["blendshape_jawOpen_mean", "blendshape_eyeBlinkLeft_std"],
    )

    assert report["evaluated_windows"] == 18
    assert report["not_evaluated_windows"] == 6
    assert len(report["folds"]) == 3
    with pytest.raises(ValueError, match="did not cover every labelled window"):
        require_complete_loso_evaluation(report)


def test_empathicschool_loso_single_class_heldout_fold_does_not_emit_metric_warning():
    import warnings

    frame = _balanced_empathicschool_loso_frame()
    frame.loc[frame["participant_id"] == "S1", "stress_label"] = 0

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        report = evaluate_loso(
            frame,
            ["blendshape_jawOpen_mean", "blendshape_eyeBlinkLeft_std"],
        )

    assert report["evaluated_windows"] == len(frame)
    assert report["not_evaluated_windows"] == 0
    assert not any(
        issubclass(warning.category, UserWarning)
        for warning in captured
    )


def test_empathicschool_loso_gate_rejects_incomplete_or_empty_reports():
    with pytest.raises(ValueError, match="did not cover every labelled window"):
        require_complete_loso_evaluation({
            "windows": 12,
            "evaluated_windows": 10,
            "not_evaluated_windows": 2,
        })
    with pytest.raises(ValueError, match="did not cover every labelled window"):
        require_complete_loso_evaluation({
            "windows": 0,
            "evaluated_windows": 0,
            "not_evaluated_windows": 0,
        })

