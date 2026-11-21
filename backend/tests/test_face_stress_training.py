from pathlib import Path

import pandas as pd
import pytest

from ml_model.train_face_stress_model import (
    build_feature_windows,
    build_label_intervals,
    label_for_window,
    parse_stress_label,
    read_stress_labels,
)

from ml_model.train_emowork_face_stress_model import (
    build_face_windows,
    discover_dataset_pairs,
    load_aligned_face_stress,
    select_face_feature_columns,
    timestamp_to_seconds,
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


def test_five_interval_labels_use_published_task_timing():
    intervals = build_label_intervals([0, 1, 2, 1, 0], 540)

    assert intervals == [
        (0.0, 90.0, 0),
        (90.0, 210.0, 1),
        (210.0, 330.0, 2),
        (330.0, 450.0, 1),
        (450.0, 540.0, 0),
    ]


def test_feature_window_is_skipped_when_it_crosses_label_boundary():
    intervals = [(0.0, 20.0, 0), (20.0, 40.0, 2)]

    assert label_for_window(0.0, 10.0, intervals) == 0
    assert label_for_window(20.0, 30.0, intervals) == 2
    assert label_for_window(15.0, 25.0, intervals) is None


def test_blendshape_windows_are_grouped_and_labelled_without_wearable_features():
    samples = [
        (float(second), {"jawOpen": second / 40.0, "eyeBlinkLeft": 0.1})
        for second in range(40)
    ]

    rows = build_feature_windows(
        samples=samples,
        duration_seconds=40.0,
        labels=[0, 2],
        subject_id="S7",
        task_id="T2",
        window_seconds=10.0,
        step_seconds=10.0,
        sample_fps=1.0,
    )

    assert [row["stress_label"] for row in rows] == [0, 0, 2, 2]
    assert {row["participant_id"] for row in rows} == {"S7"}
    assert {row["task_id"] for row in rows} == {"T2"}
    assert "blendshape_jawOpen_mean" in rows[0]
    assert "heart_rate_bpm" not in rows[0]
    assert "gsr_raw" not in rows[0]


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

    assert seconds.tolist() == [0.0, 1.0, 2.0]
    assert milliseconds.iloc[1] - milliseconds.iloc[0] == pytest.approx(1.0)


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

    assert len(aligned) == 149
    assert "pupil_diameter" not in face_columns
    assert "gsr_raw" not in face_columns
    assert "heart_rate_bpm" not in face_columns
    assert "stress_score" in aligned.columns
    assert aligned["stress_score"].between(1, 20).all()


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
