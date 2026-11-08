from pathlib import Path

import pytest

from ml_model.train_face_stress_model import (
    build_feature_windows,
    build_label_intervals,
    label_for_window,
    parse_stress_label,
    read_stress_labels,
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
