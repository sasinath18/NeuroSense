"""Regression tests for the browser-local Face Scan UI contract.

These tests do not claim to validate stress inference. They protect the honest
"not available" state until a separately trained and validated face-stress
model is integrated.
"""
from __future__ import annotations

import re
from html import unescape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HTML_PATH = ROOT / "frontend" / "index.html"


def face_scan_source() -> tuple[str, str]:
    html = HTML_PATH.read_text(encoding="utf-8")
    start = html.find('<section id="page-face"')
    if start < 0:
        raise AssertionError("Face Scan page section was not found.")
    end = html.find("</section>", start)
    if end < 0:
        raise AssertionError("Face Scan page section is not closed.")
    section = html[start : end + len("</section>")]

    scripts = re.findall(
        r"<script>(.*?)</script>", html, flags=re.IGNORECASE | re.DOTALL
    )
    if len(scripts) != 1:
        raise AssertionError(
            f"Expected one inline application script, found {len(scripts)}."
        )
    return section, scripts[0]


def metric_card_texts(section: str) -> list[str]:
    cards = re.findall(
        r'<div class="face-scan-metric">(.*?)</div>',
        section,
        flags=re.DOTALL,
    )
    return [
        re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", card))).strip()
        for card in cards
    ]


def test_face_scan_shows_explicit_unavailable_stress_state_until_model_is_ready():
    section, _ = face_scan_source()
    cards = metric_card_texts(section)

    stress_cards = [card for card in cards if "Face stress estimate" in card]
    assert len(stress_cards) == 1
    assert "Not available" in stress_cards[0]
    assert "Dedicated face-stress model pending" in stress_cards[0]


def test_face_scan_hides_separate_ear_numbers_but_keeps_tracking_quality():
    section, _ = face_scan_source()
    cards = metric_card_texts(section)

    assert not any("Left eye EAR" in card for card in cards)
    assert not any("Right eye EAR" in card for card in cards)
    assert any("Both-eye tracking coverage" in card for card in cards)
    assert "leftEyeEarValue" not in section
    assert "rightEyeEarValue" not in section


def test_blink_counter_still_uses_both_eyes_for_internal_blink_detection():
    _, script = face_scan_source()
    start = script.find("function updateBlinkCounter(landmarks, now)")
    assert start >= 0, "Bilateral blink-counter function is missing."
    next_function = re.search(
        r"\n(?:async\s+)?function\s+",
        script[start + len("function updateBlinkCounter(landmarks, now)"):],
    )
    assert next_function, "Could not isolate updateBlinkCounter() body."
    body_start = start + len("function updateBlinkCounter(landmarks, now)")
    body_end = body_start + next_function.start()
    body = script[body_start:body_end]

    assert "eyeAspectRatio(landmarks, [33, 160, 158, 133, 153, 144])" in body
    assert "eyeAspectRatio(landmarks, [362, 385, 387, 263, 373, 380])" in body
    assert "calibration.leftOpen" in body
    assert "calibration.rightOpen" in body
    assert "meanOpenRatio" in body
    assert "leastOpenRatio" in body
    assert "blinkClosing" in body


def test_affec_felt_valence_remains_separate_from_face_stress():
    section, _ = face_scan_source()

    assert "AFFEC Felt-Valence Test" in section
    assert "This optional AFFEC research experiment estimates felt valence, not stress." in section
    assert "It stays disabled until the model's stimulus-window feature parity" in section
    assert re.search(
        r'<span id="faceValenceValue"\s+hidden>--</span>',
        section,
    )
    cards = metric_card_texts(section)
    assert not any("Felt valence estimate" in card for card in cards)
