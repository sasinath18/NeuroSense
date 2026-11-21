"""Hardware-independent tests for the ESP32's optional sensor contract.

The firmware imports MicroPython modules and performs hardware setup at import
time, so these tests compile only the small functions under test from its AST.
"""
from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FIRMWARE_PATH = REPO_ROOT / "firmware" / "esp32_firmware.py"


def _firmware_tree() -> ast.Module:
    return ast.parse(FIRMWARE_PATH.read_text(encoding="utf-8"))


def _module_constant(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                return ast.literal_eval(node.value)
    raise AssertionError(f"Firmware constant {name!r} was not found.")


def _load_function(tree: ast.Module, name: str, namespace: dict):
    function = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    isolated_module = ast.Module(body=[function], type_ignores=[])
    code = compile(ast.fix_missing_locations(isolated_module), str(FIRMWARE_PATH), "exec")
    exec(code, namespace)
    return namespace[name]


def test_unselected_gsr_and_unconfirmed_temperature_are_disabled_by_default():
    tree = _firmware_tree()

    assert _module_constant(tree, "GSR_SENSOR_ENABLED") is False
    assert _module_constant(tree, "TEMP_SENSOR_ENABLED") is False


def test_gsr_adc_is_only_constructed_inside_the_explicit_enable_gate():
    tree = _firmware_tree()
    guarded_initializers = []

    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name):
            if node.test.id != "GSR_SENSOR_ENABLED":
                continue
            guarded_initializers.extend(
                child for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "ADC"
            )

    assert guarded_initializers, "GSR ADC must be initialized only behind its opt-in flag."


def test_disabled_gsr_returns_none_without_reading_a_floating_adc():
    tree = _firmware_tree()

    class UnexpectedADC:
        def read(self):
            raise AssertionError("ADC must not be read when GSR is disabled.")

    read_gsr = _load_function(
        tree,
        "read_gsr",
        {"GSR_SENSOR_ENABLED": False, "gsr_adc": UnexpectedADC()},
    )

    assert read_gsr() is None


def test_explicitly_enabled_gsr_reads_the_optional_adc():
    tree = _firmware_tree()

    class FakeADC:
        def read(self):
            return 1234

    read_gsr = _load_function(
        tree,
        "read_gsr",
        {"GSR_SENSOR_ENABLED": True, "gsr_adc": FakeADC()},
    )

    assert read_gsr() == 1234


def test_sensor_upload_keeps_gsr_and_unconfirmed_temperature_null():
    tree = _firmware_tree()
    captured = {}

    def fake_http_request(method, path, payload):
        captured.update({"method": method, "path": path, "payload": payload})
        return 201, "created"

    upload = _load_function(
        tree,
        "upload_sensor_reading",
        {"API_PATH": "/api/devices/test/readings", "http_request": fake_http_request},
    )
    metrics = {
        "heart_rate_bpm": 72.0,
        "hrv_ms": 35.0,
        "ibi_ms": [810.0, 830.0, 820.0],
    }

    assert upload(metrics, None) == 201
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/devices/test/readings"
    assert captured["payload"]["gsr_raw"] is None
    assert captured["payload"]["skin_temperature_c"] is None
    assert captured["payload"]["ibi_ms"] == metrics["ibi_ms"]
