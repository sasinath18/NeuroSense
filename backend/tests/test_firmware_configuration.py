"""Hardware-independent tests for the ESP32's optional sensor contract.

The firmware imports MicroPython modules and performs hardware setup at import
time, so these tests compile only the small functions under test from its AST.
"""
from __future__ import annotations

import ast
import pytest
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



def test_motor_boot_configuration_starts_with_gpio_low():
    tree = _firmware_tree()
    motor_assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "motor"
            for target in node.targets
        )
    )

    assert isinstance(motor_assignment.value, ast.Call)
    assert isinstance(motor_assignment.value.func, ast.Name)
    assert motor_assignment.value.func.id == "Pin"
    keyword_values = {
        item.arg: item.value
        for item in motor_assignment.value.keywords
    }
    assert ast.literal_eval(keyword_values["value"]) == 0


def test_local_haptic_disabled_gate_keeps_motor_low_without_sleeping():
    tree = _firmware_tree()

    class FakeMotor:
        def __init__(self):
            self.values = []

        def value(self, level):
            self.values.append(level)

    class FakeTime:
        @staticmethod
        def sleep_ms(_duration):
            raise AssertionError("Disabled haptics must never energize or sleep.")

    motor = FakeMotor()
    pulse_motor = _load_function(
        tree,
        "pulse_motor",
        {
            "HAPTIC_ENABLED": False,
            "motor": motor,
            "time": FakeTime(),
        },
    )

    pulse_motor(500)

    assert motor.values == [0]


def test_haptic_command_duration_is_clamped_to_100_through_3000_ms():
    import json

    tree = _firmware_tree()
    calls = []
    pulses = []

    response_body = json.dumps({
        "commands": [
            {"command_id": "short", "duration_ms": 1},
            {"command_id": "long", "duration_ms": 9000},
        ]
    })
    response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n" + response_body

    def fake_http_request(method, path, payload=None):
        calls.append((method, path, payload))
        return (200, response if method == "GET" else "HTTP/1.1 200 OK")

    poll = _load_function(
        tree,
        "poll_and_process_haptic_commands",
        {
            "DEVICE_TOKEN": "configured-device-token",
            "COMMAND_PATH": "/api/devices/test/commands",
            "HAPTIC_ENABLED": True,
            "http_request": fake_http_request,
            "ujson": json,
            "pulse_motor": pulses.append,
        },
    )

    poll()

    assert pulses == [100, 3000]
    assert [call[0] for call in calls] == ["GET", "POST", "POST"]
    assert calls[1][1] == "/api/devices/test/commands/short/ack"
    assert calls[2][1] == "/api/devices/test/commands/long/ack"


def test_server_haptic_commands_respect_local_disabled_gate():
    import json

    tree = _firmware_tree()
    motor_values = []
    pulses = []

    response_body = json.dumps({
        "commands": [{"command_id": "ignored", "duration_ms": 500}]
    })
    response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n" + response_body

    def fake_http_request(method, path, payload=None):
        return (200, response if method == "GET" else "HTTP/1.1 200 OK")

    poll = _load_function(
        tree,
        "poll_and_process_haptic_commands",
        {
            "DEVICE_TOKEN": "configured-device-token",
            "COMMAND_PATH": "/api/devices/test/commands",
            "HAPTIC_ENABLED": False,
            "http_request": fake_http_request,
            "ujson": json,
            "pulse_motor": pulses.append,
            "motor": type("FakeMotor", (), {
                "value": staticmethod(motor_values.append)
            })(),
        },
    )

    poll()

    assert pulses == []
    assert motor_values == [0]



def _valid_firmware_config(**overrides):
    return {
        "WIFI_SSID": "test-network",
        "WIFI_PASSWORD": "test-password",
        "DEVICE_TOKEN": "test-device-token",
        "USE_HTTPS": True,
        "API_PORT": 443,
        "API_HOST": "neurosense-app.onrender.com",
        **overrides,
    }


def test_firmware_configuration_rejects_missing_wifi_ssid():
    tree = _firmware_tree()
    validate = _load_function(
        tree,
        "validate_configuration",
        _valid_firmware_config(WIFI_SSID=""),
    )

    with pytest.raises(RuntimeError, match="WIFI_SSID"):
        validate()


def test_firmware_configuration_rejects_missing_wifi_password():
    tree = _firmware_tree()
    validate = _load_function(
        tree,
        "validate_configuration",
        _valid_firmware_config(WIFI_PASSWORD="YOUR_WIFI_PASSWORD"),
    )

    with pytest.raises(RuntimeError, match="WIFI_PASSWORD"):
        validate()


def test_firmware_configuration_rejects_unpaired_device_token():
    tree = _firmware_tree()
    validate = _load_function(
        tree,
        "validate_configuration",
        _valid_firmware_config(DEVICE_TOKEN="PASTE_NEW_DEVICE_TOKEN_HERE"),
    )

    with pytest.raises(RuntimeError, match="one-time device token"):
        validate()


def test_firmware_configuration_blocks_public_cloud_http_downgrade():
    tree = _firmware_tree()
    validate = _load_function(
        tree,
        "validate_configuration",
        _valid_firmware_config(USE_HTTPS=False),
    )

    with pytest.raises(RuntimeError, match="Plain HTTP is not permitted"):
        validate()


def test_firmware_configuration_requires_standard_cloud_https_port():
    tree = _firmware_tree()
    validate = _load_function(
        tree,
        "validate_configuration",
        _valid_firmware_config(API_PORT=8443),
    )

    with pytest.raises(RuntimeError, match="port 443"):
        validate()


def test_firmware_configuration_allows_verified_cloud_https_and_local_http():
    tree = _firmware_tree()

    cloud_validate = _load_function(
        tree,
        "validate_configuration",
        _valid_firmware_config(),
    )
    cloud_validate()

    local_validate = _load_function(
        tree,
        "validate_configuration",
        _valid_firmware_config(
            USE_HTTPS=False,
            API_HOST="192.168.1.25",
            API_PORT=8000,
        ),
    )
    local_validate()


def test_tls_context_requires_certificate_validation_and_configured_ca_bundle():
    tree = _firmware_tree()
    captured = {}

    class FakeTLSContext:
        def __init__(self):
            self.verify_mode = None

        def load_verify_locations(self, cafile):
            captured["cafile"] = cafile

    class FakeSSL:
        PROTOCOL_TLS_CLIENT = "tls-client"
        CERT_REQUIRED = "certificate-required"

        @staticmethod
        def SSLContext(protocol):
            captured["protocol"] = protocol
            context = FakeTLSContext()
            captured["context"] = context
            return context

    initialize_tls = _load_function(
        tree,
        "initialize_tls",
        {
            "USE_HTTPS": True,
            "CA_CERT_PATH": "/render_ca_bundle.pem",
            "ssl": FakeSSL(),
            "TLS_CONTEXT": None,
        },
    )

    initialize_tls()

    assert captured["protocol"] == FakeSSL.PROTOCOL_TLS_CLIENT
    assert captured["context"].verify_mode == FakeSSL.CERT_REQUIRED
    assert captured["cafile"] == "/render_ca_bundle.pem"


def test_disabling_https_does_not_construct_a_tls_context():
    tree = _firmware_tree()

    class UnexpectedSSL:
        def SSLContext(self, _protocol):
            raise AssertionError("TLS context must not be created for local HTTP.")

    namespace = {
        "USE_HTTPS": False,
        "ssl": UnexpectedSSL(),
        "TLS_CONTEXT": object(),
    }
    initialize_tls = _load_function(tree, "initialize_tls", namespace)

    initialize_tls()

    assert namespace["TLS_CONTEXT"] is None
