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



def test_wifi_reconnect_preserves_an_already_connected_station():
    tree = _firmware_tree()

    class ConnectedStation:
        def isconnected(self):
            return True

        def active(self, _enabled):
            raise AssertionError("A healthy Wi-Fi station must not be reset.")

    station = ConnectedStation()

    def unexpected_reconnect():
        raise AssertionError("A healthy Wi-Fi station must not reconnect.")

    ensure = _load_function(
        tree,
        "ensure_wifi_connected",
        {
            "connect_wifi": unexpected_reconnect,
            "time": type("FakeTime", (), {
                "sleep": staticmethod(lambda _seconds: None)
            })(),
        },
    )

    assert ensure(station) is station


def test_wifi_reconnect_resets_disconnected_station_then_rebuilds_tls():
    tree = _firmware_tree()
    events = []

    class DisconnectedStation:
        def isconnected(self):
            return False

        def active(self, enabled):
            events.append(("active", enabled))

    class FakeTime:
        @staticmethod
        def sleep(seconds):
            events.append(("sleep", seconds))

    replacement = object()

    def fake_connect_wifi():
        events.append(("connect_wifi",))
        return replacement

    ensure = _load_function(
        tree,
        "ensure_wifi_connected",
        {
            "connect_wifi": fake_connect_wifi,
            "time": FakeTime(),
        },
    )

    result = ensure(DisconnectedStation())

    assert result is replacement
    assert events == [
        ("active", False),
        ("sleep", 1),
        ("connect_wifi",),
    ]


def test_wifi_reconnect_reports_failure_instead_of_returning_stale_station():
    tree = _firmware_tree()

    class DisconnectedStation:
        def isconnected(self):
            return False

        def active(self, _enabled):
            pass

    def fail_reconnect():
        raise RuntimeError("Wi-Fi connection failed")

    ensure = _load_function(
        tree,
        "ensure_wifi_connected",
        {
            "connect_wifi": fail_reconnect,
            "time": type("FakeTime", (), {
                "sleep": staticmethod(lambda _seconds: None)
            })(),
        },
    )

    with pytest.raises(RuntimeError, match="Wi-Fi connection failed"):
        ensure(DisconnectedStation())


def test_main_attempts_wifi_reconnect_only_from_oserror_recovery_path():
    tree = _firmware_tree()
    main = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    oserror_handlers = [
        handler
        for node in ast.walk(main)
        if isinstance(node, ast.Try)
        for handler in node.handlers
        if isinstance(handler.type, ast.Name)
        and handler.type.id == "OSError"
    ]

    assert oserror_handlers, "Firmware main loop must retain its OSError recovery handler."
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ensure_wifi_connected"
        for handler in oserror_handlers
        for node in ast.walk(handler)
    ), "An OSError must trigger a Wi-Fi link-state check and reconnect attempt."



def _load_class(tree: ast.Module, name: str, namespace: dict):
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )
    isolated_module = ast.Module(body=[cls], type_ignores=[])
    code = compile(ast.fix_missing_locations(isolated_module), str(FIRMWARE_PATH), "exec")
    exec(code, namespace)
    return namespace[name]


def test_mpu6050_is_enabled_as_an_optional_shared_i2c_sensor():
    tree = _firmware_tree()
    assert _module_constant(tree, "MPU6050_SENSOR_ENABLED") is True


def test_mpu6050_detects_default_and_alternate_i2c_addresses_and_configures_ranges():
    tree = _firmware_tree()

    class FakeBus:
        def __init__(self, addresses):
            self.addresses = addresses
            self.writes = []

        def scan(self):
            return self.addresses

        def writeto_mem(self, address, register, data):
            self.writes.append((address, register, data[0]))

    class FakeTime:
        @staticmethod
        def sleep_ms(_duration):
            pass

    MPU6050 = _load_class(tree, "MPU6050", {"time": FakeTime()})
    for address in (0x68, 0x69):
        bus = FakeBus([0x57, address])
        sensor = MPU6050(bus)
        assert sensor.address == address
        assert bus.writes == [
            (address, 0x6B, 0x00),
            (address, 0x1A, 0x03),
            (address, 0x1B, 0x00),
            (address, 0x1C, 0x00),
        ]


def test_mpu6050_converts_signed_accelerometer_and_gyroscope_samples():
    tree = _firmware_tree()

    def word(value):
        return int(value & 0xFFFF).to_bytes(2, "big")

    sample = b"".join(
        word(value)
        for value in (
            16384, -8192, 0, 0,  # acceleration + unused temperature
            131, -262, 393,      # gyroscope
        )
    )

    class FakeBus:
        def scan(self):
            return [0x68]

        def writeto_mem(self, _address, _register, _data):
            pass

        def readfrom_mem(self, address, register, count):
            assert address == 0x68
            assert register == 0x3B
            assert count == 14
            return sample

    class FakeTime:
        @staticmethod
        def sleep_ms(_duration):
            pass

    MPU6050 = _load_class(tree, "MPU6050", {"time": FakeTime()})
    reading = MPU6050(FakeBus()).read()

    assert reading == {
        "accel_x_g": 1.0,
        "accel_y_g": -0.5,
        "accel_z_g": 0.0,
        "gyro_x_dps": 1.0,
        "gyro_y_dps": -2.0,
        "gyro_z_dps": 3.0,
    }


def test_absent_mpu6050_does_not_block_firmware_startup():
    tree = _firmware_tree()

    class FakeBus:
        def scan(self):
            return [0x57]  # MAX30102 only; no MPU6050 address

    class FakeTime:
        @staticmethod
        def sleep_ms(_duration):
            pass

    MPU6050 = _load_class(tree, "MPU6050", {"time": FakeTime()})
    initialise = _load_function(
        tree,
        "initialize_optional_mpu6050",
        {
            "MPU6050_SENSOR_ENABLED": True,
            "MPU6050": MPU6050,
        },
    )

    assert initialise(FakeBus()) is None


def test_optional_motion_read_returns_none_when_sensor_is_unavailable_or_errors():
    tree = _firmware_tree()

    class FailedSensor:
        def read(self):
            raise OSError("I2C read failed")

    read_motion = _load_function(
        tree,
        "read_optional_motion",
        {},
    )

    assert read_motion(None) is None
    assert read_motion(FailedSensor()) is None


def test_motion_only_firmware_upload_includes_axes_and_keeps_ppg_fields_null():
    tree = _firmware_tree()
    captured = {}

    def fake_http_request(method, path, payload):
        captured.update(method=method, path=path, payload=payload)
        return 201, "created"

    upload = _load_function(
        tree,
        "upload_sensor_reading",
        {
            "API_PATH": "/api/devices/test/readings",
            "http_request": fake_http_request,
        },
    )
    motion = {
        "accel_x_g": 0.1,
        "accel_y_g": -0.2,
        "accel_z_g": 0.98,
        "gyro_x_dps": 2.5,
        "gyro_y_dps": -1.5,
        "gyro_z_dps": 0.0,
    }

    assert upload(None, None, motion) == 201
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/devices/test/readings"
    assert captured["payload"]["heart_rate_bpm"] is None
    assert captured["payload"]["hrv_ms"] is None
    assert captured["payload"]["ibi_ms"] is None
    assert captured["payload"]["signal_quality"] == "poor"
    for axis, value in motion.items():
        assert captured["payload"][axis] == value


def test_main_initializes_optional_mpu6050_and_uploads_motion_without_ppg():
    tree = _firmware_tree()
    main = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    calls = [
        node.func.id
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "initialize_optional_mpu6050" in calls
    assert "read_optional_motion" in calls

    upload_calls = [
        node for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "upload_sensor_reading"
    ]
    assert upload_calls
    assert any(
        isinstance(node, ast.Name) and node.id == "motion_values"
        for node in ast.walk(upload_calls[0])
    )


def test_mpu6050_data_is_stored_outside_the_wearable_model_feature_set():
    tree = _firmware_tree()
    upload = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "upload_sensor_reading"
    )
    payload_keys = {
        item.value.value
        for node in ast.walk(upload)
        if isinstance(node, ast.Dict)
        for item in node.keys
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }
    assert {
        "accel_x_g", "accel_y_g", "accel_z_g",
        "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
    }.issubset(payload_keys)
