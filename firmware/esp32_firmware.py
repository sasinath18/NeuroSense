
"""
NeuroSense - ESP32 Wearable Firmware
Board: ESP32 DevKit V1
Framework: MicroPython

Sensors:
    MAX30102 - Wrist PPG (BVP)
    GSR      - Analog skin conductance module
    TEMP     - Optional; currently disabled

Communication:
    ESP32 -> Wi-Fi -> NeuroSense FastAPI backend

Safety:
    Motor stays OFF by default.
    Do not connect a vibration motor directly to GPIO.
    Do not use localhost as the ESP32 backend address.
"""

import network
import machine
import socket
import time
import ujson
import math

from machine import Pin, I2C, ADC


# ============================================================
# 1. USER CONFIGURATION
# ============================================================

WIFI_SSID = "YOUR_WIFI_NAME"
WIFI_PASSWORD = "YOUR_WIFI_PASSWORD"

# Enter your laptop's Wi-Fi IPv4 address from ipconfig.
# Example only: 192.168.1.100
# NEVER use 127.0.0.1 here.

API_HOST = "192.168.1.100"
API_PORT = 8001

DEVICE_ID = "neurosense-esp32-01"

# Use a newly generated device token.
# Do not use the token previously exposed in a screenshot.
DEVICE_TOKEN = "PASTE_NEW_DEVICE_TOKEN_HERE"

DEMO_USER = "local-demo-user"

# Swagger showed an "authorization" header.
# This implementation sends the token as its value.
# Add "Bearer " only if your backend authentication code requires it.

AUTH_HEADER_NAME = "Authorization"
AUTH_PREFIX = ""

API_PATH = "/api/devices/{}/readings".format(DEVICE_ID)

# Sensor upload interval
UPLOAD_INTERVAL_MS = 5000

# MAX30102 sample rate
PPG_SAMPLE_RATE = 100

# Physical sensor pins
I2C_SDA_PIN = 21
I2C_SCL_PIN = 22
GSR_PIN = 34
MOTOR_PIN = 25

# Temperature sensor has not been selected yet.
TEMP_SENSOR_ENABLED = False

# Keep OFF until sensor calibration and API validation.
SEND_SENSOR_DATA = True

# Haptic hardware remains OFF until backend command integration.
HAPTIC_ENABLED = False


# ============================================================
# 2. MAX30102 REGISTER DEFINITIONS
# ============================================================

MAX30102_ADDRESS = 0x57

REG_INT_STATUS_1 = 0x00
REG_FIFO_WRITE_PTR = 0x04
REG_FIFO_OVERFLOW = 0x05
REG_FIFO_READ_PTR = 0x06
REG_FIFO_DATA = 0x07
REG_FIFO_CONFIG = 0x08
REG_MODE_CONFIG = 0x09
REG_SPO2_CONFIG = 0x0A
REG_LED1_PA = 0x0C
REG_LED2_PA = 0x0D
REG_PART_ID = 0xFF


# ============================================================
# 3. HARDWARE INITIALIZATION
# ============================================================

print("\n===================================")
print("       NEUROSENSE ESP32")
print("===================================")


# Main motor output.
# Keep motor OFF during startup.

motor = Pin(
    MOTOR_PIN,
    Pin.OUT,
    value=0
)


# GSR analog input

gsr_adc = ADC(
    Pin(GSR_PIN)
)

gsr_adc.atten(
    ADC.ATTN_11DB
)

try:
    gsr_adc.width(
        ADC.WIDTH_12BIT
    )
except AttributeError:
    pass


# I2C initialization

i2c = I2C(
    0,
    scl=Pin(I2C_SCL_PIN),
    sda=Pin(I2C_SDA_PIN),
    freq=400000
)


# ============================================================
# 4. MAX30102 DRIVER
# ============================================================

class MAX30102:

    def __init__(self, bus):

        self.i2c = bus
        self.address = MAX30102_ADDRESS

        devices = self.i2c.scan()

        if self.address not in devices:

            raise RuntimeError(
                "MAX30102 not detected at I2C address 0x57"
            )

        print("MAX30102 detected.")

        self.initialize()


    def write_register(
        self,
        register,
        value
    ):

        self.i2c.writeto_mem(
            self.address,
            register,
            bytes([value])
        )


    def read_register(self, register):

        return self.i2c.readfrom_mem(
            self.address,
            register,
            1
        )[0]


    def initialize(self):

        # Software reset

        self.write_register(
            REG_MODE_CONFIG,
            0x40
        )

        time.sleep_ms(100)

        # Clear FIFO pointers

        self.write_register(
            REG_FIFO_WRITE_PTR,
            0x00
        )

        self.write_register(
            REG_FIFO_OVERFLOW,
            0x00
        )

        self.write_register(
            REG_FIFO_READ_PTR,
            0x00
        )

        # FIFO averaging and rollover

        self.write_register(
            REG_FIFO_CONFIG,
            0x5F
        )

        # SpO2 mode
        # ADC range: 4096
        # Sample rate: 100 Hz
        # Pulse width: 411 microseconds

        self.write_register(
            REG_SPO2_CONFIG,
            0x27
        )

        # Red and IR LED current

        self.write_register(
            REG_LED1_PA,
            0x1F
        )

        self.write_register(
            REG_LED2_PA,
            0x1F
        )

        # Enable SpO2 mode

        self.write_register(
            REG_MODE_CONFIG,
            0x03
        )

        time.sleep_ms(100)

        print("MAX30102 initialized.")


    def read_available_samples(self):

        write_pointer = (
            self.read_register(
                REG_FIFO_WRITE_PTR
            ) & 0x1F
        )

        read_pointer = (
            self.read_register(
                REG_FIFO_READ_PTR
            ) & 0x1F
        )

        available = (
            write_pointer - read_pointer
        ) & 0x1F

        results = []

        for _ in range(available):

            raw = self.i2c.readfrom_mem(
                self.address,
                REG_FIFO_DATA,
                6
            )

            # Three bytes Red + three bytes IR

            red = (
                (raw[0] << 16)
                | (raw[1] << 8)
                | raw[2]
            ) & 0x3FFFF

            infrared = (
                (raw[3] << 16)
                | (raw[4] << 8)
                | raw[5]
            ) & 0x3FFFF

            results.append(
                {
                    "red": red,
                    "ir": infrared
                }
            )

        return results


# ============================================================
# 5. WIFI CONNECTION
# ============================================================

def connect_wifi():

    print("\nConnecting to Wi-Fi...")

    wlan = network.WLAN(
        network.STA_IF
    )

    wlan.active(True)

    if not wlan.isconnected():

        wlan.connect(
            WIFI_SSID,
            WIFI_PASSWORD
        )

        timeout = 25

        while (
            not wlan.isconnected()
            and timeout > 0
        ):

            print("Waiting for Wi-Fi...")

            time.sleep(1)

            timeout -= 1

    if not wlan.isconnected():

        raise RuntimeError(
            "Wi-Fi connection failed. Check SSID and password."
        )

    print("Wi-Fi connected.")

    print(
        "ESP32 IP:",
        wlan.ifconfig()[0]
    )

    print(
        "Backend address:",
        API_HOST,
        API_PORT
    )

    return wlan


# ============================================================
# 6. PULSE AND HRV FEATURE EXTRACTION
# ============================================================

ir_samples = []

last_ibi_values = []


def estimate_pulse(samples):

    if len(samples) < 900:

        return None

    segment = samples[-1000:]

    minimum = min(segment)
    maximum = max(segment)

    amplitude = maximum - minimum

    # Weak or flat PPG signal

    if amplitude < 30 or maximum < 5000:

        return None

    threshold = (
        minimum + amplitude * 0.55
    )

    peaks = []

    last_peak = -1000

    # Simple exploratory peak detector.
    # Real signal quality must be validated on the wrist.

    for i in range(
        1,
        len(segment) - 1
    ):

        is_peak = (
            segment[i] > segment[i - 1]
            and segment[i] >= segment[i + 1]
            and segment[i] > threshold
        )

        if (
            is_peak
            and (i - last_peak) >= 30
        ):

            peaks.append(i)

            last_peak = i

    if len(peaks) < 4:

        return None

    ibi_values = []

    for i in range(
        1,
        len(peaks)
    ):

        interval_samples = (
            peaks[i] - peaks[i - 1]
        )

        interval_ms = (
            interval_samples
            * 1000.0
            / PPG_SAMPLE_RATE
        )

        # Exploratory physiological limits

        if (
            300 <= interval_ms <= 1500
        ):

            ibi_values.append(
                interval_ms
            )

    if len(ibi_values) < 3:

        return None

    sorted_ibi = sorted(
        ibi_values
    )

    median_ibi = sorted_ibi[
        len(sorted_ibi) // 2
    ]

    heart_rate = (
        60000.0 / median_ibi
    )

    if not (
        40 <= heart_rate <= 180
    ):

        return None

    # RMSSD from successive beat intervals

    differences = []

    for i in range(
        1,
        len(ibi_values)
    ):

        differences.append(
            ibi_values[i]
            - ibi_values[i - 1]
        )

    if not differences:

        return None

    rmssd = math.sqrt(
        sum(
            value * value
            for value in differences
        ) / len(differences)
    )

    # This HRV estimate is preliminary.
    # 64 Hz WESAD training data and real device
    # measurements require validation before deployment.

    return {
        "heart_rate_bpm": round(
            heart_rate,
            2
        ),

        "hrv_ms": round(
            rmssd,
            2
        ),

        "ibi_ms": [
            round(value, 2)
            for value in ibi_values
        ]
    }


# ============================================================
# 7. GSR READING
# ============================================================

def read_gsr():

    value = gsr_adc.read()

    return int(value)


# ============================================================
# 8. SIMPLE HTTP REQUEST
# ============================================================

def http_request(
    method,
    path,
    payload=None
):

    if API_HOST == "192.168.1.100":

        raise ValueError(
            "Replace API_HOST with your laptop Wi-Fi IPv4 address."
        )

    address = socket.getaddrinfo(
        API_HOST,
        API_PORT
    )[0][-1]

    connection = socket.socket()

    connection.settimeout(8)

    try:

        connection.connect(address)

        headers = {

            "Host": "{}:{}".format(
                API_HOST,
                API_PORT
            ),

            "Connection": "close",

            "Content-Type": "application/json",

            "Accept": "application/json",

            "X-Demo-User": DEMO_USER
        }

        # Device token uses the Authorization
        # header shown by the current Swagger API.

        if (
            DEVICE_TOKEN
            and DEVICE_TOKEN
            != "PASTE_NEW_DEVICE_TOKEN_HERE"
        ):

            headers[AUTH_HEADER_NAME] = (
                AUTH_PREFIX + DEVICE_TOKEN
            )

        body = ""

        if payload is not None:

            body = ujson.dumps(payload)

            headers["Content-Length"] = str(
                len(body.encode())
            )

        request = (
            "{} {} HTTP/1.1\r\n".format(
                method,
                path
            )
        )

        for key, value in headers.items():

            request += "{}: {}\r\n".format(
                key,
                value
            )

        request += "\r\n"

        request += body

        connection.send(
            request.encode()
        )

        response = b""

        while True:

            chunk = connection.recv(1024)

            if not chunk:
                break

            response += chunk

        response_text = response.decode(
            "utf-8",
            "ignore"
        )

        first_line = response_text.split(
            "\r\n",
            1
        )[0]

        print(
            "Backend response:",
            first_line
        )

        parts = first_line.split()

        status_code = int(
            parts[1]
        ) if len(parts) >= 2 else 0

        return (
            status_code,
            response_text
        )

    finally:

        connection.close()


# ============================================================
# 9. SEND SENSOR DATA TO BACKEND
# ============================================================

def upload_sensor_reading(
    metrics,
    gsr_value
):

    payload = {

        "heart_rate_bpm":
            metrics["heart_rate_bpm"],

        "hrv_ms":
            metrics["hrv_ms"],

        "gsr_raw":
            gsr_value,

        # Temperature module is not selected yet.
        # Backend schema must allow null for this field.

        "skin_temperature_c": None,

        "signal_quality": "moderate",

        "source": "wearable",

        "ibi_ms": metrics["ibi_ms"]
    }

    status_code, response = http_request(
        "POST",
        API_PATH,
        payload
    )

    if status_code in (200, 201):

        print(
            "Sensor reading uploaded successfully."
        )

    elif status_code == 401:

        print(
            "Authentication failed. Check device token."
        )

    elif status_code == 422:

        print(
            "Backend rejected the sensor payload."
        )

        print(response[-700:])

    else:

        print(
            "Sensor upload failed. HTTP:",
            status_code
        )

    return status_code


# ============================================================
# 10. HAPTIC MOTOR CONTROL
# ============================================================

def set_haptic(enabled):

    global HAPTIC_ENABLED

    HAPTIC_ENABLED = bool(
        enabled
    )

    if not HAPTIC_ENABLED:

        motor.value(0)

    print(
        "Haptic feedback:",
        "ON" if HAPTIC_ENABLED else "OFF"
    )


def pulse_motor(duration_ms=300):

    if not HAPTIC_ENABLED:

        print(
            "Haptic disabled. Motor remains OFF."
        )

        motor.value(0)

        return

    motor.value(1)

    time.sleep_ms(
        duration_ms
    )

    motor.value(0)


# No automatic motor activation here.
# Backend command polling and acknowledgement
# will be connected after its exact response schema
# is verified.


# ============================================================
# 11. MAIN SENSOR LOOP
# ============================================================

def main():

    motor.value(0)

    connect_wifi()

    sensor = MAX30102(i2c)

    print("\nNeuroSense wearable is ready.")

    print(
        "Place your finger/wrist steadily on MAX30102."
    )

    print(
        "Press Ctrl+C to stop."
    )

    last_upload = time.ticks_ms()

    global ir_samples

    while True:

        try:

            new_samples = (
                sensor.read_available_samples()
            )

            for sample in new_samples:

                ir_samples.append(
                    sample["ir"]
                )

            # Keep a rolling 12-second buffer.

            if len(ir_samples) > 1200:

                ir_samples = ir_samples[-1200:]


            now = time.ticks_ms()

            elapsed = time.ticks_diff(
                now,
                last_upload
            )

            if elapsed >= UPLOAD_INTERVAL_MS:

                last_upload = now

                metrics = estimate_pulse(
                    ir_samples
                )

                gsr_value = read_gsr()

                print("\n------------------------------")

                print(
                    "GSR ADC:",
                    gsr_value
                )

                print(
                    "MAX30102 IR samples:",
                    len(ir_samples)
                )

                if metrics is None:

                    print(
                        "PPG signal not ready or insufficient quality."
                    )

                    print(
                        "No sensor record uploaded."
                    )

                    continue

                print(
                    "Heart rate:",
                    metrics["heart_rate_bpm"],
                    "BPM"
                )

                print(
                    "Exploratory HRV:",
                    metrics["hrv_ms"],
                    "ms"
                )

                if SEND_SENSOR_DATA:

                    upload_sensor_reading(
                        metrics,
                        gsr_value
                    )

                else:

                    print(
                        "Sensor upload disabled."
                    )

            time.sleep_ms(10)

        except OSError as error:

            print(
                "Hardware / network error:",
                error
            )

            motor.value(0)

            time.sleep(2)

        except Exception as error:

            print(
                "Firmware error:",
                error
            )

            motor.value(0)

            time.sleep(2)


# ============================================================
# 12. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nNeuroSense firmware stopped."
        )

        motor.value(0)
