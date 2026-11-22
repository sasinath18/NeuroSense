# NeuroSense — Architecture and Implementation Handover

**Last reviewed:** 2026-11-22  
**Repository:** `sasinath18/NeuroSense`  
**Production service:** `neurosense-app` on Render

## 1. Purpose and scope

NeuroSense is a low-cost student/IT-worker stress-monitoring and recovery **prototype**. It combines an ESP32 sensor device, a browser dashboard, an experimental wearable stress model, camera-local measurements and recovery tools. It is not a medical device and must not be used for diagnosis, screening, treatment or emergency response.

The application has three intended monitoring paths:

1. **Sensors only:** ESP32 readings are collected and analysed at the configured monitoring interval.
2. **Face only:** a user-triggered camera scan provides local blink and experimental rPPG pulse measurements.
3. **Combined:** sensor monitoring and the user-triggered face scan can be used in the same session.

The current camera path does not produce a validated face-based stress score.

## 2. System architecture

```text
ESP32 + attached sensors
  |  Wi-Fi, paired device token, verified HTTPS
  v
FastAPI backend  <---- Firebase ID token ----  Browser dashboard
  |                                            |
  |                                            +-- MediaPipe face landmarks
  |                                            +-- Local blink / rPPG processing
  |                                            +-- Recovery UI and music playback
  |
  +-- Wearable feature extraction / stress-model inference
  +-- SQLite (local persistence)
  +-- Firebase Admin SDK -> Firestore (when configured)
  +-- Firebase Storage (optional durable music uploads)
  +-- Telegram Bot API (opt-in contact alerts)
  |
  +-- Haptic command queue -> ESP32
```

## 3. Main components

| Component | Location | Responsibility |
|---|---|---|
| Web dashboard | `frontend/index.html` | Single-page UI, themes, monitoring views, Face Scan, recovery, settings and client-side API calls |
| API server | `backend/main.py` | FastAPI routes, authentication, user data, sensor ingestion, model inference, alerts, history and haptic commands |
| Face-valence adapter | `backend/face_valence.py` | Separate experimental AFFEC felt-valence processing; it is not the face-stress model |
| Face-valence worker | `backend/face_valence_worker.py` | Isolated worker environment for the separate AFFEC artifact |
| ESP32 firmware | `firmware/esp32_firmware.py` | Wi-Fi connection, paired-device authentication, verified HTTPS sensor uploads and haptic command polling |
| Wearable ML | `ml_model/` | Existing wearable model and offline training/evaluation assets |
| Face-only stress trainer | `ml_model/train_face_stress_model.py` | Offline EmpathicSchool video-to-blendshape stress-training pipeline |
| Face-only alternative trainer | `ml_model/train_emowork_face_stress_model.py` | Separate EmoWork facial-feature CSV experiment; not the same feature contract as the EmpathicSchool pipeline |

## 4. Data and request flows

### Wearable readings

1. The user pairs the ESP32 from the authenticated dashboard.
2. The API issues a one-time device token. The firmware uses that token in its device requests; real secrets remain in the ESP32's private secrets file.
3. The device uploads sensor readings over verified HTTPS.
4. The backend validates the device token, stores readings, derives available features and evaluates the wearable model when sufficient data are present.
5. The dashboard retrieves readings, model output, feature-coverage information, monitoring summaries and events for the signed-in user.
6. Haptic requests are queued by the backend, polled by the device and acknowledged by firmware.

When a socket/hardware `OSError` is handled by the main loop, the firmware checks whether the Wi-Fi station is still connected. It preserves a healthy station (for example, when only the backend is temporarily unavailable). If the station link is down, it deactivates the old interface, reconnects using the configured credentials, refreshes device time through NTP and rebuilds the certificate-verifying TLS context before retrying the normal loop. A failed reconnect is logged and retried on the next loop; motor output is forced LOW during the error path.

The wearable model input-coverage response reports how many of its eight configured input features are available. Missing values may be median-imputed; the dashboard must retain the limited-input warning rather than presenting an imputed result as a full sensor measurement.

**Current hardware/firmware contract (2026-11-21):** GSR is not part of the selected hardware BOM. The ESP32 firmware therefore sets `GSR_SENSOR_ENABLED = False`, does not construct/read the GPIO34 ADC, and sends `gsr_raw: null`. Do not treat a floating ADC pin as a GSR measurement. Keep all GSR-derived model inputs excluded until a physical GSR module is added, its circuit is calibrated in the units used by the model, and the conversion is independently checked. The temperature sensor is present in the parts list, but its exact part/interface is not confirmed; `TEMP_SENSOR_ENABLED = False` and uploads `skin_temperature_c: null`. The dashboard labels absent GSR as “Not fitted” and temperature as “Not configured” instead of presenting an empty field as a fresh reading. The backend accepts PPG/IBI uploads with these optional fields null. The MPU6050 now shares the ESP32 I2C bus (SDA GPIO21, SCL GPIO22), auto-detects addresses 0x68/0x69 and verifies WHO_AM_I (0x68) before configuring the sensor. It uploads six axis values (accel_x_g/y_g/z_g, gyro_x_dps/y_dps/z_dps) to SQLite/Firestore and the dashboard. It is non-blocking when absent or when an unexpected I2C device is found, and can upload motion telemetry even when a pulse estimate is unavailable. These motion fields are deliberately excluded from the wearable stress model until separately validated. Physical wiring, orientation and measurement checks remain pending. MAX30102 signal quality/HRV, the temperature interface, the 5 V motor/MOSFET/power path and end-to-end operation still require physical validation.

### Camera Face Scan

- The camera is requested only after the user starts it.
- The 120-second blink/rPPG scan processes face landmarks and facial colour samples locally in the browser. Raw frames from this scan are not uploaded.
- The scan also prepares a separate local-only face-stress feature stream from MediaPipe blendshapes: 5 samples/second, 20-second windows, 5-second step, and per-feature mean/std with at least 60% sample coverage. The feature-window schema is aligned with the offline EmpathicSchool trainer.
- These blendshape samples/windows remain in browser memory; they are not posted to the backend, saved to account history, or used for inference. Manual scan stop clears the in-progress derived data. This is feature-extractor preparation only, not a stress estimate.
- Blink summaries are derived metrics. They can be saved to the authenticated user's history; blink rate is kept separate from wearable stress predictions.
- The rPPG pulse value is an experimental camera estimate, not a medical heart-rate measurement.
- The API's `/api/face-scan/status` response explicitly reports `stress_inference: false`.

The separate AFFEC path targets **felt valence**, not stress. It uses a short, explicitly initiated clip-upload experiment and is gated by the experiment opt-in, feature-parity verification, new-user validation, a loaded model and OpenFace availability. These gates default to off. Do not enable it as a FaceScan stress feature.

### Recovery and alerts

The dashboard provides guided breathing, recovery history, music selection/queueing, haptic controls and Telegram contact invites. Telegram contacts must opt in before receiving notifications. Telegram alerts are a check-in feature, not an emergency-response service.

## 5. ML status and validation boundary

### Wearable model

The existing wearable stress model is separate from the camera pipeline. Its output is experimental and should be shown with available-input coverage. Uncalibrated GSR and disabled temperature values must not be represented as real measurements.

### Face-only stress model

The EmpathicSchool trainer uses only MediaPipe Face Landmarker blendshape features and documented self-reported stress labels. It creates fixed 20-second windows, uses participant-held-out Leave-One-Subject-Out (LOSO) evaluation, and writes the model, feature windows and evaluation report locally.

Before training, run the aggregate-only metadata audit:

```bash
python ml_model/train_face_stress_model.py --dataset-root data/empathicschool --audit-summary-only
```

A passing audit is only a structural preflight; it does not guarantee enough usable windows or acceptable model performance. Training requires the authorised dataset release, consented video files and documented labels. The dataset is access-restricted; the user must request access and accept its Research Use License. Do not place the archive, videos, participant labels or derived participant-level data in GitHub, Render or chat.

**Current verified status (2026-11-21): the FaceScan stress trainer has been code-reviewed, but no authorised EmpathicSchool dataset was available for an actual training run. No trained NeuroSense face-stress artifact or FaceScan stress accuracy result is being claimed.** The training code is offline only and is not integrated into the live Face Scan UI/API.

The EmoWork trainer is a separate face-derived feature experiment with a different data contract and call-centre context. Its published multimodal metrics are not face-only model results and must not be attributed to NeuroSense.

## 6. Authentication, storage and secrets

- Google Sign-In uses Firebase Authentication when the Firebase Web App configuration is supplied.
- Protected API routes verify Firebase ID tokens server-side.
- The Firebase Admin SDK accesses Firestore. Browser-side Firestore rules are intentionally deny-all; do not add direct client access without an architecture and security review.
- User-owned readings, blink summaries, recovery records, music metadata and contact settings are scoped by the authenticated Firebase UID.
- Render's local filesystem (including `/tmp`) is ephemeral. Configure a suitable durable store before relying on local files for long-term music or other persistent user uploads.
- Firebase Storage requires checking the Firebase project's current billing eligibility and plan before enabling it.
- Keep Firebase service-account JSON, Telegram bot tokens, device tokens, Wi-Fi credentials, `.env` files and private firmware secrets out of Git and chat. Store server secrets in Render's secret environment and device secrets only on the device.

## 7. Deployment and operations

The application is deployed as a Docker-based Render web service from the repository's `main` branch. The configured health-check path is `/health`; the application also documents `/dashboard` and `/docs`.

**Operational verification (2026-11-22, 20:46 IST):**
- Render service `neurosense-app` is linked to `main`, uses the root Dockerfile and has the `/health` check configured.
- MPU6050 dashboard/API release commit `b6e090d4906d214bc76addbdf1c80dca84323380` reached Render status `live` at 2026-11-22 15:14:36 UTC (20:44:36 IST). The following identity-guard firmware commit `ae96449d7092f7005f321bc0b6d9a2a9f44fae1b` built successfully; its Render deployment was still `update_in_progress` at the time of this review. Do not assume the WHO_AM_I addition is live until Render marks that deployment live.
- GitHub Actions backend run `45037127955` completed successfully on `ae96449d`: ESP32 firmware syntax and frontend JavaScript syntax passed; browser-local Node tests **12 passed, 0 failed**; backend pytest **248 passed in 17.06s**; Firebase Firestore/Storage Rules Emulator passed.
- Secret Scan run `45037127910` completed successfully on the same source commit. The conditional OpenFace Docker smoke job was skipped because its build inputs were unchanged.
- The source-level test suite covers MPU6050 address detection and WHO_AM_I validation, signed six-axis conversion, absent-sensor fallback, motion-only device uploads, API bounds, SQLite additive migration, model-feature isolation and dashboard unavailable states.

Automated CI/deployment checks do not replace acceptance on the physical board or signed-in browser. The MPU6050 firmware, API, persistence and dashboard paths are implemented, but sensor wiring/orientation/real-world readings still need bench validation. Google login/UID persistence, Telegram delivery, browser camera permission and user-facing dashboard flows have not been manually confirmed in this run. ESP32 firmware still must be flashed to the board and supplied with local secrets and the CA bundle.

## 8. Remaining acceptance checks

The assistant-side software, firmware source, regression tests, security scan, Firebase emulator suite and current Render deployment have been completed and verified as described above. The following items require access to the physical prototype, owner-controlled accounts or a manual browser session.

### Physical ESP32 / hardware work (user-side)
- Flash the latest `firmware/esp32_firmware.py` to the ESP32 and create the untracked `neurosense_secrets.py` from the example with the local Wi-Fi credentials and the one-time token generated by dashboard pairing.
- Install the matching Render CA bundle on the device and verify clock synchronization plus certificate-checked HTTPS on the actual MicroPython build.
- Bench-test MAX30102 placement, PPG signal quality and repeatability; integrate and verify the temperature sensor only after its exact model/pinout is confirmed.
- Wire the MPU6050 to the shared I2C bus (SDA GPIO21, SCL GPIO22, common GND and a module-compatible supply), flash the latest firmware, confirm the detected address and WHO_AM_I check, then verify six-axis readings. Motion values are telemetry only and are not part of stress prediction.
- Verify the 5 V vibration motor through the correct MOSFET/driver and power path. Never connect the motor directly to an ESP32 GPIO. Confirm motor-off behaviour at reset, loss of Wi-Fi and device shutdown.
- GSR remains disabled because it is not in the selected BOM. Do not enable it or report ADC counts as calibrated conductance.

### Owner-account and manual acceptance (user-side)
- Verify Google Sign-In with the intended Firebase Web App and Admin service account; confirm private readings/history persist only under the signed-in user's UID.
- Verify Telegram delivery only with a real bot token and an explicitly opted-in test contact. Do not treat the feature as an emergency-response service.
- Run the browser checklist in `docs/FACESCAN_MANUAL_QA.md` for camera permission, 120-second scan, stop controls, unavailable stress state and local processing.
- Manually exercise the three modes, saved monitor history, breathing timer, music queue/playback, and mobile/responsive navigation in a signed-in browser.

### Deferred face-only stress estimate
The dedicated FaceScan stress estimate remains unavailable. Do not enable a user-facing percentage from blink rate, rPPG or the separate AFFEC felt-valence experiment. Model training and validation are deferred; existing feature-preparation code stays local-only and is not an inference pipeline.

Keep NeuroSense labelled as a research/college prototype, not a medical device, until the physical hardware, account-specific end-to-end and manual browser checks are complete.