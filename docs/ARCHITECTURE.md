# NeuroSense — Architecture and Implementation Handover

**Last reviewed:** 2026-11-21  
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

The wearable model input-coverage response reports how many of its eight configured input features are available. Missing values may be median-imputed; the dashboard must retain the limited-input warning rather than presenting an imputed result as a full sensor measurement.

**Sensor-unit limitation:** the current ESP32 GSR value is a raw ADC count (0–4095), whereas WESAD training EDA uses calibrated microsiemens. Those units are not interchangeable. Live GSR mean/delta are excluded from the wearable model until the actual circuit has a validated conversion. The temperature channel is currently disabled in firmware and uploads `null`. MAX30102 signal quality, HR/HRV extraction, temperature sensing, motor wiring and live device operation still require physical validation.

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

**Operational verification (2026-11-21, 23:15 IST):**
- Render service `neurosense-app` is linked to `main`, uses the Dockerfile at the repository root, and has the HTTP health-check path `/health` configured.
- The latest observed deployment at review time used commit `1a2f9373a29107fffdfdc4f8a7a4bdafd9e5d2ee`; Render reported both its build and deploy as successful and the deployment status as `live`.
- GitHub Actions backend tests and Firebase Rules Emulator tests completed successfully for run `44931192064`. The optional OpenFace Docker smoke job was skipped because the relevant image inputs had not changed.
- The full-history Gitleaks run `44931192089` completed successfully; its report scanned 426 commits and found no leaks.
- The Render log query returned two older error-level build records from 2026-11-17 (a Docker build could not find the root `Dockerfile`). Those records are historical and are not associated with the 2026-11-21 live deployment.
- A narrow request-log query for `/health` during the latest deployment window returned no records, so this snapshot does not claim a separate application-level HTTP probe result.

This is a point-in-time operational check, not an end-to-end acceptance test. It does not establish that Google login, Firestore persistence, Telegram delivery, browser camera access or physical ESP32 integration has passed an end-to-end user test. Re-check the Render Deploys page for current live state before a demonstration.

## 8. Remaining acceptance checks

### Dataset/model work
- Obtain authorised EmpathicSchool v3 access and read/accept its Research Use License.
- Extract the released consented data locally and run the aggregate-only audit.
- Train only if the audit passes; inspect per-participant LOSO results, class support, failure counts and data quality.
- Implement and validate a live browser feature extractor that exactly matches the trained model contract before any model integration.
- Add calibrated confidence/abstention handling and independent-user/student-context validation before considering a user-facing face-stress estimate.

### Account and cloud configuration
- Verify Firebase Google Sign-In, Admin credentials and per-user Firestore persistence in the target project.
- Decide whether durable music storage is required and configure Firebase Storage only after checking billing.
- Verify Telegram bot credentials and an explicitly opted-in test contact.
- Configure Google Drive Picker credentials only if Drive browsing is required.

### Hardware and end-to-end
- Validate ESP32 pairing, Wi-Fi, TLS certificate verification, sensor uploads and haptic acknowledgements.
- Bench-test MPU6050, MAX30102, temperature sensing and the 5 V vibration motor/MOSFET/power path.
- Calibrate GSR before enabling any GSR-derived model feature.
- Run signed-in browser tests for monitoring modes, saved history, camera permissions, blink tracking, breathing timing, music playback/queue, Telegram test delivery and responsive layouts.

The project should remain labelled as a research/college prototype until these hardware, dataset, independent-validation and user-flow checks are complete.
