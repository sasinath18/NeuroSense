# NeuroSense

Personalized wearable stress monitoring and recovery feedback prototype.

## Project preview

After deployment, open:

- Dashboard: `/dashboard`
- API health: `/health`
- API documentation: `/docs`

## Deploy with Render

This repository includes a Render Blueprint at `render.yaml`.

1. Open the [Render Blueprint dashboard](https://dashboard.render.com/blueprints).
2. Choose **New Blueprint Instance**.
3. Connect the GitHub repository `sasinath18/NeuroSense` and select the `main` branch.
4. Review the service settings from `render.yaml`, then apply the Blueprint.
5. When the service is live, open the service URL followed by `/dashboard`.

The Blueprint uses the free web-service plan, the Singapore region, and the repository Dockerfile. The image builds OpenFace and installs the pinned production Python dependencies from `requirements-render.txt`.

## Configure Firebase Authentication

The application code supports Firebase Google Sign-In. Configure the Firebase project and deploy credentials before enabling the production sign-in flow.

1. Create a Firebase project in the [Firebase Console](https://console.firebase.google.com/).
2. Add a Web App to that project. Copy its web configuration values.
3. In **Authentication → Sign-in method**, enable **Google**.
4. In **Authentication → Settings → Authorized domains**, add `neurosense-app.onrender.com`. Add `localhost` only if you also use local development.
5. Create a **Cloud Firestore** database in **Production mode** if you want cloud sync. This backend accesses Firestore through the server-side Admin SDK; the browser does not access Firestore directly.
6. In **Firestore → Rules**, keep direct client access denied. The Admin SDK uses service-account IAM permissions and bypasses client security rules:

   ```text
   rules_version = '2';
   service cloud.firestore {
     match /databases/{database}/documents {
       match /{document=**} {
         allow read, write: if false;
       }
     }
   }
   ```
7. In **Project settings → Service accounts**, generate a private key JSON. Keep this private file secret.
8. In Render, open the `neurosense-app` service → **Environment** and add the variables below. Save changes and redeploy.

| Render environment variable | Value |
|---|---|
| `FIREBASE_WEB_API_KEY` | Web app `apiKey` |
| `FIREBASE_WEB_AUTH_DOMAIN` | Web app `authDomain` |
| `FIREBASE_WEB_PROJECT_ID` | Web app `projectId` |
| `FIREBASE_WEB_APP_ID` | Web app `appId` |
| `FIREBASE_WEB_MESSAGING_SENDER_ID` | Web app `messagingSenderId` |
| `FIREBASE_SERVICE_ACCOUNT_JSON` | Complete service-account JSON, stored as a secret |
| `FIREBASE_STORAGE_BUCKET` | Firebase Storage bucket name for persistent per-user music uploads |
| `GOOGLE_DRIVE_API_KEY` | Google Picker API key; restrict it to your dashboard domains |
| `GOOGLE_DRIVE_APP_ID` | Google Cloud project number used by Google Picker |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token, stored as a secret |
| `TELEGRAM_BOT_USERNAME` | Telegram bot username without `@` |
| `NEUROSENSE_FACE_VALENCE_MODEL_URL` | Optional Google Drive share URL for the model artifact (Viewer access) |
| `NEUROSENSE_FACE_VALENCE_MODEL_SHA256` | Optional model SHA-256 checksum |
| `NEUROSENSE_FACE_VALENCE_WEBCAM_EXPERIMENT_ENABLED` | Keep `false`; the AFFEC selfie inference path is disabled pending validation |

The web configuration is browser-visible by design. The service-account JSON is a private Admin credential and must remain only in Render's secret environment. Never commit it to GitHub or paste it into chat.

The backend exposes only the public Firebase Web App configuration from `/api/public-config`. It never returns the Admin service-account JSON. Google Sign-In activates only when both browser configuration and backend Firebase Admin verification are ready. The backend verifies the Firebase ID token on protected API routes.


## Firebase Security Rules

The browser does not access Firestore or Firebase Storage directly. It authenticates with Firebase
and calls the NeuroSense API with a Firebase ID token; the server uses Firebase Admin SDK and
service-account IAM. The repository therefore ships restrictive client rules that deny all direct
browser reads and writes while preserving server-side Admin SDK access.

Before applying rules, confirm the target Firebase project is `neurosense-d7f69`. Deploy both
rule sets with Firebase CLI:

```bash
npx --yes firebase-tools@15.36.0 deploy --project neurosense-d7f69 --only firestore:rules,storage
```

### Publish rules through GitHub Actions

The repository also has a manual-only workflow at
`.github/workflows/deploy-firebase-rules.yml`. It deploys only the two rules files above;
ordinary pushes and pull requests do not publish Firebase security policy.

1. In Google Cloud IAM for project `neurosense-d7f69`, create a dedicated service account for
   rules deployment. Grant it these two roles:
   - **Firebase Rules Admin** (`roles/firebaserules.admin`) to create and release rulesets.
   - **Service Usage Viewer** (`roles/serviceusage.serviceUsageViewer`) so Firebase CLI can
     check whether required Google APIs are enabled. This is read-only; it cannot enable APIs.
2. Using a project owner or another principal with API-enablement permission, enable
   [Firebase Rules API](https://console.cloud.google.com/apis/library/firebaserules.googleapis.com?project=neurosense-d7f69),
   [Cloud Firestore API](https://console.cloud.google.com/apis/library/firestore.googleapis.com?project=neurosense-d7f69),
   and [Cloud Storage for Firebase API](https://console.cloud.google.com/apis/library/firebasestorage.googleapis.com?project=neurosense-d7f69)
   if any is disabled. The deploy service account is deliberately not granted API-enablement
   privileges. The workflow checks all three APIs before deployment and reports a missing
   Service Usage Viewer role separately from a disabled API.
3. Create a service-account JSON key for that dedicated account. Treat it like a password.
   Prefer Workload Identity Federation instead of a long-lived key where that is already set up.
4. In GitHub, open **Settings → Secrets and variables → Actions** for this repository and add
   repository secret `NEUROSENSE_FIREBASE_RULES_DEPLOY_CREDENTIALS_JSON` with the full JSON
   contents. Never commit the key or paste it into chat.
5. Open **Actions → Deploy Firebase Security Rules → Run workflow** and select `main`.
   The workflow validates that the credential is for `neurosense-d7f69` before deploying.

Cloud Storage for Firebase requires the project's **Blaze** pay-as-you-go pricing
plan as of February 3, 2026; the no-cost Spark plan cannot access Firebase Storage buckets.
A new Google Cloud customer may qualify for a $300 Free Trial credit for 90 days, and Google says
usage covered by that trial credit is not billed. However, when the credit is exhausted or the
90-day trial ends without a manual upgrade to a paid billing account, trial resources are stopped;
Google provides a 30-day recovery period before trial resources are permanently deleted. A paid
upgrade can incur charges for usage beyond remaining credits and applicable Free Tier limits.
Check eligibility, billing status, usage and budget alerts before linking billing. Do not assume
Firebase Storage will remain available after the trial ends without an active paid billing account.

Do not reuse the Render runtime service-account key for this workflow. Keep the dedicated deploy
account limited to the two roles above. Firebase CLI deployment overwrites the corresponding
rules currently published in the Firebase Console, so keep the repository rule files as the
source of truth.

### Automated rules verification

The repository CI also runs the Firestore and Storage rules against the Firebase Local Emulator
Suite using the `demo-neurosense-rules` demo project. The tests exercise unauthenticated and
authenticated client reads, writes, uploads and deletes and expect every direct client request to
be denied. This is an emulator test only; it does not deploy or modify production Firebase rules.

Run the same check locally (Node.js 24.12.0+ and Java 21 are required):

```bash
npm install --prefix tests/firebase-rules --no-audit --no-fund
npx --yes firebase-tools@15.36.0 emulators:exec \
  --only firestore,storage \
  --project demo-neurosense-rules \
  "npm --prefix tests/firebase-rules test"
```

These rules are intentionally deny-all for untrusted clients. Do not add direct client access
unless the product architecture is deliberately changed and the rules are reviewed separately.

## Current prototype limitations

- The dashboard uses the FastAPI backend and does not fabricate sensor readings.
- Face Scan provides browser-local facial landmarks, blink metrics and an experimental camera rPPG pulse estimate. These are wellness research outputs, not medical measurements.
- The separate AFFEC research model targets **felt valence**, not stress. Its training data used stimulus-locked video-phase windows (variable duration; median 4.61 seconds, observed 3.53–about 5.3 seconds). An arbitrary 4.8-second webcam selfie is not the same input context. Therefore the live webcam inference endpoint is OFF by default and must remain disabled until numerical feature parity and new-user validation are completed. The existing browser-local rPPG/blink scan is independent and remains available. If an explicitly approved research experiment is enabled later, its clip and derived OpenFace CSV are handled in a temporary directory and deleted after processing.
- The main API keeps scikit-learn 1.9.1 for the existing wearable model. AFFEC runs in a separate `/opt/affec-venv` environment pinned to the model's serialized scikit-learn 1.6.1, NumPy 2.4.6, pandas 2.2.3 and joblib 1.6.0. The model is loaded only by this isolated worker, not by the wearable inference process.
- The AFFEC model's reported subject-wise cross-validation is weak (mean-fold R² around 0.08; high/low-valence balanced accuracy around 0.60). The metadata explicitly says numerical feature parity is still required, and new-user validation has not been performed. Do not interpret its output as a reliable emotion classification, stress score, diagnosis or treatment recommendation.
- ESP32 firmware includes authenticated one-time pairing and verified HTTPS transport. Physical sensor accuracy and end-to-end hardware operation remain to be validated on the actual device.
- The wearable stress model is experimental. The selected hardware BOM does not include GSR, so firmware disables the GPIO34 ADC and uploads `gsr_raw: null`; no uncalibrated GSR value is presented as a measurement. The temperature sensor model/interface is not yet confirmed and firmware uploads `skin_temperature_c: null`. Missing inputs are disclosed through model input-coverage information. Do not enable GSR model features until a GSR module is fitted and a validated ADC-to-microsiemens calibration exists. MPU6050 firmware/API/dashboard support is implemented, but its physical wiring and six-axis output still require bench validation. MAX30102 signal quality/HRV, temperature sensor identification, motor-driver wiring and physical end-to-end operation also require bench validation.
- Firebase Google Sign-In and Firestore cloud persistence require the project's Firebase Web App configuration and a valid server-side Firebase Admin service account.
- The free Render service currently uses SQLite and local music storage under `/tmp`; this filesystem is ephemeral. Configure Firebase Storage for durable per-user music uploads and cloud data backup.
- Telegram emergency alerts require a configured Telegram bot token and a contact who explicitly accepts the invite. This is a check-in feature, not an emergency-response service.

## Face Scan manual QA

Use the [Face Scan Manual QA Checklist](docs/FACESCAN_MANUAL_QA.md) to verify camera permission, the 120-second browser-local scan, blink tracking, the model-pending stress state and stop/privacy behaviour before a prototype demonstration.

## Local development

Install dependencies from `requirements.txt`, then run:

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8001
```

Open `http://127.0.0.1:8001/dashboard` for the dashboard and `http://127.0.0.1:8001/docs` for API documentation.

For local Firebase testing, set the same `FIREBASE_WEB_*` variables and `FIREBASE_SERVICE_ACCOUNT_JSON` in an untracked `.env` file. Never commit `.env`.

## ESP32 firmware setup

The firmware targets an ESP32 DevKit V1 running a MicroPython build that supports
`ssl.SSLContext`, `SSLContext.load_verify_locations`, and verified client TLS.

1. Pair the device from the signed-in NeuroSense dashboard and copy its one-time device token.
2. Copy `firmware/neurosense_secrets.py.example` to `neurosense_secrets.py` on the ESP32 and enter the Wi-Fi SSID, Wi-Fi password and paired device token. Do not commit the real file.
3. On your computer, install the Mozilla CA bundle package and print its bundle path:

   ```bash
   python -m pip install certifi
   python -m certifi
   ```

   Open the printed PEM file and upload its contents to the ESP32 filesystem as `/render_ca_bundle.pem`. This is a public CA trust bundle, not a private key.
4. The firmware uses NTP to set the RTC before certificate validation. The network must permit NTP. Startup now stops immediately with a clear error when the Wi-Fi credentials, paired token, or secure transport settings are still placeholders.
5. The MPU6050 shares the I2C bus with the MAX30102: SDA GPIO21 and SCL GPIO22. Power the module only at a supply voltage supported by the exact breakout board, connect common GND, and keep ESP32 I2C logic at 3.3 V. The firmware detects the usual MPU6050 addresses (0x68/0x69); if it is absent, the wearable continues without motion telemetry.
6. The firmware uploads readings using X-Device-Token, polls authenticated haptic commands and acknowledges completed commands. Vibration durations are bounded to 100–3000 ms. Six-axis motion values are saved as telemetry and are not inputs to the current stress model.
7. For isolated local bench testing only, set USE_HTTPS = False, use the laptop LAN IP and API port, and keep the board on the same trusted Wi-Fi network. The firmware blocks plain HTTP to the public Render hostname. Restore verified HTTPS before cloud use.

The firmware fails closed when verified TLS cannot be initialized. Do not use `CERT_NONE` or disable certificate checks for cloud connections.

On a socket/hardware `OSError`, the firmware checks the Wi-Fi station state. A healthy link is left untouched; when the station has disconnected, the device resets that interface, reconnects, refreshes time using NTP and recreates the certificate-verifying TLS context. Failed reconnects are reported and retried by the main loop while the motor remains OFF.

**Current sensor configuration:** GSR is not in the selected hardware BOM, so `GSR_SENSOR_ENABLED = False`. The firmware does not initialize/read the GPIO34 ADC and sends `gsr_raw: null`; it will not turn a floating pin into a fake measurement. The temperature module is present in the parts list, but its exact model and interface are not yet confirmed, so `TEMP_SENSOR_ENABLED = False` and `skin_temperature_c` remains `null`. The dashboard labels these as “Not fitted” / “Not configured”. The MPU6050 driver is integrated on the shared I2C bus (SDA GPIO21, SCL GPIO22), supports addresses 0x68/0x69 and uploads six-axis acceleration/gyroscope telemetry to the API and dashboard. The sensor is optional at startup; if it is absent, PPG acquisition continues. Motion data is not used by the stress model. Wire and bench-test the module on the physical board before relying on these values. Do not enable GSR or temperature acquisition until the hardware and calibration/interface are verified. MAX30102 signal quality/HRV, the temperature interface, MPU6050 integration, motor-driver wiring and actual device-to-Render operation still require bench testing.

## Safety and privacy

Do not commit `.env`, Firebase service-account files, device tokens, Wi-Fi credentials, private firmware secrets, CA bundles, personal data, or raw datasets. Keep secrets in the hosting provider's environment-variable/secret settings.


## NeuroSense recovery and personalization updates

- The existing Google Sign-In Firebase UID remains the sole account key. Sensor readings, baselines, recovery history, blink summaries, music metadata and emergency-contact settings are isolated by that UID.
- ESP32 pairing uses the one-time device token and verified HTTPS firmware path.
- Wearable sensor acquisition uploads readings on its existing cadence. My Monitoring supports per-user summaries and alert checks; the personal reference is preliminary and is not clinical validation.
- Face Scan calibrates left/right eye-aspect-ratio baselines and stores derived blink summaries only. Camera frames remain in the browser for the local rPPG/blink scan.
- Haptic vibration remains independently enabled/disabled. Manual/Automatic mode applies only to music.
- Each Google UID can keep up to 10 saved music tracks (20 MB maximum per file). Durable hosted playback requires Firebase Storage. Drive tracks are streamed in the browser and must be selected again after refresh.
- Telegram replaces the legacy WhatsApp emergency-alert flow. Contacts must explicitly opt in through the invite flow, and the app exposes test-send and `/stop` controls.

### Telegram emergency-alert setup

NeuroSense uses **Telegram as its only emergency-contact notification channel**. Browser Web Push / Firebase Cloud Messaging endpoints and the old emergency-invite page are not part of the active application.

Configure `TELEGRAM_BOT_TOKEN` and `TELEGRAM_BOT_USERNAME` in Render's secret environment. The backend registers its HTTPS webhook at startup. Each emergency contact must open the invite link and press Start before receiving check-in alerts. Use the in-app test message and explain the experimental nature of the underlying stress signal; this is not an emergency service.

### Google Drive music browsing

To enable the **Browse Google Drive · extra songs** button:

1. In the same Google Cloud project used by Firebase, enable **Google Drive API** and **Google Picker API**.
2. Create a browser API key and restrict it to the dashboard's allowed web origins and the Picker/Drive APIs.
3. Find the numeric Google Cloud project number and use it as `GOOGLE_DRIVE_APP_ID`.
4. Add `GOOGLE_DRIVE_API_KEY` and `GOOGLE_DRIVE_APP_ID` to Render's environment.
5. Sign in to NeuroSense, select **Browse Google Drive · extra songs**, and grant the optional read-only Drive permission when Google prompts.

Drive-selected tracks are fetched into browser memory for playback and queueing; they do not consume the 10 saved NeuroSense slots and are not copied into the account library. They must be selected again after a page refresh or a new browser session. A durable NeuroSense library requires Firebase Storage to be configured.
