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

The Blueprint uses the free web-service plan, the Singapore region, Python 3.13.14, and the lightweight production dependency list in `requirements-render.txt`.

## Configure Firebase Authentication

The application code supports Firebase Google Sign-In, but Firebase must be configured in your own Firebase project before the sign-in button becomes active.

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
| `FIREBASE_SERVICE_ACCOUNT_JSON` | Complete service-account JSON, pasted as compact JSON; mark as secret |
| `FIREBASE_STORAGE_BUCKET` | Firebase Storage bucket name for persistent per-user music uploads; mark as non-public configuration |
| `TWILIO_ACCOUNT_SID` | Twilio account SID; mark as secret |
| `TWILIO_AUTH_TOKEN` | Twilio auth token; mark as secret |
| `TWILIO_WHATSAPP_FROM` | Approved WhatsApp sender in `whatsapp:+...` format |
| `GOOGLE_DRIVE_API_KEY` | Google Picker API key; restrict it to your dashboard domains |
| `GOOGLE_DRIVE_APP_ID` | Google Cloud project number used by Google Picker |

The web configuration is browser-visible by design. The service-account JSON is a private Admin credential and must remain only in Render's secret environment. Never commit it to GitHub or paste it into chat.

The backend exposes only the public Firebase Web App configuration from `/api/public-config`. It never returns the Admin service-account JSON. Google Sign-In activates only when both browser configuration and backend Firebase Admin verification are ready. The backend verifies the Firebase ID token on protected API routes.

## Current prototype limitations

- The dashboard reads backend health and, after secure authentication is configured, loads dashboard overview, wearable history, settings and recovery history from FastAPI. It does not generate sample sensor values.
- Face Scan uses MediaPipe Face Landmarker in the browser to draw facial landmarks and a mesh when the user explicitly starts the camera.
- The optional 120-second Face-only scan samples small forehead/cheek skin regions locally and applies a POS-style remote photoplethysmography (rPPG) signal pipeline. A camera-estimated pulse rate is displayed only when the 30-second signal window passes basic continuity, brightness and spectral-peak checks.
- Webcam frames and extracted RGB traces remain in browser memory for the scan; they are not uploaded to the NeuroSense backend or persisted by this feature.
- Face-only pulse estimates are experimental and non-medical. Webcam lighting, motion, frame rate, skin reflectance and device characteristics can reduce accuracy. The browser brightness indicator is not a lux measurement and cannot guarantee rPPG quality.
- A face-only stress score is intentionally not displayed yet. It requires a separately trained and independently validated model; facial expression alone is not treated as a stress measurement.
- ESP32 hardware integration and live sensor validation are not yet complete.
- Firebase project credentials must be configured before production sign-in and private account data can be used.
- The SQLite database is configured at `/tmp/neurosense.db` on the free preview service. This filesystem is ephemeral, so stored records can be lost when the service restarts or redeploys.
- The WESAD-trained model is an experimental research prototype, not clinically validated and not intended for diagnosis or treatment.

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
5. The firmware uploads readings using `X-Device-Token`, polls authenticated haptic commands and acknowledges completed commands. Vibration durations are bounded to 100–3000 ms.
6. For isolated local bench testing only, set `USE_HTTPS = False`, use the laptop LAN IP and API port, and keep the board on the same trusted Wi-Fi network. The firmware blocks plain HTTP to the public Render hostname. Restore verified HTTPS before cloud use.

The firmware fails closed when verified TLS cannot be initialized. Do not use `CERT_NONE` or disable certificate checks for cloud connections.

**Sensor validation limitation:** the ESP32 sends raw GSR ADC counts (0–4095), while the WESAD training data uses calibrated EDA measurements. These values are not interchangeable. The temperature channel is currently disabled and uploads `null`; the model's missing-value imputer handles that field, but it does not validate the physical sensors. MAX30102 signal quality, HR/HRV extraction, GSR calibration to the training unit, temperature integration, motor-driver wiring and live device-to-Render operation all require hardware testing before treating stress estimates as meaningful.

## Safety and privacy

Do not commit `.env`, Firebase service-account files, device tokens, Wi-Fi credentials, private firmware secrets, CA bundles, personal data, or raw datasets. Keep secrets in the hosting provider's environment-variable/secret settings.


## NeuroSense recovery and personalization updates

- The existing Google Sign-In Firebase UID remains the sole account key. Sensor readings, baselines, recovery history, blink summaries, music metadata and emergency-contact settings are isolated by that UID; the app does not create a second profile.
- ESP32 pairing remains Wi-Fi based, using the existing one-time device token and verified HTTPS firmware path.
- Wearable summaries continue on the existing three-hour interval. The personal reference endpoint evaluates the most recent 72 hours: two distinct days and at least ten reliable readings allow a preliminary reference; three days and at least twenty readings are recommended. This is not a new ML model or a clinical validation.
- Face Scan counts blinks locally using eye-landmark geometry. Only the derived count, scan duration and blink rate can be saved; camera frames remain in the browser.
- Haptic vibration remains independently enabled/disabled. Manual/Automatic mode applies only to music: Manual lets the user select a single track or playlist; Automatic attempts the one saved default track after breathing.
- Music files can be uploaded from local storage or imported from Google Drive with the optional Google Picker integration (20 MB maximum per file). For durable hosted playback, configure `FIREBASE_STORAGE_BUCKET`; without it, the server uses local fallback storage, which may be lost on Render restarts/redeploys. Drive files are copied into the NeuroSense library; the player does not stream from the Drive website.
- Warning sound and browser notifications are separate preferences. Browser notifications require browser permission.
- WhatsApp check-in alerts require an opted-in emergency contact plus Twilio WhatsApp credentials. The backend sends only after three elevated model readings within 30 minutes and applies a six-hour cooldown. The model remains experimental; this is not an emergency service.

### WhatsApp setup

Configure `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, and an approved `TWILIO_WHATSAPP_FROM` sender in the hosting provider's secret environment. The user must save and explicitly enable an E.164-format emergency contact in Settings. Use the in-app test message before relying on alerts. Never commit Twilio credentials.


### Google Drive music import

To enable the **Import audio from Google Drive** button:

1. In the same Google Cloud project used by Firebase, enable **Google Drive API** and **Google Picker API**.
2. Create a browser API key and restrict it to the dashboard's allowed web origins and the Picker/Drive APIs.
3. Find the numeric Google Cloud project number and use it as `GOOGLE_DRIVE_APP_ID`.
4. Add `GOOGLE_DRIVE_API_KEY` and `GOOGLE_DRIVE_APP_ID` to Render's environment.
5. Sign in to NeuroSense, select **Import audio from Google Drive**, and grant the optional read-only Drive permission when Google prompts.

Drive files are copied into the user's NeuroSense music library; playback then uses NeuroSense storage. The app does not stream directly from the Drive website. A durable music library requires Firebase Storage to be configured.
