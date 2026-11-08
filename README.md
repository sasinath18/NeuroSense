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

The web configuration is browser-visible by design. The service-account JSON is a private Admin credential and must remain only in Render's secret environment. Never commit it to GitHub or paste it into chat.

The backend exposes only the public Firebase Web App configuration from `/api/public-config`. It never returns the Admin service-account JSON. Google Sign-In activates only when both browser configuration and backend Firebase Admin verification are ready. The backend verifies the Firebase ID token on protected API routes.

## Current prototype limitations

- The dashboard reads backend health and, after secure authentication is configured, loads dashboard overview, wearable history, settings and recovery history from FastAPI. It does not generate sample sensor values.
- Face Scan uses MediaPipe Face Landmarker in the browser to draw facial landmarks and a mesh when the user explicitly starts the camera. Frames are processed locally and are not uploaded to the NeuroSense backend. This feature does not infer emotion or stress.
- Face Scan displays a lighting advisory and a browser-local brightness estimate. This estimate is not a lux measurement and cannot guarantee detection accuracy.
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
3. Obtain the trusted CA certificate bundle that validates the configured NeuroSense HTTPS host and copy it to `/render_ca_bundle.pem` on the ESP32.
4. The firmware uses NTP to set the RTC before certificate validation. The network must permit NTP.
5. The firmware uploads readings using `X-Device-Token`, polls authenticated haptic commands and acknowledges completed commands. Vibration durations are bounded to 100–3000 ms.
6. For isolated local bench testing only, set `USE_HTTPS = False`, use the laptop LAN IP and API port, and keep the board on the same trusted Wi-Fi network. Restore verified HTTPS before cloud use.

The firmware fails closed when verified TLS cannot be initialized. Do not use `CERT_NONE` or disable certificate checks for cloud connections. Physical MAX30102 signal quality, GSR unit conversion, temperature integration, driver wiring and live device-to-Render operation still require hardware verification.

## Safety and privacy

Do not commit `.env`, Firebase service-account files, device tokens, Wi-Fi credentials, private firmware secrets, CA bundles, personal data, or raw datasets. Keep secrets in the hosting provider's environment-variable/secret settings.
