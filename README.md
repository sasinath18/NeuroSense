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

## Current prototype limitations

- The dashboard frontend is currently a standalone UI preview and still uses demo/simulated display values. It is not yet connected to the live FastAPI sensor endpoints.
- ESP32 hardware integration and live sensor validation are not yet complete.
- Firebase Authentication and Firestore are not configured in this deployment.
- The SQLite database is configured at `/tmp/neurosense.db` on the free preview service. This filesystem is ephemeral, so stored records can be lost when the service restarts or redeploys.
- The WESAD-trained model is an experimental research prototype, not clinically validated and not intended for diagnosis or treatment.

## Local development

Install dependencies from `requirements.txt`, then run:

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8001
```

Open `http://127.0.0.1:8001/dashboard` for the dashboard and `http://127.0.0.1:8001/docs` for API documentation.

## Safety and privacy

Do not commit `.env`, Firebase service-account files, device tokens, Wi-Fi credentials, personal data, or raw datasets. Keep secrets in the hosting provider's environment-variable/secret settings.
