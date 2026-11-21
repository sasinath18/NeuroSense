# Face Scan Manual QA Checklist

Use this checklist to verify the camera-only Face Scan in a supported desktop browser before the college prototype demonstration.

## Preconditions

- Open the deployed NeuroSense dashboard over HTTPS: https://neurosense-app.onrender.com/dashboard
- Sign in with the test account used for the prototype.
- Allow camera access when prompted.
- Use a well-lit room, face the light source and keep the camera near eye level.
- No ESP32 or wearable sensor needs to be connected for this camera-only test.

## Camera and preview

- [ ] Open **Face Scan** from the dashboard navigation.
- [ ] Select **Start Camera** and confirm the browser camera permission prompt.
- [ ] Confirm the live preview appears and the page reports face landmarks detected.
- [ ] Move slightly out of frame and return; confirm face tracking resumes when the face is visible.
- [ ] Deny camera permission in a separate check and confirm the page reports that camera access is unavailable rather than pretending a scan started.

## 120-second local scan

- [ ] With the face visible, select **Start 120-second Face Scan**.
- [ ] Keep both eyes visible and look naturally toward the camera during the initial eye calibration.
- [ ] Allow the scan to reach **120 / 120 sec**.
- [ ] Confirm blink count and blink rate are populated when enough valid eye-tracking time was captured.
- [ ] Confirm eye-tracking quality reports the proportion of valid tracked time; it should show a lower quality/coverage state if tracking is interrupted.
- [ ] Confirm the camera pulse estimate is shown only when the rPPG signal passes its quality checks. It is an experimental camera estimate, not a medical heart-rate measurement.
- [ ] Confirm the result grid does **not** show separate Left eye EAR or Right eye EAR cards.
- [ ] Confirm bilateral eye tracking remains active internally and the blink count changes when natural blinks are detected.

## Stress-result and research-model guardrails

- [ ] Confirm the result grid shows **Face stress estimate — Not available** while the dedicated face-stress model is not trained and validated.
- [ ] Confirm the page explains that blink rate and pulse alone are not used as a stress score.
- [ ] Confirm the optional AFFEC experiment is labelled **felt valence, not stress** and remains disabled while its validation gates are incomplete.
- [ ] Do not treat the wearable stress score, AFFEC felt-valence output or camera pulse estimate as a face-only stress result.

## Stop and privacy behaviour

- [ ] Select **Stop Analysis** and confirm the 120-second measurement stops.
- [ ] Select **Stop Camera** and confirm the camera indicator turns OFF and the preview stops.
- [ ] Repeat a short preview check after granting camera permission again.
- [ ] The ordinary 120-second blink/rPPG scan is browser-local; it does not upload its raw camera frames. The separate AFFEC clip-upload experiment is a different path and must remain disabled until its validation gates are satisfied.

## Record the result

Record the browser/version, date, lighting conditions, whether camera permission was granted, scan completion, eye-tracking quality, whether blink detection appeared responsive, and any visible error message. Do not record or upload a participant's raw video as part of this QA report.

A passing manual QA check confirms that the prototype UI and local measurement flow behave as expected on that test machine. It does **not** validate face-based stress inference or clinical accuracy.
