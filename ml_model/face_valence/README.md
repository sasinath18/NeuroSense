# AFFEC face-only felt-valence model

This folder contains the deployment contract for the AFFEC face-only felt-valence
Random Forest model.

## Required model artifact

The binary model is intentionally not committed to Git because it is a private
research artifact and is supplied separately:

- `neurosense_face_valence_rf.joblib` (approximately 7.5 MB)

The backend downloads it at startup from the optional
`NEUROSENSE_FACE_VALENCE_MODEL_URL` environment variable. For a Google Drive
source, set the file sharing permission to **Anyone with the link – Viewer**,
then set that variable in Render. The URL is used only server-side.

Optional integrity check:

- `NEUROSENSE_FACE_VALENCE_MODEL_SHA256`

The original `ClipQ` and `KB` definitions are reproduced in
`backend/face_valence.py` and registered under `__main__` before `joblib.load`,
because the Colab artifact serialized those classes with `__module__="__main__"`.

## Locked inference contract

- Target: `beh_f_emotion_v` (felt valence), not stress or perceived valence.
- Window: first 4.8 seconds from video onset.
- Valid frames: `success == 1` and `confidence >= 0.8`.
- Minimum valid frames: 40; otherwise return no prediction.
- Frame feature values: float32 before aggregation.
- Aggregation: mean and standard deviation (`ddof=0`) for each of 25 OpenFace
  columns, in the exact 50-feature order defined in `backend/face_valence.py`.
- Inputs: 8 gaze/head-pose measurements and 17 AU intensity (`AU*_r`) values.
- Pupil and GSR are not inputs to this face-only model.

## Limitations and privacy

The cross-validation result is modest (mean-fold R² around 0.08; the
high/low-valence balanced accuracy was around 0.60). New-user validation has
not been performed. Never label this output as a stress score, diagnosis, or
reliable emotion classification.

The optional valence action sends a user-triggered 4.8-second camera clip to
the backend. The backend processes it in a temporary directory and deletes the
clip and OpenFace CSV when the request ends. The existing browser-local rPPG
and blink scan remains separate and does not upload its frames.
