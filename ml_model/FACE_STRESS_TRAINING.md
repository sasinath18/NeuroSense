# NeuroSense — Face-only Stress Model

This training path is separate from the WESAD wearable model. It uses only
MediaPipe Face Landmarker blendshape scores extracted from face video. HR, BVP,
EDA, skin temperature, accelerometer values and wearable-model outputs are
excluded from the feature set.

## Dataset and access

The selected research dataset is EmpathicSchool:

- Dataset DOI: https://doi.org/10.5281/zenodo.7372497
- Research paper: https://doi.org/10.1038/s41597-025-05812-0
- Stress labels are self-assessments at roughly two-minute intervals.
- Video recordings are released only for participants who consented to video sharing.
- The Zenodo record currently marks the files as restricted. A user must sign in,
  request access and personally accept the Research Use License. NeuroSense does
  not download, redistribute or commit the restricted dataset.

The dataset agreement is for research use and includes privacy/security obligations.
Do not upload the downloaded archive, participant videos, labels or derived feature
CSV to GitHub or a public service.

## Expected extracted folder layout

Extract the authorized dataset under data/empathicschool/. The script searches
for task videos and a matching task-level Stress_level.csv:

    data/
      empathicschool/
        S7/
          T2/
            V.mp4
            Stress_level.csv
          T3/
            V.mp4
            Stress_level.csv
        S8/
          T2/
            V.mp4
            Stress_level.csv

Only tasks with both an available, consented MP4 video and documented stress
labels are included. The script reads the documented classes 0 (no stress), 1
(low stress), 2 (high stress); unknown labels are skipped. If the provider's
download uses another CSV structure, inspect it and adapt the label reader
rather than guessing the column or class mapping.

## Install and train

From the repository root, create/activate a Python virtual environment and install
the project's dependencies:

    python -m pip install -r requirements.txt
    python ml_model/train_face_stress_model.py

The script downloads the same MediaPipe Face Landmarker task asset used by the
browser frontend into ml_model/assets/face_landmarker.task if it is not already
present. It samples face video at 5 frames per second and computes mean and
standard deviation for each blendshape in 20-second windows with a 5-second step.
A window is labelled only when it lies entirely inside one self-report interval.
For the documented five-interval task format, the first and last intervals are
treated as 90 seconds and the three middle intervals as 120 seconds.

Outputs are local and ignored by Git:

- data/face_stress_windows.csv — derived features and labels
- ml_model/face_stress_model.joblib — trained research model
- ml_model/face_stress_evaluation.json — evaluation metrics and limitations

Optional arguments:

    python ml_model/train_face_stress_model.py --dataset-root data/empathicschool
    python ml_model/train_face_stress_model.py --sample-fps 3

## Validation policy

Evaluation uses Leave-One-Subject-Out (LOSO) validation, so windows from a held-out
participant are not mixed into that participant's training fold. Accuracy,
balanced accuracy, macro F1, the confusion matrix and per-class metrics are written
to the JSON report. Class weighting is used instead of creating synthetic examples.

Do not report training-set accuracy as model performance. Do not deploy a
user-facing stress score simply because the script completes. First review the
held-out results, class support and failed-task list; then independently test on
participants who were not part of EmpathicSchool.

The published EmpathicSchool study reports much weaker results for emotion-only
features than for multimodal features. Its published metrics are not the performance
of this new MediaPipe blendshape model and must not be copied into NeuroSense's report.

## Product integration status

This is an offline training pipeline only. It does not activate a face-only stress
score in the web app. The browser currently estimates pulse from local rPPG; the
blendshape model has a different feature contract and must be integrated only after
a genuine training run and a matching browser inference implementation. Raw frames
must remain local to the browser.
