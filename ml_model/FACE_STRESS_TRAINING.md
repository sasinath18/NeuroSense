# NeuroSense — Face-only Stress Model

The face-only research path is separate from the WESAD wearable model. Its feature
contract must contain only facial video/landmark-derived measurements; HR, BVP, EDA,
skin temperature, pupil data, accelerometer values and wearable-model outputs are
excluded. The existing EmpathicSchool trainer uses MediaPipe blendshape features.
The EmoWork dataset described below provides a different facial-landmark feature
contract and requires a separate adapter before it can be trained or deployed.

## Dataset and access

The selected research dataset is EmpathicSchool:

- Current dataset record (v3): https://doi.org/10.5281/zenodo.15556502
- Earlier record: https://doi.org/10.5281/zenodo.7372497
- Research paper: https://doi.org/10.1038/s41597-025-05812-0
- Stress labels are self-assessments at roughly two-minute intervals.
- Video recordings are released only for participants who consented to video sharing.
- The Zenodo record currently marks the files as restricted. A user must sign in,
  request access and personally accept the Research Use License. NeuroSense does
  not download, redistribute or commit the restricted dataset.

The dataset agreement is for research use and includes privacy/security obligations.
Do not upload the downloaded archive, participant videos, labels or derived feature
CSV to GitHub or a public service.

## Required files for the Face-only stress experiment

The EmpathicSchool paper lists these release files: `HR.csv`, `EDA.csv`,
`TEMP.csv`, `Acc.csv`, `V.mp4`, `Stress_level.csv`, `Landmarks.csv`,
`BVP.csv` and `IBI.csv`. The release includes synchronized facial video and
physiological streams; the published file list does not identify a pupil-diameter
CSV. If a local archive contains a pupil file, verify its source and metadata
rather than assuming it belongs to EmpathicSchool.

For this **Face-only** model, use only the following matching pair for each
participant/task:

- `V.mp4`: the participant's video, available only where video-sharing consent
  was granted.
- `Stress_level.csv`: the corresponding self-reported stress labels. Each label
  covers a two-minute interval; documented values are 0 (no stress), 1 (low
  stress), 2 (high stress), and unknown.

`Landmarks.csv` is an optional video-derived alternative feature source, but the
current EmpathicSchool trainer extracts MediaPipe blendshapes directly from
`V.mp4`. It does not need or read `EDA.csv`, `BVP.csv`, `IBI.csv`,
`HR.csv`, `TEMP.csv`, or `Acc.csv`; those streams belong to a separate
multimodal experiment. Pupil measurements are not an input to this trainer.

**Do not confuse the earlier AFFEC/DevEmo+ files with EmpathicSchool stress
labels.** The previously reviewed AFFEC pack's pupil/GSR and video data are
labelled for felt valence/arousal, not the self-reported EmpathicSchool stress
classes. DevEmo+ labels are facial-expression categories. Neither can replace
`Stress_level.csv` for stress training.


## Dataset audit — do not substitute emotion labels for stress labels

The NeuroSense Library was audited on 2026-11-21. The available AFFEC pack contains
face, pupil and GSR recordings/features, but its behavioural labels are felt/perceived
emotion valence and arousal (`f_emotion_v`, `f_emotion_a`, `p_emotion_v`,
`p_emotion_a`). Its `trial_type` field is the emotion category of the presented
stimulus (sad, fear, happy, neutral, disgust or angry), not a participant stress
rating. These columns must not be remapped to stress classes.

The available DevEmo+ archive contains facial-expression categories such as neutral,
confusion, happiness, surprise, disgust and anger. It does not contain self-reported
stress labels. It is suitable only for an explicitly labelled facial-expression
experiment, not this stress-training pipeline.

The separate AFFEC felt-valence model is not a face-only stress detector. Do not use
its output as a stress target or silently relabel arousal/valence as stress.

## Additional candidate: EmoWork (facial-feature CSV + continuous stress labels)

EmoWork is a more recent candidate for a second, face-only feature pipeline:

- Latest Zenodo version: https://zenodo.org/records/17513978
- Dataset DOI: https://doi.org/10.5281/zenodo.17513978
- Data paper: https://doi.org/10.1038/s41597-025-06531-2
- The data include 31 call-agent participants and approximately 395 minutes of
  recorded video. Raw videos are not distributed; timestamped facial features
  extracted from MediaPipe/OpenCV are provided instead.
- The video-derived features include head rotation/translation, total head
  movement, blink events, gaze direction, lip-corner movement and an error flag.
- The LABELS folder contains timestamped, self-reported stress values on a
  1–20 scale, collected at approximately 100 ms intervals.
- This makes a facial-feature-only stress experiment possible without loading
  the dataset's audio, wearable, EEG, EDA, BVP, temperature or other physiological
  modalities.

**Access is controlled.** Download and read the dataset Data Usage Agreement,
request access on Zenodo, and submit the required data-use form. Do not try to
bypass the access gate or place restricted data, labels, derived participant
features or dataset archives in GitHub/Render. See the paper's Usage Notes and
the official code repository: https://github.com/Kaist-ICLab/EmoWork

The EmoWork context is simulated customer-service interactions and emotional
workload, not students' everyday study stress. Its features are derived facial
measurements, not raw video. The published paper's multimodal stress metrics are
not face-only performance and must not be copied as NeuroSense results.

### Offline EmoWork training pipeline

NeuroSense now includes `ml_model/train_emowork_face_stress_model.py`. It supports
the official EmoWork v3 CSV mirror. Its exact folders are `CSV/LABEL_CSV`
and `CSV/VIDEO_csv` (the latter name uses a lowercase `csv` suffix):

    data/emowork/
      CSV/
        LABEL_CSV/
          1/
            c1/
              stress.csv
        VIDEO_csv/
          1/
            c1.csv

The loader also accepts legacy `LABEL` / `LABELS` and `VIDEO` fixture layouts
for unit tests and small, manually exported samples. It does not require or read
the raw-video files; the official release supplies timestamped face-derived
features in the CSV mirror.

The loader strictly selects the documented timestamped facial-feature columns
(head rotation/translation, total translation, blink, gaze and left/right lip
corner movement). It filters unsuccessful landmark rows using `error_type`,
then aligns each face frame to its nearest self-reported stress observation
within 250 ms. It uses only video-derived facial features and the self-reported
stress target; it does not open the dataset's SENSORS, AUDIO, pupil, EDA/GSR,
BVP, temperature, HR or EEG data.

The trainer creates 5-second windows with a 2.5-second step, aggregates facial
features using mean and standard deviation, and predicts the continuous 1–20
stress score with a Random Forest regressor. Evaluation is Leave-One-Participant-Out
(LOSO), not a random window split. The output is an offline research artifact,
not enabled in FaceScan.

After receiving access and accepting the dataset DUA, extract the authorized
dataset locally and run:

    python ml_model/train_emowork_face_stress_model.py --dataset-root data/emowork

Outputs are local and ignored by Git:

- `data/emowork_face_stress_windows.csv`
- `ml_model/emowork_face_stress_rf.joblib`
- `ml_model/emowork_face_stress_evaluation.json`

The face-only experiment still requires independent-user validation and a
student/study-context dataset before a NeuroSense student-facing stress score can
be justified. The EmoWork face-feature schema must also be matched by a live
browser-side facial extractor before product integration. Do not enable it merely
because offline training completes.

The EmpathicSchool stress-training path below remains the intended source, subject to
the dataset's access restrictions and Research Use License. Training may proceed only
when each included video can be paired with the provider's documented self-reported
stress labels and the participant has consented to video sharing. If the authorized
video/label files are unavailable, stop and report the missing data instead of
creating a proxy stress label.

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

## Pre-training dataset audit

Before downloading the MediaPipe model or extracting any frames, validate the
authorised local dataset:

    python ml_model/train_face_stress_model.py --dataset-root data/empathicschool --audit-only

For an aggregate-only report that omits participant IDs, task IDs, file names and
per-task details, use:

    python ml_model/train_face_stress_model.py --dataset-root data/empathicschool --audit-summary-only

The summary still contains aggregate dataset counts and class distribution. Keep it
within the permissions of the Research Use License; do not publish or share it unless
the agreement permits that use. This mode does not download MediaPipe or decode video.

The audit checks each consented MP4 against its task-level `Stress_level.csv`,
validates the video duration and label-interval mapping (including shortened T3
tasks), and reports participant count and the 0/1/2 class distribution. It
also reports any videos that have no matching task-level stress-label file,
including when every video is missing its labels. It does not decode frames,
download the model asset, save participant data or upload any files. A passing
audit is only a structural/readiness check; training still requires enough usable
labelled windows and successful LOSO evaluation.

The normal training command repeats this metadata-only preflight automatically
before downloading the MediaPipe asset or decoding participant videos. If the
participant/class readiness checks fail, it stops and prints the warnings.

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
For standard five-label tasks, the first and last intervals are
treated as 90 seconds and the three middle intervals as 120 seconds. The paper
notes that some T3 presentation recordings ended early and therefore have fewer
two-minute self-report intervals. For those shortened T3 tasks, each available
label covers its consecutive 120-second segment; the final segment is clipped
to the video end when needed, and any remaining footage without a label is left
unlabelled. Other tasks with fewer than five labels are rejected instead of
redistributing labels evenly across the video.

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
The trainer also fails closed before saving a model artifact if LOSO skips any
participant fold or leaves labelled windows unevaluated. This can happen when a
held-out participant is the only source of one class, leaving a single-class
training fold. Resolve the participant-level class-support problem and rerun the
complete evaluation; do not treat partial-fold metrics as validated model results.

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
