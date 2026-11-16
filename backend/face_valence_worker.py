"""Isolated AFFEC model worker.

This file runs under /opt/affec-venv, pinned to the scikit-learn version used
when the AFFEC joblib artifact was serialized. It must be executed as a script
so the unpickled custom transformers are available under __main__.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression


EXPECTED_SKLEARN_VERSION = "1.6.1"
EXPECTED_FEATURES = 50


class ClipQ(BaseEstimator, TransformerMixin):
    """Training-compatible 1st-99th percentile feature clipping."""

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=np.float64)
        self.lo_ = np.nanpercentile(values, 1, axis=0)
        self.hi_ = np.nanpercentile(values, 99, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lo_, self.hi_)


class KB(SelectKBest):
    """Training-compatible SelectKBest with a capped feature count."""

    def __init__(self, score_func=f_regression, k_max=60):
        super().__init__(score_func=score_func, k=k_max)
        self.k_max = k_max

    def fit(self, X, y):
        self.k = min(self.k_max, X.shape[1])
        return super().fit(X, y)


def main() -> int:
    if sklearn.__version__ != EXPECTED_SKLEARN_VERSION:
        raise RuntimeError(
            f"AFFEC worker requires scikit-learn {EXPECTED_SKLEARN_VERSION}; "
            f"found {sklearn.__version__}."
        )
    if len(sys.argv) != 2:
        raise ValueError("Usage: face_valence_worker.py MODEL_PATH")

    model_path = Path(sys.argv[1]).resolve()
    if not model_path.is_file() or model_path.stat().st_size < 1_000_000:
        raise FileNotFoundError(f"Invalid AFFEC model artifact: {model_path}")

    payload = json.load(sys.stdin)
    feature_names = payload.get("feature_names")
    feature_values = payload.get("features")
    if not isinstance(feature_names, list) or len(feature_names) != EXPECTED_FEATURES:
        raise ValueError("The AFFEC request must contain exactly 50 feature names.")
    if not isinstance(feature_values, dict) or set(feature_values) != set(feature_names):
        raise ValueError("The AFFEC feature values do not match the saved schema.")

    frame = pd.DataFrame(
        [[feature_values[name] for name in feature_names]],
        columns=feature_names,
        dtype=np.float32,
    )
    model = joblib.load(model_path)
    model_feature_count = getattr(model, "n_features_in_", None)
    if model_feature_count is not None and int(model_feature_count) != EXPECTED_FEATURES:
        raise RuntimeError(
            f"AFFEC model expects {model_feature_count} features; "
            f"received {EXPECTED_FEATURES}."
        )

    prediction = float(np.asarray(model.predict(frame)).reshape(-1)[0])
    if not np.isfinite(prediction):
        raise RuntimeError("AFFEC returned a non-finite felt-valence prediction.")

    sys.stdout.write(json.dumps({
        "felt_valence": prediction,
        "model_runtime": f"scikit-learn {sklearn.__version__}",
    }))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        sys.stderr.write(f"AFFEC worker error: {type(error).__name__}: {error}\n")
        raise SystemExit(1)
