"""Model packaging and artifact bundling for SolarCast (Task M1.5 / Step 5).

Provides `ModelBundle`, `save_bundle`, and `load_bundle` to persist trained estimators
with preprocessing, ordered feature schemas, data hashes, and strict physical constraints.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODELS_DIR = Path("models")
DEFAULT_BUNDLE_PATH = DEFAULT_MODELS_DIR / "solarcast_f1_bundle.joblib"
DEFAULT_CAPACITY_LIMIT_KWH = 893.0


@dataclass
class ModelBundle:
    """Exportable and self-contained model bundle."""

    estimator: Any
    feature_names: list[str]
    training_window: dict[str, str]
    data_hashes: dict[str, str]
    mlflow_run_id: str | None
    capacity_limit_kwh: float = DEFAULT_CAPACITY_LIMIT_KWH
    model_name: str = "solarcast-f1"
    feature_schema_version: str = "1.0.0"
    created_at_utc: str = ""

    def __post_init__(self) -> None:
        if not self.created_at_utc:
            self.created_at_utc = datetime.datetime.now(datetime.UTC).isoformat()

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Generate physically constrained predictions from input features.

        Args:
            features_df: DataFrame containing required feature columns.

        Returns:
            1D numpy array of predicted energy in kWh, clipped to [0, capacity]
            with nighttime hours forced strictly to 0.0.
        """
        # Validate that all required features are present
        missing = [f for f in self.feature_names if f not in features_df.columns]
        if missing:
            raise KeyError(f"Input features missing required columns: {missing}")

        X = features_df[self.feature_names].values

        # Determine daylight condition (solar elevation > 0)
        if "solar_elevation_target_deg" in features_df.columns:
            is_daylight = (features_df["solar_elevation_target_deg"] > 0.0).values
        elif "is_daylight" in features_df.columns:
            is_daylight = features_df["is_daylight"].values.astype(bool)
        else:
            is_daylight = np.ones(len(features_df), dtype=bool)

        raw_preds = self.estimator.predict(X)
        preds = np.asarray(raw_preds, dtype=float).ravel()

        # Enforce physical constraints: 0.0 at night, clip to [0, capacity]
        preds = np.where(is_daylight, preds, 0.0)
        preds = np.clip(preds, 0.0, self.capacity_limit_kwh)
        return preds


def save_bundle(bundle: ModelBundle, output_path: Path | str = DEFAULT_BUNDLE_PATH) -> Path:
    """Save ModelBundle to disk using joblib."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(bundle_path: Path | str = DEFAULT_BUNDLE_PATH) -> ModelBundle:
    """Load ModelBundle from disk using joblib."""
    path = Path(bundle_path)
    if not path.exists():
        raise FileNotFoundError(f"Bundle not found at {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"Loaded object is not a ModelBundle (got {type(bundle)})")
    return bundle
