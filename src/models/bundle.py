"""Model packaging and artifact bundling for SolarCast (Task M1.5 / Step 4 & 5).

Provides `ModelBundle`, `save_bundle`, and `load_bundle` to persist trained estimators
with preprocessing, ordered feature schemas, data hashes, and strict physical constraints.
Supports Feature Set A and Feature Set B (with weather features).
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
    feature_schema_version: str = "1.1.0"
    feature_set: str = "A"
    needs_weather: bool = False
    created_at_utc: str = ""
    feature_config: dict[str, Any] | None = None

    def build_features(self, issue_times_utc: Any, **sources: Any) -> pd.DataFrame:
        """Build inference inputs using this release's frozen feature configuration."""
        from src.features.preprocessing import build_features

        return build_features(
            issue_times_utc, config=self.feature_config, feature_set=self.feature_set, **sources,
        )

    def __post_init__(self) -> None:
        if not self.created_at_utc:
            self.created_at_utc = datetime.datetime.now(datetime.UTC).isoformat()
        if not self.needs_weather and self.feature_set == "B":
            self.needs_weather = True

    def predict(
        self,
        features_df: pd.DataFrame,
        weather_df: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Generate physically constrained predictions from input features.

        Args:
            features_df: DataFrame containing required feature columns.
            weather_df: Optional DataFrame containing weather features to merge.

        Returns:
            1D numpy array of predicted energy in kWh, clipped to [0, capacity]
            with nighttime hours forced strictly to 0.0.
        """
        df = features_df.copy()
        if weather_df is not None:
            for col in self.feature_names:
                if col not in df.columns and col in weather_df.columns:
                    df[col] = weather_df[col]

        # Validate that all required features are present
        missing = [f for f in self.feature_names if f not in df.columns]
        if missing:
            raise KeyError(f"Input features missing required columns: {missing}")

        X = df[self.feature_names].values

        # Determine daylight condition (solar elevation > 0)
        if "solar_elevation_target_deg" in df.columns:
            is_daylight = (df["solar_elevation_target_deg"] > 0.0).values
        elif "is_daylight" in df.columns:
            is_daylight = df["is_daylight"].values.astype(bool)
        else:
            is_daylight = np.ones(len(df), dtype=bool)

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
