"""Verify calibration excludes unavailable labels and survives serialization."""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.features import preprocessing
from src.models.bundle import ModelBundle, load_bundle, save_bundle


def test_scale_uses_only_available_training_labels(monkeypatch, tmp_path):
    times = pd.date_range("2022-01-01", periods=4, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "interval_start_utc": times, "target_kwh": [100, 200, 900, 9000],
        "quality_flag": "VALID",
    })
    monkeypatch.setattr(preprocessing.pvlib.location, "Location", lambda *a, **k: SimpleNamespace(
        get_solarposition=lambda t: pd.DataFrame({"elevation": 30}, index=t),
        get_clearsky=lambda t: pd.DataFrame({"ghi": 100}, index=t),
    ))
    config = preprocessing.fit_clearsky_scale(frame, times[3], preprocessing.load_feature_config())
    assert config["p95_clearsky_ratio"] == pytest.approx(1.95)
    assert config["clearsky_scale_provenance"]["sample_count"] == 2
    bundle = ModelBundle(None, [], {}, {}, None, feature_config=config)
    path = save_bundle(bundle, tmp_path / "bundle.joblib")
    monkeypatch.setattr(preprocessing, "build_features", lambda *a, **k: k["config"])
    assert load_bundle(path).build_features(times) == config
