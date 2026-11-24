"""Contract tests verifying model bundle predictions match sample requests/responses (Task Step 3).

Verifies that `models/solarcast_f1_bundle.joblib` deterministically reproduces
the sample predictions recorded in `docs/samples/predict_responses.json` within 1e-6 tolerance,
and validates inference on `tests/fixtures/features_sample.csv`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.bundle import ModelBundle, load_bundle

DEFAULT_BUNDLE_PATH = Path("models/solarcast_f1_bundle.joblib")
REQUESTS_SAMPLE_PATH = Path("docs/samples/predict_requests.json")
RESPONSES_SAMPLE_PATH = Path("docs/samples/predict_responses.json")
FEATURES_FIXTURE_PATH = Path("tests/fixtures/features_sample.csv")


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    """Load model bundle artifact."""
    assert DEFAULT_BUNDLE_PATH.exists(), f"Bundle not found at {DEFAULT_BUNDLE_PATH}"
    b = load_bundle(DEFAULT_BUNDLE_PATH)
    assert isinstance(b, ModelBundle)
    return b


def test_bundle_metadata(bundle: ModelBundle) -> None:
    """Verify bundle metadata conforms to Schema 1.1.0."""
    assert bundle.model_name == "solarcast-f1"
    assert bundle.feature_schema_version == "1.1.0"
    assert bundle.feature_set == "B"
    assert bundle.needs_weather is True
    assert len(bundle.feature_names) == 29
    assert bundle.capacity_limit_kwh == 893.0


def test_contract_samples_reproducibility(bundle: ModelBundle) -> None:
    """Verify bundle reproduces sample predictions in predict_responses.json within 1e-6."""
    with open(REQUESTS_SAMPLE_PATH, encoding="utf-8") as f:
        requests = json.load(f)
    with open(RESPONSES_SAMPLE_PATH, encoding="utf-8") as f:
        responses = {r["example_id"]: r for r in json.load(f)}

    for req in requests:
        ex_id = req["example_id"]
        resp = responses[ex_id]

        if resp.get("status_code") == 503:
            # Stale rejection example
            assert resp["error"] == "StaleFeatureError"
            assert resp["feature_staleness_hours"] > 2.0
            continue

        # Valid inference request
        features = req["features"]
        df_in = pd.DataFrame([features])
        pred = float(bundle.predict(df_in)[0])
        expected = float(resp["predicted_energy_kwh"])

        diff = abs(pred - expected)
        assert np.isclose(
            pred, expected, atol=1e-6
        ), f"Discrepancy in {ex_id}: got {pred}, expected {expected} (diff: {diff:.2e})"


def test_contract_sample_specific_behaviors(bundle: ModelBundle) -> None:
    """Verify specific behavioral contracts on the sample cases."""
    with open(REQUESTS_SAMPLE_PATH, encoding="utf-8") as f:
        requests = {r["example_id"]: r for r in json.load(f)}
    with open(RESPONSES_SAMPLE_PATH, encoding="utf-8") as f:
        responses = {r["example_id"]: r for r in json.load(f)}

    # 1. Nighttime forcing guarantee (Example 3)
    req_night = requests["example_3_nighttime"]
    df_night = pd.DataFrame([req_night["features"]])
    pred_night = float(bundle.predict(df_night)[0])
    assert pred_night == 0.0, f"Night prediction must be identically 0.0, got {pred_night}"
    assert responses["example_3_nighttime"]["predicted_energy_kwh"] == 0.0

    # 2. Missing weather support (Example 4)
    req_miss = requests["example_4_weather_missing"]
    assert req_miss["features"]["weather_missing"] == 1
    assert req_miss["features"]["tmp_c"] is None
    assert req_miss["features"]["tcdc_pct"] is None
    df_miss = pd.DataFrame([req_miss["features"]])
    pred_miss = float(bundle.predict(df_miss)[0])
    assert np.isfinite(pred_miss)
    assert 0.0 <= pred_miss <= 893.0
    assert "weather_missing" in responses["example_4_weather_missing"]["warnings"]

    # 3. Solar noon peak bounds (Example 2)
    req_noon = requests["example_2_clear_solar_noon"]
    df_noon = pd.DataFrame([req_noon["features"]])
    pred_noon = float(bundle.predict(df_noon)[0])
    assert pred_noon > 400.0, f"Expected strong generation at solar noon, got {pred_noon}"
    assert pred_noon <= 893.0


def test_features_sample_fixture_evaluation(bundle: ModelBundle) -> None:
    """Verify features_sample.csv has ~48 rows, < 200 KB, and yields valid predictions."""
    assert FEATURES_FIXTURE_PATH.exists()
    file_size_bytes = FEATURES_FIXTURE_PATH.stat().st_size
    assert file_size_bytes < 200 * 1024, f"Fixture too large: {file_size_bytes} bytes"

    df = pd.read_csv(FEATURES_FIXTURE_PATH)
    assert len(df) == 48, f"Expected 48 rows, got {len(df)}"

    # Verify attribution column exists
    assert "attribution" in df.columns
    assert "10.25984/1846021" in df["attribution"].iloc[0]

    # Verify predictions on all 48 rows
    preds = bundle.predict(df)
    assert len(preds) == 48
    assert np.all(np.isfinite(preds)), "Non-finite predictions detected"
    assert np.all(preds >= 0.0), "Negative predictions detected"
    assert np.all(preds <= 893.0), "Predictions exceed plant DC capacity"
