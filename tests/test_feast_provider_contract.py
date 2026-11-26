"""Target-hour safety checks using local Parquet and deterministic fake Feast responses.

These tests run without Feast; actual backend integration remains a separate check.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.features import feast_provider as provider

ISSUE = pd.Timestamp("2023-06-15T16:00:00Z")
FEATURES = ["solar_elevation_target_deg", "tmp_c"]


def online_response(
    times: list[pd.Timestamp | None], values: dict[str, list[Any]] | None = None,
) -> SimpleNamespace:
    """Mimic Feast's per-feature event timestamps, statuses and values."""
    results = [SimpleNamespace(statuses=[1], event_timestamps=[])]
    for stamp in times:
        timestamps = [] if stamp is None else [SimpleNamespace(
            seconds=stamp.value // 1_000_000_000, nanos=stamp.value % 1_000_000_000,
        )]
        results.append(SimpleNamespace(statuses=[3 if stamp is None else 1],
                                       event_timestamps=timestamps))
    payload = values or {"site_id": ["2107"], FEATURES[0]: [35.0], FEATURES[1]: [None]}
    return SimpleNamespace(
        proto=SimpleNamespace(
            metadata=SimpleNamespace(feature_names=SimpleNamespace(val=["site_id", *FEATURES])),
            results=results,
        ),
        to_dict=lambda: payload,
    )


def query_online(monkeypatch, times: list[pd.Timestamp | None], **kwargs) -> dict[str, Any]:
    """Query the provider with a deterministic backend response."""
    response = online_response(times, kwargs.pop("values", None))
    store = SimpleNamespace(get_online_features=lambda **k: response)
    monkeypatch.setattr(provider, "get_feature_store", lambda *a: store)
    return provider.get_online_features("2107", kwargs.pop("issue_time_utc", ISSUE),
                                        feature_names=FEATURES, **kwargs)


def test_online_exact_hour_preserves_missing_weather(monkeypatch) -> None:
    """An exact record with missing weather remains a usable model input."""
    row = query_online(monkeypatch, [ISSUE, ISSUE])
    assert row["event_timestamp"] == ISSUE
    assert row["staleness_hours"] == 0
    assert row["tmp_c"] is None


def test_equivalent_timezone_offset_is_normalized(monkeypatch) -> None:
    """An explicit non-UTC offset can still identify the same UTC issue hour."""
    row = query_online(monkeypatch, [ISSUE, ISSUE], issue_time_utc="2023-06-15T09:00:00-07:00")
    assert row["requested_issue_time_utc"] == ISSUE


@pytest.mark.parametrize("age", [pd.Timedelta(hours=1), pd.Timedelta(nanoseconds=1)])
def test_online_old_target_is_rejected_inside_freshness_limit(monkeypatch, age) -> None:
    """Freshness tolerance cannot authorize a different target hour."""
    with pytest.raises(provider.FeastProviderError, match="exact|target|requested"):
        query_online(monkeypatch, [ISSUE - age, ISSUE - age])


@pytest.mark.parametrize("second", [ISSUE - pd.Timedelta(hours=1), None])
def test_every_feature_timestamp_is_checked(monkeypatch, second) -> None:
    """The first feature cannot hide another feature's old or absent record."""
    with pytest.raises(provider.FeastProviderError):
        query_online(monkeypatch, [ISSUE, second])


def test_online_future_is_rejected(monkeypatch) -> None:
    """Never answer an earlier issue time with later features."""
    with pytest.raises(provider.FutureFeatureError):
        query_online(monkeypatch, [ISSUE + pd.Timedelta(hours=1)] * 2)


def test_online_stale_error_remains_compatible(monkeypatch) -> None:
    """Existing callers can still distinguish stale snapshots."""
    with pytest.raises(provider.StaleFeatureError):
        query_online(monkeypatch, [ISSUE - pd.Timedelta(hours=5)] * 2)


@pytest.mark.parametrize("limit", [-1.0, float("nan"), float("inf")])
def test_invalid_staleness_policy_is_rejected(monkeypatch, limit: float) -> None:
    """Invalid policies must not disable temporal checks."""
    with pytest.raises(ValueError):
        query_online(monkeypatch, [ISSUE, ISSUE], max_staleness_hours=limit)


def test_naive_online_timestamp_is_rejected(monkeypatch) -> None:
    """Do not silently interpret local civil time as UTC."""
    with pytest.raises(ValueError):
        query_online(monkeypatch, [ISSUE, ISSUE], issue_time_utc="2023-06-15T16:00:00")


def test_online_wrong_site_is_rejected(monkeypatch) -> None:
    """A backend row must belong to the requested site."""
    with pytest.raises(provider.FeastProviderError):
        query_online(monkeypatch, [ISSUE, ISSUE], values={
            "site_id": ["other-site"], FEATURES[0]: [35.0], FEATURES[1]: [20.0],
        })


@pytest.fixture
def offline_store(tmp_path: Path, monkeypatch):
    """Register a tiny local source and a controllable as-of retrieval response."""
    source = pd.DataFrame({
        "site_id": ["2107", "2107"],
        "event_timestamp": [ISSUE - pd.Timedelta(hours=1), ISSUE],
        FEATURES[0]: [25.0, 35.0], FEATURES[1]: [20.0, float("nan")],
    })
    path = tmp_path / "features.parquet"
    source.to_parquet(path, index=False)
    store = MagicMock()
    store.repo_path = tmp_path
    store.get_feature_view.return_value = SimpleNamespace(batch_source=SimpleNamespace(
        path=str(path), timestamp_field="event_timestamp", field_mapping={},
    ))
    store.get_historical_features.return_value.to_df.return_value = source.iloc[[1]].copy()
    monkeypatch.setattr(provider, "get_feature_store", lambda *a: store)
    return store, source, path


def query_offline(time=ISSUE, site_id: str = "2107") -> pd.DataFrame:
    """Request one exact prediction hour."""
    return provider.get_historical_features(pd.DataFrame({
        "site_id": [site_id], "event_timestamp": [time],
    }), feature_names=FEATURES)


def test_offline_exact_hour_includes_source_timestamp(offline_store) -> None:
    """Return audited exact rows while preserving weather NaN."""
    result = query_offline()
    assert result["source_feature_time_utc"].iloc[0] == ISSUE
    assert pd.isna(result["tmp_c"].iloc[0])


@pytest.mark.parametrize("time", [ISSUE + pd.Timedelta(hours=1), ISSUE + pd.Timedelta(hours=10)])
def test_offline_missing_hour_or_night_rejects_prior_row(offline_store, time) -> None:
    """No prior daylight row may substitute for missing hours or night requests."""
    store, _, _ = offline_store
    with pytest.raises(provider.FeastProviderError, match="exact|target|Missing"):
        query_offline(time)
    store.get_historical_features.assert_not_called()


def test_offline_wrong_site_is_rejected(offline_store) -> None:
    """Exact time in another site does not satisfy a request."""
    with pytest.raises(provider.FeastProviderError):
        query_offline(site_id="other-site")


def test_offline_asof_values_cannot_masquerade_as_exact_row(offline_store) -> None:
    """Feast echoes the request timestamp; verify values against the exact source too."""
    store, source, _ = offline_store
    wrong = source.iloc[[0]].copy()
    wrong["event_timestamp"] = ISSUE
    store.get_historical_features.return_value.to_df.return_value = wrong
    with pytest.raises(provider.FeastProviderError, match="match|value|source"):
        query_offline()


def test_offline_duplicate_source_keys_are_rejected(offline_store) -> None:
    """Ambiguous versions cannot silently choose an arbitrary record."""
    _, source, path = offline_store
    pd.concat([source, source.iloc[[1]]]).to_parquet(path, index=False)
    with pytest.raises(provider.FeastProviderError, match="duplicate|unique|ambiguous"):
        query_offline()


def test_offline_missing_response_row_is_rejected(offline_store) -> None:
    """A shortened response cannot silently drop a requested prediction."""
    store, source, _ = offline_store
    store.get_historical_features.return_value.to_df.return_value = source.iloc[:0].copy()
    with pytest.raises(provider.FeastProviderError):
        query_offline()


def test_offline_naive_timestamp_is_rejected(offline_store) -> None:
    """Offline requests use explicit UTC too."""
    with pytest.raises(ValueError):
        query_offline("2023-06-15T16:00:00")


def test_reordered_offline_rows_keep_exact_source_identity(offline_store) -> None:
    """Value checks align by site/time, not a backend's accidental output order."""
    store, source, _ = offline_store
    store.get_historical_features.return_value.to_df.return_value = source.iloc[::-1].copy()
    result = provider.get_historical_features(source[["site_id", "event_timestamp"]],
                                              feature_names=FEATURES)
    assert result[FEATURES[0]].tolist() == [35.0, 25.0]
    assert (result["source_feature_time_utc"] == result["event_timestamp"]).all()


def test_replay_cannot_bypass_missing_hour_check(offline_store) -> None:
    """The replay helper uses the same exact-key rules as direct offline calls."""
    with pytest.raises(provider.FeastProviderError):
        provider.replay_historical_features("2107", [ISSUE + pd.Timedelta(hours=1)],
                                            feature_names=FEATURES)
