"""Feast feature provider for SolarCast online serving and historical replay (Task M1.5 / Step 4).

Provides point-in-time historical feature retrieval and safe online feature serving
with strict staleness and future-timestamp rejection.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.preprocessing import get_feature_names

try:
    import feast
    from feast import FeatureStore

    FEAST_AVAILABLE = True
except ImportError:
    feast = None
    FeatureStore = None
    FEAST_AVAILABLE = False

DEFAULT_REPO_PATH = Path("feature_store/feature_repo")
DEFAULT_FEATURE_VIEW = "solarcast_f1_set_b"


class FeastProviderError(Exception):
    """Base exception for Feast provider errors."""


class StaleFeatureError(FeastProviderError):
    """Raised when online features are older than the allowed staleness threshold."""


class FutureFeatureError(FeastProviderError):
    """Raised when online features have an event timestamp later than the query time."""


class EntityNotFoundError(FeastProviderError):
    """Raised when an entity key is not present in the feature store."""


def _ensure_feast_available() -> None:
    """Ensure Feast is installed in the current environment."""
    if not FEAST_AVAILABLE or FeatureStore is None:
        raise ImportError(
            "Feast is not installed in the active environment. "
            "Please use .venv-feast or install requirements-feast.txt."
        )


def get_feature_store(repo_path: Path | str = DEFAULT_REPO_PATH) -> FeatureStore:
    """Instantiate and return a Feast FeatureStore.

    Args:
        repo_path: Path to the feature repository directory containing feature_store.yaml.

    Returns:
        Instantiated FeatureStore object.
    """
    _ensure_feast_available()
    path = Path(repo_path).resolve()
    if not (path / "feature_store.yaml").exists():
        raise FileNotFoundError(f"feature_store.yaml not found in {path}")
    return FeatureStore(repo_path=str(path))


def get_historical_features(
    entity_df: pd.DataFrame,
    feature_names: list[str] | None = None,
    repo_path: Path | str = DEFAULT_REPO_PATH,
    feature_view: str = DEFAULT_FEATURE_VIEW,
) -> pd.DataFrame:
    """Retrieve point-in-time correct historical features using Feast offline store.

    Performs an as-of join joining the most recent feature record preceding or at
    entity_df['event_timestamp'] within the view TTL.

    Args:
        entity_df: DataFrame with at least ['site_id', 'event_timestamp'].
        feature_names: Optional list of feature names (defaults to all Set B features).
        repo_path: Path to the feature repository directory.
        feature_view: Name of the feature view to query.

    Returns:
        DataFrame containing entity columns plus retrieved feature columns.
    """
    store = get_feature_store(repo_path)
    if feature_names is None:
        feature_names = get_feature_names("B")
    feature_refs = [f"{feature_view}:{feat}" for feat in feature_names]

    return store.get_historical_features(entity_df=entity_df, features=feature_refs).to_df()


def get_online_features(
    site_id: str,
    issue_time_utc: datetime.datetime | str,
    max_staleness_hours: float = 2.0,
    repo_path: Path | str = DEFAULT_REPO_PATH,
    feature_names: list[str] | None = None,
    feature_view: str = DEFAULT_FEATURE_VIEW,
) -> dict[str, Any]:
    """Retrieve online features for a site, strictly enforcing staleness and future rejection.

    Rejection semantics:
    - If no features exist in the online store for the entity, raises EntityNotFoundError.
    - If the stored event_timestamp is strictly greater than issue_time_utc,
      raises FutureFeatureError (never answering an old issue time with newer values).
    - If the stored event_timestamp is older than (issue_time_utc - max_staleness_hours),
      raises StaleFeatureError (rejecting stale data).

    Args:
        site_id: Site identifier string (e.g. "2107").
        issue_time_utc: Target prediction issue timestamp (UTC).
        max_staleness_hours: Maximum allowed feature age in hours relative to issue_time_utc.
        repo_path: Path to feature repository directory.
        feature_names: Optional list of feature names (defaults to all Set B features).
        feature_view: Name of the feature view to query.

    Returns:
        Dictionary containing site_id, event_timestamp, requested_issue_time_utc,
        staleness_hours, and feature key-value pairs.
    """
    store = get_feature_store(repo_path)
    if feature_names is None:
        feature_names = get_feature_names("B")

    # Normalize issue_time_utc
    if isinstance(issue_time_utc, str):
        target_dt = datetime.datetime.fromisoformat(issue_time_utc)
    else:
        target_dt = issue_time_utc
    if target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=datetime.UTC)

    feature_refs = [f"{feature_view}:{feat}" for feat in feature_names]
    response = store.get_online_features(
        features=feature_refs,
        entity_rows=[{"site_id": str(site_id)}],
    )

    # Inspect event timestamp from response protobuf
    record_event_ts: datetime.datetime | None = None
    for i, name in enumerate(response.proto.metadata.feature_names.val):
        if name == "site_id":
            continue
        res = response.proto.results[i]
        # Status 3 indicates FieldStatus.NOT_FOUND in Feast protobuf
        if list(res.statuses) == [3] or len(res.event_timestamps) == 0:
            raise EntityNotFoundError(
                f"Entity site_id='{site_id}' not found in online store."
            )
        ts_proto = res.event_timestamps[0]
        if ts_proto.seconds == 0 and ts_proto.nanos == 0:
            raise EntityNotFoundError(
                f"Entity site_id='{site_id}' has uninitialized timestamp in online store."
            )
        record_event_ts = datetime.datetime.fromtimestamp(
            ts_proto.seconds + ts_proto.nanos * 1e-9, tz=datetime.UTC
        )
        break

    if record_event_ts is None:
        raise EntityNotFoundError(f"No valid features found for site_id='{site_id}'")

    # Future rejection: record timestamp > requested issue time
    if record_event_ts > target_dt:
        raise FutureFeatureError(
            f"Online store feature event_timestamp ({record_event_ts.isoformat()}) is later than "
            f"requested issue_time_utc ({target_dt.isoformat()}). "
            f"Cannot serve future features for past issue times."
        )

    # Staleness rejection: record timestamp is too old
    age = target_dt - record_event_ts
    age_hours = age.total_seconds() / 3600.0
    if age_hours > max_staleness_hours:
        raise StaleFeatureError(
            f"Online store features are stale for issue_time_utc={target_dt.isoformat()}. "
            f"Feature event_timestamp={record_event_ts.isoformat()} is {age_hours:.2f} hours old "
            f"(exceeds max_staleness_hours={max_staleness_hours}h)."
        )

    # Extract feature values
    resp_dict = response.to_dict()
    features: dict[str, Any] = {}
    for feat in feature_names:
        features[feat] = resp_dict[feat][0] if feat in resp_dict else None

    result: dict[str, Any] = {
        "site_id": str(site_id),
        "event_timestamp": record_event_ts,
        "requested_issue_time_utc": target_dt,
        "staleness_hours": age_hours,
        "features": features,
    }
    # Also populate top-level feature keys for convenience
    result.update(features)
    return result


def replay_historical_features(
    site_id: str,
    issue_times: list[datetime.datetime | str] | pd.Series | pd.DatetimeIndex,
    feature_names: list[str] | None = None,
    repo_path: Path | str = DEFAULT_REPO_PATH,
    feature_view: str = DEFAULT_FEATURE_VIEW,
) -> pd.DataFrame:
    """Replay historical features for issue times using point-in-time offline retrieval.

    Args:
        site_id: Site identifier string.
        issue_times: Iterable of timestamps representing prediction issue times (UTC).
        feature_names: Optional list of feature names (defaults to all Set B features).
        repo_path: Path to the feature repository directory.
        feature_view: Name of the feature view to query.

    Returns:
        DataFrame containing site_id, event_timestamp, and all retrieved features.
    """
    times = pd.to_datetime(list(issue_times), utc=True)
    entity_df = pd.DataFrame(
        {
            "site_id": [str(site_id)] * len(times),
            "event_timestamp": times,
        }
    )
    return get_historical_features(
        entity_df=entity_df,
        feature_names=feature_names,
        repo_path=repo_path,
        feature_view=feature_view,
    )
