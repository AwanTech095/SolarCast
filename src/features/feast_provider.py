"""Feast feature provider for SolarCast online serving and historical replay (Task M1.5 / Step 4).

Provides point-in-time historical feature retrieval and safe online feature serving
with exact target-hour identity, staleness and future-timestamp rejection.
"""

from __future__ import annotations

import datetime
import math
from collections import Counter
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


class TargetHourMismatchError(FeastProviderError):
    """Target-specific features do not belong to the exact requested issue time."""


def _utc(value: Any) -> pd.Timestamp:
    """Require an explicit timezone and preserve nanoseconds when normalizing UTC."""
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Feature timestamps must be timezone-aware; supply an explicit UTC offset")
    return stamp.tz_convert("UTC")


def _feature_names(names: list[str] | None) -> list[str]:
    """Validate requested model fields without allowing audit metadata collisions."""
    result = get_feature_names("B") if names is None else list(names)
    reserved = {
        "site_id", "event_timestamp", "source_feature_time_utc", "requested_issue_time_utc",
        "staleness_hours", "features",
    }
    if not result or len(set(result)) != len(result) or reserved.intersection(result):
        raise ValueError("Request nonempty, unique feature names without reserved metadata fields")
    return result


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

    Requires an exact site/time row in the registered local Parquet source before
    asking Feast for features. Validates returned values against that source row;
    the as-of join's echoed request timestamp alone is not source-time evidence.
    Missing hours (including absent night rows) fail instead of reusing daylight rows.

    Args:
        entity_df: DataFrame with at least ['site_id', 'event_timestamp'].
        feature_names: Optional list of feature names (defaults to all Set B features).
        repo_path: Path to the feature repository directory.
        feature_view: Name of the feature view to query.

    Returns:
        Entity/feature columns and source_feature_time_utc from the exact source row.
        Duplicate source keys and unsupported sources fail closed. This adapter is
        for this project's local Parquet FileSource, not a general backend adapter.
    """
    store = get_feature_store(repo_path)
    feature_names = _feature_names(feature_names)
    keys = ["site_id", "event_timestamp"]
    if not set(keys).issubset(entity_df.columns) or entity_df.empty:
        raise ValueError("Request at least one site_id and explicit event_timestamp")
    if "source_feature_time_utc" in entity_df.columns or set(feature_names) & set(entity_df):
        raise ValueError("Entity request columns collide with feature/audit output columns")
    requests = entity_df.copy()
    if requests["site_id"].isna().any():
        raise ValueError("site_id cannot be missing")
    requests["site_id"] = requests["site_id"].astype(str)
    requests["event_timestamp"] = pd.DatetimeIndex([_utc(t) for t in requests["event_timestamp"]])

    source = store.get_feature_view(feature_view).batch_source
    source_path = getattr(source, "path", None)
    if (
        not source_path or "://" in str(source_path)
        or source.timestamp_field != "event_timestamp"
        or getattr(source, "field_mapping", None)
    ):
        raise FeastProviderError("Exact replay requires an unmapped local Parquet FileSource")
    path = Path(source_path)
    if not path.is_absolute():
        path = Path(store.repo_path) / path
    table = pd.read_parquet(path, columns=[*keys, *feature_names])
    if table["site_id"].isna().any():
        raise FeastProviderError("Source contains missing site keys")
    table["site_id"] = table["site_id"].astype(str)
    table["event_timestamp"] = pd.DatetimeIndex([_utc(t) for t in table["event_timestamp"]])
    if table.duplicated(keys).any():
        raise FeastProviderError(
            "Source site/time keys must be unique; duplicate rows are ambiguous"
        )
    exact = requests[keys].merge(table, on=keys, how="left", indicator=True, validate="many_to_one")
    if (exact["_merge"] != "both").any():
        raise TargetHourMismatchError("Missing exact target-hour source row; no as-of substitution")
    feature_refs = [f"{feature_view}:{feat}" for feat in feature_names]
    result = store.get_historical_features(entity_df=requests, features=feature_refs).to_df().copy()
    if not {*keys, *feature_names}.issubset(result.columns):
        raise FeastProviderError("Historical response is missing requested keys/features")
    result["event_timestamp"] = pd.DatetimeIndex([_utc(t) for t in result["event_timestamp"]])
    requested_keys = Counter(requests[keys].itertuples(index=False, name=None))
    returned_keys = Counter(result[keys].itertuples(index=False, name=None))
    if requested_keys != returned_keys:
        raise TargetHourMismatchError("Historical response does not match requested site/time rows")
    expected = result[keys].merge(table, on=keys, how="left", validate="many_to_one")
    for name in feature_names:
        actual = result[name].reset_index(drop=True)
        wanted = expected[name]
        equal = actual.eq(wanted) | (actual.isna() & wanted.isna())
        if not equal.fillna(False).all():
            raise TargetHourMismatchError(
                f"Returned feature {name} does not match exact source values"
            )
    result["source_feature_time_utc"] = expected["event_timestamp"].to_numpy()
    return result


def get_online_features(
    site_id: str,
    issue_time_utc: datetime.datetime | str,
    max_staleness_hours: float = 2.0,
    repo_path: Path | str = DEFAULT_REPO_PATH,
    feature_names: list[str] | None = None,
    feature_view: str = DEFAULT_FEATURE_VIEW,
) -> dict[str, Any]:
    """Retrieve features only for the exact site and issue hour requested.

    Rejection semantics:
    - If no features exist in the online store for the entity, raises EntityNotFoundError.
    - If the stored event_timestamp is strictly greater than issue_time_utc,
      raises FutureFeatureError (never answering an old issue time with newer values).
    - If the stored event_timestamp is older than (issue_time_utc - max_staleness_hours),
      raises StaleFeatureError (rejecting stale data).
    - Any remaining timestamp mismatch raises TargetHourMismatchError, even inside
      the freshness limit. Every requested feature is checked, not only the first.

    Args:
        site_id: Site identifier string (e.g. "2107").
        issue_time_utc: Target prediction issue timestamp (UTC).
        max_staleness_hours: Diagnostic age threshold; never authorizes target-hour reuse.
        repo_path: Path to feature repository directory.
        feature_names: Optional list of feature names (defaults to all Set B features).
        feature_view: Name of the feature view to query.

    Returns:
        Dictionary containing site_id, event_timestamp, requested_issue_time_utc,
        staleness_hours, and feature key-value pairs.
    """
    target_dt = _utc(issue_time_utc)
    if not math.isfinite(max_staleness_hours) or max_staleness_hours < 0:
        raise ValueError("max_staleness_hours must be finite and nonnegative")
    feature_names = _feature_names(feature_names)
    store = get_feature_store(repo_path)

    feature_refs = [f"{feature_view}:{feat}" for feat in feature_names]
    response = store.get_online_features(
        features=feature_refs,
        entity_rows=[{"site_id": str(site_id)}],
    )

    # Inspect event timestamp from response protobuf
    metadata_names = list(response.proto.metadata.feature_names.val)
    record_times = []
    for name in feature_names:
        if name not in metadata_names:
            raise EntityNotFoundError(f"Missing feature {name} for site_id='{site_id}'")
        i = metadata_names.index(name)
        res = response.proto.results[i]
        # Accept PRESENT or explicit NULL (e.g. missing weather), never missing records.
        if list(res.statuses) not in ([1], [2]) or len(res.event_timestamps) != 1:
            raise EntityNotFoundError(
                f"Entity site_id='{site_id}' not found in online store."
            )
        ts_proto = res.event_timestamps[0]
        if ts_proto.seconds == 0 and ts_proto.nanos == 0:
            raise EntityNotFoundError(
                f"Entity site_id='{site_id}' has uninitialized timestamp in online store."
            )
        record_event_ts = pd.Timestamp(ts_proto.seconds, unit="s", tz="UTC") + pd.Timedelta(
            ts_proto.nanos, unit="ns"
        )
        record_times.append(record_event_ts)
        if record_event_ts > target_dt:
            raise FutureFeatureError(
                f"Feature {name} event_timestamp ({record_event_ts.isoformat()}) is later than "
                f"requested issue_time_utc ({target_dt.isoformat()})."
            )
        age_hours = (target_dt - record_event_ts) / pd.Timedelta(hours=1)
        if age_hours > max_staleness_hours:
            raise StaleFeatureError(
                f"Feature {name} is {age_hours:.2f} hours old "
                f"(exceeds max_staleness_hours={max_staleness_hours}h)."
            )
        if record_event_ts != target_dt:
            raise TargetHourMismatchError(
                f"Feature {name} is for {record_event_ts.isoformat()}, not exact requested "
                f"target issue time {target_dt.isoformat()}; rebuild or materialize that hour."
            )

    # Extract feature values
    resp_dict = response.to_dict()
    if resp_dict.get("site_id") != [str(site_id)]:
        raise EntityNotFoundError("Online response site_id does not match the requested site")
    features: dict[str, Any] = {}
    for feat in feature_names:
        if feat not in resp_dict or len(resp_dict[feat]) != 1:
            raise EntityNotFoundError(f"Missing online value for requested feature {feat}")
        features[feat] = resp_dict[feat][0]

    result: dict[str, Any] = {
        "site_id": str(site_id),
        "event_timestamp": record_times[0],
        "source_feature_time_utc": record_times[0],
        "requested_issue_time_utc": target_dt,
        "staleness_hours": 0.0,
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
    times = pd.DatetimeIndex([_utc(t) for t in issue_times])
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
