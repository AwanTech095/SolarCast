"""Feast FeatureView definitions for SolarCast."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

from feast import FeatureView, FileSource

# Support both Feast CLI discovery and Python package imports
REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

try:
    from entities import site_entity

    from features import get_feature_fields
except ImportError:
    from .entities import site_entity
    from .features import get_feature_fields

# Absolute path to features_set_b.parquet ensuring Windows cross-directory execution
SOURCE_PARQUET_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "processed"
    / "feast"
    / "features_set_b.parquet"
).as_posix()

solar_source = FileSource(
    path=SOURCE_PARQUET_PATH,
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# FeatureView for Feature Set B
# TTL Semantics in Feast 0.68.0:
# - Historical retrieval (get_historical_features): TTL defines the maximum lookback window
#   [entity_timestamp - ttl, entity_timestamp]. If no feature event falls within this window,
#   nulls are returned.
# - Online retrieval (get_online_features): Feast SQLite online store returns the latest
#   materialized record for the entity key without evaluating TTL or staleness at query time.
#   The provider requires every requested feature to match the exact issue time.
#   Offline requests also verify exact source keys/values; TTL never permits reuse
#   of an older target-specific row for an absent hour.
solarcast_f1_set_b_view = FeatureView(
    name="solarcast_f1_set_b",
    entities=[site_entity],
    ttl=timedelta(days=7),
    schema=get_feature_fields(),
    online=True,
    source=solar_source,
    description=(
        "Next-hour solar forecasting features "
        "(Set B: production telemetry, solar geometry, NOAA HRRR weather)"
    ),
    owner="member1",
    tags={
        "feature_set": "B",
        "schema_version": "1.1.0",
        "target": "hourly_ac_energy_kwh",
        "freshness_expectation": "hourly",
    },
)
