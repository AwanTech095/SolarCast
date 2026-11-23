"""Feast materialization pipeline for SolarCast (Task Step 3).

Loads feature data from the offline Parquet store into the SQLite online store
for low-latency online serving and replay simulations.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
from typing import Any

from feast import FeatureStore

DEFAULT_REPO_PATH = Path("feature_store/feature_repo")


def materialize_features(
    start_date: datetime.datetime | str,
    end_date: datetime.datetime | str,
    repo_path: Path | str = DEFAULT_REPO_PATH,
    feature_views: list[str] | None = None,
) -> dict[str, Any]:
    """Materialize feature views from offline store into the SQLite online store.

    Args:
        start_date: Start of materialization window (UTC).
        end_date: End of materialization window (UTC).
        repo_path: Path to feature repository directory containing feature_store.yaml.
        feature_views: Optional list of feature view names to materialize (defaults to all).

    Returns:
        Dictionary summarizing materialization metadata.
    """
    path = Path(repo_path).resolve()
    if not (path / "feature_store.yaml").exists():
        raise FileNotFoundError(f"feature_store.yaml not found in {path}")

    # Parse datetimes to timezone-aware UTC
    if isinstance(start_date, str):
        dt_start = datetime.datetime.fromisoformat(start_date)
    else:
        dt_start = start_date
    if dt_start.tzinfo is None:
        dt_start = dt_start.replace(tzinfo=datetime.UTC)

    if isinstance(end_date, str):
        dt_end = datetime.datetime.fromisoformat(end_date)
    else:
        dt_end = end_date
    if dt_end.tzinfo is None:
        dt_end = dt_end.replace(tzinfo=datetime.UTC)

    print(
        f"Initializing Feast FeatureStore from {path}...\n"
        f"Materializing interval [{dt_start.isoformat()} to {dt_end.isoformat()}]..."
    )

    store = FeatureStore(repo_path=str(path))
    store.materialize(
        start_date=dt_start,
        end_date=dt_end,
        feature_views=feature_views,
    )

    summary = {
        "status": "SUCCESS",
        "repo_path": str(path),
        "start_date_utc": dt_start.isoformat(),
        "end_date_utc": dt_end.isoformat(),
        "feature_views": feature_views or ["solarcast_f1_set_b"],
        "materialized_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    print(f"Materialization complete: {summary}")
    return summary


def main() -> int:
    """CLI entrypoint for Feast materialization."""
    parser = argparse.ArgumentParser(description="SolarCast Feast Materialization Runner")
    parser.add_argument(
        "--start",
        type=str,
        default="2023-06-01T00:00:00+00:00",
        help="Start UTC timestamp in ISO format (e.g. 2023-06-01T00:00:00+00:00)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2023-06-15T12:00:00+00:00",
        help="End UTC timestamp in ISO format (e.g. 2023-06-15T12:00:00+00:00)",
    )
    parser.add_argument(
        "--repo-path",
        type=str,
        default=str(DEFAULT_REPO_PATH),
        help="Path to feature repository",
    )
    parser.add_argument(
        "--views",
        nargs="*",
        default=None,
        help="Optional list of feature view names",
    )
    args = parser.parse_args()

    materialize_features(
        start_date=args.start,
        end_date=args.end,
        repo_path=args.repo_path,
        feature_views=args.views,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
