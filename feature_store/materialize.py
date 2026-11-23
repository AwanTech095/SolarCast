"""Feast materialization pipeline for SolarCast (Task Step 3 & Step 5).

Loads feature data from the offline Parquet store into the SQLite online store
for low-latency online serving and replay simulations. Optionally logs materialization
metadata to MLflow.
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import time
from pathlib import Path
from typing import Any

from feast import FeatureStore

DEFAULT_REPO_PATH = Path("feature_store/feature_repo")


def _get_git_commit() -> str:
    """Retrieve current Git commit SHA."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        return commit
    except Exception:
        return "unknown"


def materialize_features(
    start_date: datetime.datetime | str,
    end_date: datetime.datetime | str,
    repo_path: Path | str = DEFAULT_REPO_PATH,
    feature_views: list[str] | None = None,
    log_mlflow: bool = False,
) -> dict[str, Any]:
    """Materialize feature views from offline store into the SQLite online store.

    Args:
        start_date: Start of materialization window (UTC).
        end_date: End of materialization window (UTC).
        repo_path: Path to feature repository directory containing feature_store.yaml.
        feature_views: Optional list of feature view names to materialize (defaults to all).
        log_mlflow: Whether to log the materialization run to MLflow.

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

    t0 = time.time()
    store = FeatureStore(repo_path=str(path))
    store.materialize(
        start_date=dt_start,
        end_date=dt_end,
        feature_views=feature_views,
    )
    duration_s = time.time() - t0

    views = feature_views or ["solarcast_f1_set_b"]
    summary: dict[str, Any] = {
        "status": "SUCCESS",
        "repo_path": str(path),
        "start_date_utc": dt_start.isoformat(),
        "end_date_utc": dt_end.isoformat(),
        "feature_views": views,
        "duration_seconds": duration_s,
        "materialized_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }

    if log_mlflow:
        import os

        import feast
        import mlflow

        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        git_commit = _get_git_commit()
        mlflow.set_experiment("solarcast")
        with mlflow.start_run(run_name="feature_store_materialization") as run:
            mlflow.set_tags(
                {
                    "feast_repo_git_commit": git_commit,
                    "feature_view_name": ",".join(views),
                    "entity": "site",
                    "entity_key": "site_id",
                    "offline_format": "parquet",
                    "online_store": "sqlite",
                    "feast_version": getattr(feast, "__version__", "unknown"),
                    "owner": "member1",
                }
            )
            mlflow.log_params(
                {
                    "repo_path": str(path),
                    "feature_views": ",".join(views),
                    "start_date_utc": dt_start.isoformat(),
                    "end_date_utc": dt_end.isoformat(),
                    "ttl_days": 7,
                }
            )
            mlflow.log_metrics(
                {
                    "materialized_views_count": len(views),
                    "duration_seconds": duration_s,
                }
            )
            summary["mlflow_run_id"] = run.info.run_id
            print(f"Logged MLflow materialization run: {run.info.run_id}")

        # Ensure backwards compatibility with MLflow 2.x tracking client
        try:
            meta_path = (
                Path("mlruns")
                / run.info.experiment_id
                / run.info.run_id
                / "meta.yaml"
            )
            if meta_path.exists():
                content = meta_path.read_text(encoding="utf-8")
                if "run_uuid:" not in content:
                    meta_path.write_text(
                        content + f"run_uuid: {run.info.run_id}\n", encoding="utf-8"
                    )
        except Exception:
            pass

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
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="Log materialization metadata to MLflow",
    )
    args = parser.parse_args()

    materialize_features(
        start_date=args.start,
        end_date=args.end,
        repo_path=args.repo_path,
        feature_views=args.views,
        log_mlflow=args.log_mlflow,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
