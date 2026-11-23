"""Demonstration script for Feast feature store integration (Step 4 demos).

Executes:
(a) get_historical_features for 20 sample issue times vs build_features (skew tolerance 1e-9).
(b) Materialize up to a replay time (2023-06-15 12:00 UTC) and query online store.
(c) get_online_features rejection of stale rows and future timestamps.
(d) Point-in-time replay helper over historical issue times.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from feature_store.materialize import materialize_features  # noqa: E402
from src.features.feast_provider import (  # noqa: E402
    FutureFeatureError,
    StaleFeatureError,
    get_historical_features,
    get_online_features,
    replay_historical_features,
)
from src.features.preprocessing import build_features, get_feature_names  # noqa: E402


def run_demos() -> None:
    print("=" * 70)
    print("Feast Feature Store Step 4 Demonstrations")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Demo (a): get_historical_features vs build_features (skew test)
    # -------------------------------------------------------------------------
    print("\n[Demo a] 20 Sample Issue Times Training-Serving Skew Test...")
    feature_table = pd.read_parquet("data/processed/feast/features_set_b.parquet")
    sample_rows = feature_table.sample(20, random_state=42).sort_values("event_timestamp")
    sample_times = pd.to_datetime(sample_rows["event_timestamp"]).reset_index(drop=True)

    print(f"Sampled 20 issue times between {sample_times.min()} and {sample_times.max()}")

    # Compute features via shared build_features
    built_features = build_features(sample_times, feature_set="B").reset_index(drop=True)

    # Retrieve features via Feast get_historical_features
    entity_df = pd.DataFrame({
        "site_id": ["2107"] * len(sample_times),
        "event_timestamp": sample_times,
    })
    hist_features = (
        get_historical_features(entity_df)
        .sort_values("event_timestamp")
        .reset_index(drop=True)
    )

    feature_names = get_feature_names("B")
    max_abs_diff = 0.0
    for feat in feature_names:
        b_vals = built_features[feat].values
        h_vals = hist_features[feat].values
        nan_match = np.isnan(b_vals) == np.isnan(h_vals)
        assert nan_match.all(), f"NaN mask mismatch in feature '{feat}'"
        valid = ~np.isnan(b_vals)
        if valid.any():
            diff = float(np.max(np.abs(b_vals[valid] - h_vals[valid])))
            if diff > max_abs_diff:
                max_abs_diff = diff

    print(f"-> Verified 20 samples across all {len(feature_names)} features.")
    print(f"-> Maximum absolute discrepancy: {max_abs_diff:.2e} (tolerance: 1e-9).")
    assert max_abs_diff < 1e-9, f"Skew {max_abs_diff} exceeds 1e-9!"
    print("-> RESULT: PASSED (Zero training-serving skew confirmed).")

    # -------------------------------------------------------------------------
    # Demo (b): Materialize up to replay time & query online store
    # -------------------------------------------------------------------------
    print("\n[Demo b] Materialize up to Replay Time & Online Store Query...")
    # Materialize up to 2023-06-15 15:00 UTC (daylight hours active)
    replay_time_str = "2023-06-15T15:00:00"
    replay_time = datetime.datetime.fromisoformat(replay_time_str).replace(tzinfo=datetime.UTC)
    start_time = replay_time - datetime.timedelta(days=2)

    mat_summary = materialize_features(
        start_date=start_time,
        end_date=replay_time,
        repo_path="feature_store/feature_repo",
    )
    print(
        f"-> Materialized interval: {mat_summary['start_date_utc']} to "
        f"{mat_summary['end_date_utc']}"
    )

    # Query online features at the exact replay issue time
    online_res = get_online_features(
        site_id="2107",
        issue_time_utc=replay_time,
        max_staleness_hours=2.0,
    )
    print(f"-> Query successful at issue_time_utc={replay_time.isoformat()}:")
    print(f"   Stored record event_timestamp: {online_res['event_timestamp'].isoformat()}")
    print(f"   Staleness: {online_res['staleness_hours']:.2f} hours")
    print(f"   clearsky_ghi_target_wm2: {online_res['clearsky_ghi_target_wm2']:.2f} W/m^2")
    print(f"   power_kw_lag_0m: {online_res['power_kw_lag_0m']:.2f} kW")
    print(f"   tcdc_pct: {online_res['tcdc_pct']}%")
    print(f"   tmp_c: {online_res['tmp_c']} degC")
    print("-> RESULT: PASSED.")

    # -------------------------------------------------------------------------
    # Demo (c): get_online_features safe rejection (future and stale)
    # -------------------------------------------------------------------------
    print("\n[Demo c] Testing Safe Online Rejections (Future and Stale)...")

    # 1. Test future rejection: query with an earlier time than the stored record
    earlier_query_time = replay_time - datetime.timedelta(hours=4)
    future_rejected = False
    try:
        get_online_features(
            site_id="2107",
            issue_time_utc=earlier_query_time,
            max_staleness_hours=24.0,
        )
    except FutureFeatureError as e:
        future_rejected = True
        print(f"-> Future rejection verified: {e}")
    assert future_rejected, "Expected FutureFeatureError was not raised!"

    # 2. Test staleness rejection: query with a later time beyond max_staleness_hours
    later_query_time = replay_time + datetime.timedelta(hours=5)
    stale_rejected = False
    try:
        get_online_features(
            site_id="2107",
            issue_time_utc=later_query_time,
            max_staleness_hours=2.0,
        )
    except StaleFeatureError as e:
        stale_rejected = True
        print(f"-> Staleness rejection verified: {e}")
    assert stale_rejected, "Expected StaleFeatureError was not raised!"
    print("-> RESULT: PASSED (Future and stale rejections strictly enforced).")

    # -------------------------------------------------------------------------
    # Demo (d): Replay helper using offline retrieval
    # -------------------------------------------------------------------------
    print("\n[Demo d] Historical Replay Helper Demonstration...")
    test_issue_times = [
        "2023-05-01T18:00:00+00:00",
        "2023-06-01T18:00:00+00:00",
        "2023-07-01T18:00:00+00:00",
        "2023-08-01T18:00:00+00:00",
    ]
    replay_df = replay_historical_features(
        site_id="2107",
        issue_times=test_issue_times,
    )
    print(f"-> Replay retrieved {len(replay_df)} rows:")
    for _, row in replay_df.iterrows():
        ts = row["event_timestamp"]
        ghi = row["clearsky_ghi_target_wm2"]
        tmp = row["tmp_c"]
        cld = row["tcdc_pct"]
        print(f"   [{ts}] GHI={ghi:.1f} W/m2, tmp={tmp:.1f}C, cloud={cld:.0f}%")
    print("-> RESULT: PASSED.")

    print("\n" + "=" * 70)
    print("All Step 4 Demos Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_demos()
