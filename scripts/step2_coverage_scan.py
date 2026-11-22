"""HRRR Zarr Full Coverage Scan (Step 2).

Scans all hourly runs from 2018-07-12 00:00 to 2023-11-09 23:00 UTC using cheap HEAD
requests (16 threads, 30s timeout, 2 retries, progress every 500 requests, resumable).
Checks whether chunk 1.4 (key 0.4.1) exists for 2m TMP and for entire_atmosphere TCDC.
Saves resumable results to data/processed/hrrr_coverage_scan.parquet.
Reports missing runs by month and by model version (v3 vs v4).
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Output paths
OUT_DIR = Path("data/processed")
SCAN_FILE = OUT_DIR / "hrrr_coverage_scan.parquet"

START_UTC = "2018-07-12 00:00:00"
END_UTC = "2023-11-09 23:00:00"
V4_START = pd.Timestamp("2020-12-02 00:00:00", tz="UTC")

thread_local = threading.local()


def get_session() -> requests.Session:
    """Get thread-local requests Session with retry adapter."""
    if not hasattr(thread_local, "session"):
        s = requests.Session()
        retries = Retry(
            total=2,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=4, pool_maxsize=4)
        s.mount("https://", adapter)
        thread_local.session = s
    return thread_local.session


def head_with_retry(url: str, timeout: int = 30) -> bool:
    """Perform HEAD request with up to 2 retries on network errors."""
    for attempt in range(3):
        try:
            s = get_session()
            r = s.head(url, timeout=timeout)
            if r.status_code == 200:
                return True
            if r.status_code == 404:
                return False
        except Exception:
            # Recreate session on socket/TLS reset
            s = requests.Session()
            adapter = HTTPAdapter(
                max_retries=Retry(total=2, backoff_factor=0.3),
                pool_connections=4,
                pool_maxsize=4,
            )
            s.mount("https://", adapter)
            thread_local.session = s
            if attempt == 2:
                return False
            time.sleep(0.3)
    return False


def check_run(dt: pd.Timestamp) -> dict[str, Any]:
    """Check existence of chunk 1.4 for 2m TMP and TCDC for a specific run."""
    date_str = dt.strftime("%Y%m%d")
    hh_str = f"{dt.hour:02d}"
    base_url = (
        f"https://hrrrzarr.s3.us-west-1.amazonaws.com/sfc/{date_str}/"
        f"{date_str}_{hh_str}z_fcst.zarr"
    )
    url_tmp = f"{base_url}/2m_above_ground/TMP/2m_above_ground/TMP/0.4.1"
    url_tcdc = f"{base_url}/entire_atmosphere/TCDC/entire_atmosphere/TCDC/0.4.1"

    tmp_exists = head_with_retry(url_tmp, timeout=30)
    tcdc_exists = False
    if tmp_exists:
        tcdc_exists = head_with_retry(url_tcdc, timeout=30)

    return {
        "run_time_utc": dt,
        "tmp_exists": tmp_exists,
        "tcdc_exists": tcdc_exists,
    }


def main() -> None:
    """Run full coverage scan."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate full hourly schedule
    all_runs = pd.date_range(START_UTC, END_UTC, freq="h", tz="UTC")
    total_runs = len(all_runs)
    print(f"Total hourly runs to scan: {total_runs:,} ({START_UTC} to {END_UTC})")

    # 2. Check for resumable partial results
    done_dict: dict[pd.Timestamp, dict[str, Any]] = {}
    if SCAN_FILE.exists():
        try:
            existing_df = pd.read_parquet(SCAN_FILE)
            for _, r in existing_df.iterrows():
                done_dict[pd.to_datetime(r["run_time_utc"], utc=True)] = {
                    "run_time_utc": pd.to_datetime(r["run_time_utc"], utc=True),
                    "tmp_exists": bool(r["tmp_exists"]),
                    "tcdc_exists": bool(r["tcdc_exists"]),
                }
            print(f"Resumed from {SCAN_FILE}: {len(done_dict):,} runs already scanned.")
        except Exception as e:
            print(f"Could not read existing scan file: {e}")

    remaining_runs = [dt for dt in all_runs if dt not in done_dict]
    print(f"Remaining runs to scan: {len(remaining_runs):,}")

    # 3. Scan remaining runs with 16 threads
    results: list[dict[str, Any]] = list(done_dict.values())
    batch_new: list[dict[str, Any]] = []
    req_counter = 0
    t0 = time.time()

    if remaining_runs:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_dt = {executor.submit(check_run, dt): dt for dt in remaining_runs}
            for future in concurrent.futures.as_completed(future_to_dt):
                res = future.result()
                results.append(res)
                batch_new.append(res)
                req_counter += 1

                if req_counter % 500 == 0 or len(results) == total_runs:
                    elapsed = time.time() - t0
                    rate = req_counter / elapsed if elapsed > 0 else 0.0
                    pct = (len(results) / total_runs) * 100.0
                    print(
                        f"Progress: {len(results):,}/{total_runs:,} runs "
                        f"({pct:.1f}%) scanned. "
                        f"Batch: {req_counter:,} runs at {rate:.1f} runs/s."
                    )
                    # Resumable checkpoint save
                    df_checkpoint = pd.DataFrame(results)
                    df_checkpoint.sort_values("run_time_utc", inplace=True)
                    df_checkpoint.to_parquet(SCAN_FILE, index=False)

    df_all = pd.DataFrame(results)
    df_all["run_time_utc"] = pd.to_datetime(df_all["run_time_utc"], utc=True)
    df_all.sort_values("run_time_utc", inplace=True)
    df_all.to_parquet(SCAN_FILE, index=False)
    print(f"Saved complete scan to {SCAN_FILE} ({len(df_all):,} records).")

    # 4. Report statistics
    df_all["month"] = df_all["run_time_utc"].dt.strftime("%Y-%m")
    df_all["model_version"] = df_all["run_time_utc"].apply(
        lambda t: "HRRR v3" if t < V4_START else "HRRR v4"
    )

    print("\n=== Missing Runs by Model Version ===")
    ver_summary = (
        df_all.groupby("model_version")
        .agg(
            total_runs=("run_time_utc", "count"),
            tmp_missing=("tmp_exists", lambda s: int((~s).sum())),
            tcdc_missing=("tcdc_exists", lambda s: int((~s).sum())),
        )
        .reset_index()
    )
    ver_summary["tmp_avail_pct"] = (
        (ver_summary["total_runs"] - ver_summary["tmp_missing"])
        / ver_summary["total_runs"]
        * 100
    )
    ver_summary["tcdc_avail_pct"] = (
        (ver_summary["total_runs"] - ver_summary["tcdc_missing"])
        / ver_summary["total_runs"]
        * 100
    )
    print(ver_summary.to_string(index=False))

    print("\n=== Missing Runs by Month (Months with missing runs) ===")
    month_summary = (
        df_all.groupby("month")
        .agg(
            total_runs=("run_time_utc", "count"),
            tmp_missing=("tmp_exists", lambda s: int((~s).sum())),
            tcdc_missing=("tcdc_exists", lambda s: int((~s).sum())),
        )
        .reset_index()
    )
    missing_months = month_summary[
        (month_summary["tmp_missing"] > 0) | (month_summary["tcdc_missing"] > 0)
    ]
    if len(missing_months) > 0:
        print(missing_months.to_string(index=False))
    else:
        print("Zero missing runs across all months!")


if __name__ == "__main__":
    main()
