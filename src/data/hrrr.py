"""NOAA HRRR forecast weather extraction pipeline for SolarCast (Mode F1).

Extracts 2m temperature and total/low/mid/high cloud cover from the MesoWest
HRRR Zarr archive on AWS S3 (s3://hrrrzarr/sfc/) for daylight target hours
between 2018-07-12 and 2023-11-09.

Operational availability and latency rules:
- Prediction issue time H = interval_start_utc.
- Target window: [H, H + 1h) UTC.
- Weather availability lag: 2 hours (weather_available_at_utc = init + 2h <= H).
- Candidate run selection: latest available run init <= H - 2h.
- If missing, cascading fallback to earlier runs up to max_run_fallback_hours (default 6h).
- Forecast leads read: F1 = H - init and F2 = F1 + 1 (trapezoidal numerical average
  representing the [H, H + 1h) interval).
- Units: TMP converted from Kelvin to Celsius (- 273.15); cloud cover in percentage (0-100%).
- Resumable monthly processing (newest months first) to data/processed/hrrr_parts/.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import s3fs
import yaml
import zarr

DEFAULT_DATA_CONFIG = Path("configs/data.yaml")
DEFAULT_HOURLY_PATH = Path("data/processed/hourly.parquet")
DEFAULT_SCAN_PATH = Path("data/processed/hrrr_coverage_scan.parquet")
DEFAULT_PARTS_DIR = Path("data/processed/hrrr_parts")
DEFAULT_MERGED_PATH = Path("data/processed/hrrr_forecast_features.parquet")
DEFAULT_LOG_PATH = Path("logs/hrrr_extract.log")

HRRR_VARIABLES = {
    "tmp": "2m_above_ground/TMP/2m_above_ground/TMP",
    "tcdc": "entire_atmosphere/TCDC/entire_atmosphere/TCDC",
    "lcdc": "low_cloud_layer/LCDC/low_cloud_layer/LCDC",
    "mcdc": "middle_cloud_layer/MCDC/middle_cloud_layer/MCDC",
    "hcdc": "high_cloud_layer/HCDC/high_cloud_layer/HCDC",
}

HRRR_FEATURE_COLUMNS = [
    "site_id",
    "issue_time_utc",
    "target_start_utc",
    "weather_run_init_utc",
    "weather_available_at_utc",
    "run_age_hours",
    "fallback_used",
    "tmp_c",
    "tcdc_pct",
    "lcdc_pct",
    "mcdc_pct",
    "hcdc_pct",
    "extraction_status",
]


@dataclass
class HRRRExtractorConfig:
    """Configuration parameters for HRRR extraction."""

    site_id: str = "2107"
    plant_ix: int = 198
    plant_iy: int = 642
    weather_lag_hours: int = 2
    max_fallback_hours: int = 6
    timeout_sec: int = 30
    max_retries: int = 2
    num_threads: int = 16
    log_interval_requests: int = 500
    start_date: str = "2018-07-12 00:00:00"
    end_date: str = "2023-11-09 23:00:00"
    s3_region: str = "us-west-1"
    variables: list[str] = field(default_factory=lambda: ["tmp", "tcdc"])

    @classmethod
    def from_yaml(cls, config_path: Path | str = DEFAULT_DATA_CONFIG) -> HRRRExtractorConfig:
        """Load configuration overrides from configs/data.yaml."""
        path = Path(config_path)
        if not path.exists():
            return cls()
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        hrrr_cfg = cfg.get("hrrr", {})
        grid_cfg = hrrr_cfg.get("grid_point", {})
        return cls(
            site_id=str(cfg.get("site_id", "2107")),
            plant_ix=int(grid_cfg.get("index_x", 198)),
            plant_iy=int(grid_cfg.get("index_y", 642)),
            weather_lag_hours=int(cfg.get("weather_availability_lag_hours", 2)),
            s3_region=str(hrrr_cfg.get("zarr_region", "us-west-1")),
        )


class ProgressTracker:
    """Thread-safe request and hour progress logger."""

    def __init__(self, log_path: Path | str = DEFAULT_LOG_PATH, interval: int = 500) -> None:
        self.log_path = Path(log_path)
        self.interval = interval
        self.lock = threading.Lock()
        self.requests_count = 0
        self.hours_count = 0
        self.success_count = 0
        self.fallback_count = 0
        self.failed_count = 0
        self.start_time = time.time()
        self.last_log_req = 0

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Append initial startup banner
        with open(self.log_path, "a", encoding="utf-8") as f:
            now_str = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
            f.write(f"\n--- HRRR Extraction Session Started: {now_str} ---\n")

    def increment(
        self,
        n_requests: int,
        status: str,
        month_label: str = "",
    ) -> None:
        """Increment progress counters and log periodically."""
        with self.lock:
            self.requests_count += n_requests
            self.hours_count += 1
            if status == "SUCCESS":
                self.success_count += 1
            elif status == "FALLBACK_SUCCESS":
                self.fallback_count += 1
            else:
                self.failed_count += 1

            if (self.requests_count - self.last_log_req) >= self.interval:
                self._write_log(month_label)
                self.last_log_req = self.requests_count

    def _write_log(self, month_label: str) -> None:
        elapsed = time.time() - self.start_time
        rate = self.requests_count / elapsed if elapsed > 0 else 0.0
        now_str = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        msg = (
            f"[{now_str}] Requests: {self.requests_count:,} | "
            f"Hours: {self.hours_count:,} | Month: {month_label} | "
            f"Rate: {rate:.1f} req/s | Success: {self.success_count:,} | "
            f"Fallback: {self.fallback_count:,} | Failed: {self.failed_count:,}\n"
        )
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg)
            f.flush()


# Thread-local storage for S3 filesystem and Zarr array handle cache
thread_local_storage = threading.local()


def get_thread_fs(region_name: str = "us-west-1") -> s3fs.S3FileSystem:
    """Retrieve or initialize thread-local S3FileSystem instance."""
    if not hasattr(thread_local_storage, "fs"):
        thread_local_storage.fs = s3fs.S3FileSystem(
            anon=True,
            client_kwargs={"region_name": region_name},
            default_cache_type="none",
        )
        thread_local_storage.zarr_cache = {}
    return thread_local_storage.fs


def get_cached_zarr_array(
    var_path: str,
    fs: s3fs.S3FileSystem,
) -> Any:
    """Open or retrieve cached Zarr array handle for a specific variable."""
    cache = getattr(thread_local_storage, "zarr_cache", None)
    if cache is not None and var_path in cache:
        return cache[var_path]

    store = s3fs.S3Map(root=var_path, s3=fs, check=False)
    arr = zarr.open(store, mode="r")
    if cache is not None:
        cache[var_path] = arr
    return arr


def load_coverage_scan_set(scan_path: Path | str = DEFAULT_SCAN_PATH) -> set[pd.Timestamp]:
    """Load valid run timestamps from multi-year coverage scan file if present."""
    path = Path(scan_path)
    if not path.exists():
        return set()
    df_scan = pd.read_parquet(path)
    valid_runs = df_scan[df_scan["tmp_exists"]]["run_time_utc"]
    return set(pd.to_datetime(valid_runs, utc=True))


def check_run_chunk_exists(
    run_dt: pd.Timestamp,
    fs: s3fs.S3FileSystem,
    valid_scan_set: set[pd.Timestamp] | None = None,
    timeout_sec: int = 30,
) -> bool:
    """Check whether spatial chunk 1.4 exists for a candidate forecast run."""
    if valid_scan_set is not None and len(valid_scan_set) > 0:
        return run_dt in valid_scan_set

    # Live check via S3 existence of 2m TMP chunk 0.4.1
    date_str = run_dt.strftime("%Y%m%d")
    hh_str = f"{run_dt.hour:02d}"
    chunk_path = (
        f"hrrrzarr/sfc/{date_str}/{date_str}_{hh_str}z_fcst.zarr/"
        f"2m_above_ground/TMP/2m_above_ground/TMP/0.4.1"
    )
    for _ in range(2):
        try:
            return bool(fs.exists(chunk_path))
        except Exception:
            time.sleep(0.3)
    return False


def find_eligible_run(
    issue_time: pd.Timestamp,
    fs: s3fs.S3FileSystem,
    valid_scan_set: set[pd.Timestamp] | None = None,
    lag_hours: int = 2,
    max_fallback_hours: int = 6,
) -> tuple[pd.Timestamp | None, int]:
    """Find the freshest available HRRR run init <= H - lag with fallback.

    Returns:
        (init_time_utc, fallback_offset_hours) or (None, -1) if no run found.
    """
    for fallback_offset in range(max_fallback_hours + 1):
        candidate_init = issue_time - pd.Timedelta(hours=lag_hours + fallback_offset)
        if check_run_chunk_exists(candidate_init, fs, valid_scan_set):
            return candidate_init, fallback_offset
    return None, -1


def extract_hour_weather(
    issue_time: pd.Timestamp,
    config: HRRRExtractorConfig,
    fs: s3fs.S3FileSystem,
    valid_scan_set: set[pd.Timestamp] | None = None,
) -> dict[str, Any]:
    """Extract forecast weather features for a single daylight prediction hour."""
    target_start = issue_time
    vars_str = ",".join(config.variables)
    empty_result: dict[str, Any] = {
        "site_id": config.site_id,
        "issue_time_utc": issue_time,
        "target_start_utc": target_start,
        "weather_run_init_utc": pd.NaT,
        "weather_available_at_utc": pd.NaT,
        "run_age_hours": np.nan,
        "fallback_used": False,
        "tmp_c": np.nan,
        "tcdc_pct": np.nan,
        "lcdc_pct": np.nan,
        "mcdc_pct": np.nan,
        "hcdc_pct": np.nan,
        "extraction_status": f"FAILED_NO_RUN:{vars_str}",
    }

    # 1. Candidate run selection with fallback
    init_time, fallback_offset = find_eligible_run(
        issue_time=issue_time,
        fs=fs,
        valid_scan_set=valid_scan_set,
        lag_hours=config.weather_lag_hours,
        max_fallback_hours=config.max_fallback_hours,
    )

    if init_time is None:
        return empty_result

    # 2. Compute forecast leads and array indices
    lead_start = int((issue_time - init_time) / pd.Timedelta(hours=1))  # e.g. 2
    lead_end = lead_start + 1  # e.g. 3
    idx1 = lead_start - 1  # 0-based index for F_lead_start
    idx2 = lead_end - 1  # 0-based index for F_lead_end

    date_str = init_time.strftime("%Y%m%d")
    hh_str = f"{init_time.hour:02d}"
    base_zarr = f"hrrrzarr/sfc/{date_str}/{date_str}_{hh_str}z_fcst.zarr"

    ix = config.plant_ix
    iy = config.plant_iy

    # 3. Read variables with retries
    extracted: dict[str, float] = {}
    read_success = True

    for var_key in config.variables:
        var_rel_path = HRRR_VARIABLES[var_key]
        val = np.nan
        var_full_path = f"{base_zarr}/{var_rel_path}"

        for attempt in range(config.max_retries + 1):
            try:
                arr = get_cached_zarr_array(var_full_path, fs)
                v1 = float(arr[idx1, iy, ix])
                v2 = float(arr[idx2, iy, ix])
                val = (v1 + v2) / 2.0
                break
            except Exception:
                if attempt == config.max_retries:
                    read_success = False
                time.sleep(0.3)

        if var_key == "tmp" and not np.isnan(val):
            val = float(np.round(val - 273.15, 2))  # Kelvin to Celsius
        elif not np.isnan(val):
            val = float(np.round(val, 2))

        extracted[var_key] = val

    req_failed = any(np.isnan(extracted.get(vk, np.nan)) for vk in config.variables)
    if not read_success or req_failed:
        empty_result["extraction_status"] = f"FAILED_READ:{vars_str}"
        empty_result["weather_run_init_utc"] = init_time
        return empty_result

    status = (
        f"FALLBACK_SUCCESS:{vars_str}" if fallback_offset > 0 else f"SUCCESS:{vars_str}"
    )
    avail_time = init_time + pd.Timedelta(hours=config.weather_lag_hours)

    return {
        "site_id": config.site_id,
        "issue_time_utc": issue_time,
        "target_start_utc": target_start,
        "weather_run_init_utc": init_time,
        "weather_available_at_utc": avail_time,
        "run_age_hours": float(lead_start),
        "fallback_used": bool(fallback_offset > 0),
        "tmp_c": extracted.get("tmp", np.nan),
        "tcdc_pct": extracted.get("tcdc", np.nan),
        "lcdc_pct": extracted.get("lcdc", np.nan),
        "mcdc_pct": extracted.get("mcdc", np.nan),
        "hcdc_pct": extracted.get("hcdc", np.nan),
        "extraction_status": status,
    }


def extract_month_worker(
    row: pd.Series,
    config: HRRRExtractorConfig,
    valid_scan_set: set[pd.Timestamp] | None,
    tracker: ProgressTracker | None,
    month_label: str,
) -> dict[str, Any]:
    """Worker task executed inside ThreadPoolExecutor for a single target hour."""
    fs = get_thread_fs(region_name=config.s3_region)
    issue_time = pd.to_datetime(row["interval_start_utc"], utc=True)
    res = extract_hour_weather(
        issue_time=issue_time,
        config=config,
        fs=fs,
        valid_scan_set=valid_scan_set,
    )
    if tracker is not None:
        tracker.increment(
            n_requests=len(config.variables),
            status=res["extraction_status"],
            month_label=month_label,
        )
    return res


def process_month_slice(
    month_df: pd.DataFrame,
    month_label: str,
    output_part_path: Path,
    config: HRRRExtractorConfig,
    valid_scan_set: set[pd.Timestamp] | None = None,
    tracker: ProgressTracker | None = None,
) -> pd.DataFrame:
    """Process all daylight hours for one month and write a small Parquet part."""
    output_part_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.num_threads) as executor:
        futures = [
            executor.submit(
                extract_month_worker,
                row=row,
                config=config,
                valid_scan_set=valid_scan_set,
                tracker=tracker,
                month_label=month_label,
            )
            for _, row in month_df.iterrows()
        ]
        for f in concurrent.futures.as_completed(futures):
            records.append(f.result())

    df_part = pd.DataFrame(records)
    # Sort chronologically within the month
    df_part = df_part.sort_values("target_start_utc").reset_index(drop=True)
    df_part = df_part[HRRR_FEATURE_COLUMNS]

    # Save small parquet part
    df_part.to_parquet(output_part_path, index=False)
    return df_part


def merge_parts(
    parts_dir: Path | str = DEFAULT_PARTS_DIR,
    output_merged_path: Path | str = DEFAULT_MERGED_PATH,
) -> pd.DataFrame:
    """Merge all monthly parquet parts into final consolidated dataset."""
    p_dir = Path(parts_dir)
    merged_path = Path(output_merged_path)
    part_files = sorted(p_dir.glob("hrrr_features_*.parquet"))

    if not part_files:
        raise FileNotFoundError(f"No monthly parts found in {p_dir}")

    dfs = [pd.read_parquet(f) for f in part_files]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df = merged_df.sort_values("target_start_utc").drop_duplicates(
        subset=["target_start_utc"]
    ).reset_index(drop=True)
    merged_df = merged_df[HRRR_FEATURE_COLUMNS]

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(merged_path, index=False)
    return merged_df


def run_hrrr_pipeline(
    config: HRRRExtractorConfig | None = None,
    hourly_path: Path | str = DEFAULT_HOURLY_PATH,
    scan_path: Path | str = DEFAULT_SCAN_PATH,
    parts_dir: Path | str = DEFAULT_PARTS_DIR,
    log_path: Path | str = DEFAULT_LOG_PATH,
    merge_when_done: bool = False,
    reverse_chronological: bool = True,
) -> dict[str, Any]:
    """Execute complete HRRR forecast extraction pipeline."""
    if config is None:
        config = HRRRExtractorConfig.from_yaml()

    p_dir = Path(parts_dir)
    p_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load hourly dataset and filter daylight evaluation window
    df_h = pd.read_parquet(hourly_path)
    df_h["interval_start_utc"] = pd.to_datetime(df_h["interval_start_utc"], utc=True)

    t_start = pd.to_datetime(config.start_date, utc=True)
    t_end = pd.to_datetime(config.end_date, utc=True)
    if len(str(config.end_date).strip()) <= 10:
        t_end = t_end + pd.Timedelta(hours=23, minutes=59, seconds=59)

    mask = (
        (df_h["interval_start_utc"] >= t_start)
        & (df_h["interval_start_utc"] <= t_end)
        & (df_h["is_daylight"])
    )
    target_df = df_h[mask].copy().reset_index(drop=True)

    # 2. Load fast coverage scan index
    scan_set = load_coverage_scan_set(scan_path)

    # 3. Partition into monthly groups
    target_df["month_key"] = target_df["interval_start_utc"].dt.strftime("%Y_%m")
    month_groups = sorted(target_df["month_key"].unique())
    if reverse_chronological:
        month_groups = sorted(month_groups, reverse=True)

    tracker = ProgressTracker(log_path=log_path, interval=config.log_interval_requests)

    processed_months = []
    skipped_months = []

    for m_key in month_groups:
        part_path = p_dir / f"hrrr_features_{m_key}.parquet"
        if part_path.exists() and part_path.stat().st_size > 0:
            skipped_months.append(m_key)
            continue

        m_df = target_df[target_df["month_key"] == m_key].copy()
        process_month_slice(
            month_df=m_df,
            month_label=m_key,
            output_part_path=part_path,
            config=config,
            valid_scan_set=scan_set,
            tracker=tracker,
        )
        processed_months.append(m_key)

    merged_df = None
    if merge_when_done:
        merged_df = merge_parts(parts_dir=p_dir)

    return {
        "total_daylight_hours": len(target_df),
        "total_months": len(month_groups),
        "processed_months": len(processed_months),
        "skipped_months": len(skipped_months),
        "merged": merged_df is not None,
    }
