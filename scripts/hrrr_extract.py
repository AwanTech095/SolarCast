"""CLI entrypoint for NOAA HRRR forecast weather extraction (scripts/hrrr_extract.py).

Extracts 2m temperature and cloud cover (total, low, mid, high) from the
MesoWest HRRR Zarr archive for all daylight hours between 2018-07-12 and 2023-11-09.
Processes in reverse chronological order (newest months first), storing monthly
parts in data/processed/hrrr_parts/, and merges them into
data/processed/hrrr_forecast_features.parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.hrrr import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_HOURLY_PATH,
    DEFAULT_LOG_PATH,
    DEFAULT_MERGED_PATH,
    DEFAULT_PARTS_DIR,
    DEFAULT_SCAN_PATH,
    HRRRExtractorConfig,
    merge_parts,
    run_hrrr_pipeline,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NOAA HRRR Forecast Weather Extraction for SolarCast (Mode F1)"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_DATA_CONFIG),
        help="Path to data configuration YAML",
    )
    parser.add_argument(
        "--hourly-file",
        default=str(DEFAULT_HOURLY_PATH),
        help="Path to hourly.parquet target dataset",
    )
    parser.add_argument(
        "--scan-file",
        default=str(DEFAULT_SCAN_PATH),
        help="Path to hrrr_coverage_scan.parquet fast lookup",
    )
    parser.add_argument(
        "--parts-dir",
        default=str(DEFAULT_PARTS_DIR),
        help="Directory to save/load monthly parquet parts",
    )
    parser.add_argument(
        "--merged-file",
        default=str(DEFAULT_MERGED_PATH),
        help="Output path for merged dataset",
    )
    parser.add_argument(
        "--log-file",
        default=str(DEFAULT_LOG_PATH),
        help="Log file for progress tracking",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Filter start UTC timestamp (e.g. 2022-06-13 00:00:00)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Filter end UTC timestamp (e.g. 2022-06-19 23:00:00)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="Number of concurrent worker threads (default 16)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default 30)",
    )
    parser.add_argument(
        "--max-fallback",
        type=int,
        default=6,
        help="Max fallback hours for missing forecast runs (default 6)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge existing monthly parts into final parquet and exit",
    )
    parser.add_argument(
        "--merge-after",
        action="store_true",
        help="Merge parts automatically after extraction finishes",
    )
    args = parser.parse_args()

    parts_dir = Path(args.parts_dir)
    merged_path = Path(args.merged_file)

    if args.merge:
        print(f"Merging monthly parts from {parts_dir} into {merged_path}...")
        df_merged = merge_parts(parts_dir=parts_dir, output_merged_path=merged_path)
        print(
            f"Successfully merged {len(df_merged):,} records into {merged_path} "
            f"({merged_path.stat().st_size / (1024 * 1024):.2f} MB)."
        )
        return 0

    cfg = HRRRExtractorConfig.from_yaml(args.config)
    cfg.num_threads = args.threads
    cfg.timeout_sec = args.timeout
    cfg.max_fallback_hours = args.max_fallback
    if args.start_date:
        cfg.start_date = args.start_date
    if args.end_date:
        cfg.end_date = args.end_date

    print("=================================================================")
    print("NOAA HRRR Forecast Weather Extraction (Mode F1)")
    print("=================================================================")
    print(f"Site ID: {cfg.site_id} (Grid X={cfg.plant_ix}, Y={cfg.plant_iy})")
    print(f"Window: {cfg.start_date} to {cfg.end_date}")
    print(f"Availability Lag: {cfg.weather_lag_hours}h | Max Fallback: {cfg.max_fallback_hours}h")
    print(f"Concurrency: {cfg.num_threads} threads | Timeout: {cfg.timeout_sec}s")
    print(f"Parts Directory: {parts_dir}")
    print(f"Log File: {args.log_file}")
    print("=================================================================\n")

    res = run_hrrr_pipeline(
        config=cfg,
        hourly_path=args.hourly_file,
        scan_path=args.scan_file,
        parts_dir=parts_dir,
        log_path=args.log_file,
        merge_when_done=args.merge_after,
        reverse_chronological=True,
    )

    print("\nExtraction Summary:")
    print(f"  Total daylight target hours: {res['total_daylight_hours']:,}")
    print(f"  Total monthly slices: {res['total_months']}")
    print(f"  Processed months: {res['processed_months']}")
    print(f"  Skipped (already existing): {res['skipped_months']}")
    if res["merged"]:
        print(f"  Consolidated dataset written to: {merged_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
