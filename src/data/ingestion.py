"""Data ingestion module for SolarCast (PVDAQ production meter & NASA POWER weather).

Registers raw PVDAQ meter data and metadata from OEDI data lake, fetches NASA POWER
hourly weather in yearly chunks with retries and backoff, and maintains an immutable
data manifest with SHA-256 checksums and provenance records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import botocore
import requests
import yaml
from botocore.config import Config

logger = logging.getLogger(__name__)


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 hex digest of a local file.

    Args:
        file_path: Path to the target file.
        chunk_size: Byte size of chunks to stream.

    Returns:
        Hex-encoded SHA-256 string.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def count_csv_rows(file_path: Path) -> int:
    """Count data rows in a CSV file excluding the header line.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Integer count of non-header lines.
    """
    count = 0
    with open(file_path, encoding="utf-8", errors="replace") as f:
        # Skip header
        f.readline()
        for line in f:
            if line.strip():
                count += 1
    return count


def register_pvdaq_files(
    config: dict[str, Any],
    repo_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Verify or download raw PVDAQ files (meter CSV and system metadata).

    Args:
        config: Loaded data configuration dictionary.
        repo_root: Root repository directory for resolving relative paths.

    Returns:
        Dictionary mapping artifact identifiers to manifest metadata dictionaries.
    """
    root = repo_root or Path.cwd()
    meter_path = root / config["paths"]["meter_csv"]
    meta_path = root / config["paths"]["metadata_json"]

    meter_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    s3_bucket = config["sources"]["pvdaq"]["s3_bucket"]
    s3_prefix = config["sources"]["pvdaq"]["s3_prefix"]

    meter_s3_key = f"{s3_prefix}data/2107_meter_15m_data.csv"
    meta_s3_key = f"{s3_prefix}metadata/2107_system_metadata.json"

    # Only initialize S3 client if download is required
    s3_client = None
    if not meter_path.exists() or not meta_path.exists():
        s3_client = boto3.client(
            "s3",
            config=Config(signature_version=botocore.UNSIGNED),
        )

    # Register / download meter CSV
    if not meter_path.exists() or meter_path.stat().st_size == 0:
        logger.info("Downloading PVDAQ meter CSV from s3://%s/%s...", s3_bucket, meter_s3_key)
        assert s3_client is not None
        s3_client.download_file(s3_bucket, meter_s3_key, str(meter_path))
    else:
        logger.info("PVDAQ meter CSV already present at %s", meter_path)

    # Register / download metadata JSON
    if not meta_path.exists() or meta_path.stat().st_size == 0:
        logger.info("Downloading PVDAQ metadata from s3://%s/%s...", s3_bucket, meta_s3_key)
        assert s3_client is not None
        s3_client.download_file(s3_bucket, meta_s3_key, str(meta_path))
    else:
        logger.info("PVDAQ metadata JSON already present at %s", meta_path)

    # Compute provenance metadata
    meter_sha = compute_sha256(meter_path)
    meter_rows = count_csv_rows(meter_path)

    meta_sha = compute_sha256(meta_path)

    return {
        "pvdaq_meter_15m": {
            "path": str(meter_path.relative_to(root)),
            "source_url": f"s3://{s3_bucket}/{meter_s3_key}",
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "sha256": meter_sha,
            "bytes": meter_path.stat().st_size,
            "row_count": meter_rows,
            "units": "kW (instantaneous power avg)",
            "aggregation": "15-minute average",
            "time_alignment": config.get("time_interval_alignment", "L"),
            "licence": config["sources"]["pvdaq"]["licence"],
            "doi": config["sources"]["pvdaq"]["doi"],
        },
        "pvdaq_metadata": {
            "path": str(meta_path.relative_to(root)),
            "source_url": f"s3://{s3_bucket}/{meta_s3_key}",
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "sha256": meta_sha,
            "bytes": meta_path.stat().st_size,
            "row_count": 1,
            "units": "metadata (JSON)",
            "licence": config["sources"]["pvdaq"]["licence"],
            "doi": config["sources"]["pvdaq"]["doi"],
        },
    }


def fetch_nasa_power_year(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    parameters: list[str],
    output_path: Path,
    api_url: str = "https://power.larc.nasa.gov/api/temporal/hourly/point",
    max_retries: int = 2,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Fetch hourly NASA POWER weather data for a single date window with retry logic.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        start_date: Start date formatted as YYYYMMDD.
        end_date: End date formatted as YYYYMMDD.
        parameters: List of NASA POWER parameter strings.
        output_path: Destination JSON file path.
        api_url: Base API URL for NASA POWER.
        max_retries: Number of retry attempts on network error.
        session: Optional pre-configured requests session (useful for unit testing).

    Returns:
        Summary dictionary containing row count and SHA-256 digest.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Idempotency check: if destination file exists and is valid JSON, reuse it
    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            with open(output_path, encoding="utf-8") as f:
                cached = json.load(f)
            props = cached.get("properties", {}).get("parameter", {})
            param_key = parameters[0]
            if param_key in props and len(props[param_key]) > 0:
                logger.info("Using cached NASA POWER file at %s", output_path)
                return {
                    "path": str(output_path),
                    "sha256": compute_sha256(output_path),
                    "bytes": output_path.stat().st_size,
                    "hourly_records": len(props[param_key]),
                }
        except Exception as e:
            logger.warning(
                "Existing weather cache at %s invalid (%s); re-fetching.", output_path, e
            )

    req_params = {
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "parameters": ",".join(parameters),
        "community": "RE",
        "start": start_date,
        "end": end_date,
        "format": "JSON",
        "time-standard": "UTC",
    }

    client = session or requests.Session()
    last_err: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            logger.info(
                "Fetching NASA POWER weather for %s to %s (attempt %d/%d)...",
                start_date,
                end_date,
                attempt + 1,
                max_retries + 1,
            )
            response = client.get(api_url, params=req_params, timeout=45)
            response.raise_for_status()
            data = response.json()

            # Ensure expected payload structure
            props = data.get("properties", {}).get("parameter", {})
            if not props:
                msg = f"Incomplete NASA POWER payload received for {start_date}-{end_date}"
                raise ValueError(msg)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            hourly_count = len(props.get(parameters[0], {}))
            return {
                "path": str(output_path),
                "sha256": compute_sha256(output_path),
                "bytes": output_path.stat().st_size,
                "hourly_records": hourly_count,
            }

        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                backoff_sec = 2.0**attempt
                logger.warning(
                    "NASA POWER fetch failed (%s). Retrying in %.1f seconds...", exc, backoff_sec
                )
                time.sleep(backoff_sec)

    msg = f"Failed to fetch NASA POWER weather after {max_retries + 1} attempts: {last_err}"
    raise RuntimeError(msg) from last_err


def fetch_all_weather(
    config: dict[str, Any],
    repo_root: Path | None = None,
    session: requests.Session | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch complete historical weather coverage in yearly slices.

    Args:
        config: Loaded data configuration dictionary.
        repo_root: Root repository directory.
        session: Optional HTTP session for testing.

    Returns:
        Dictionary mapping year keys to weather file manifest entries.
    """
    root = repo_root or Path.cwd()
    weather_dir = root / config["paths"]["weather_dir"]
    weather_dir.mkdir(parents=True, exist_ok=True)

    lat = float(config["latitude"])
    lon = float(config["longitude"])
    params = config["weather_parameters"]
    api_url = config["sources"]["nasa_power"]["api_url"]

    start_year = int(config["start_date"].split("-")[0])
    end_year = int(config["end_date"].split("-")[0])

    manifest_entries = {}

    for year in range(start_year, end_year + 1):
        # Format start and end date for the specific year
        if year == start_year:
            y_start = config["start_date"].replace("-", "")
        else:
            y_start = f"{year}0101"

        if year == end_year:
            y_end = config["end_date"].replace("-", "")
        else:
            y_end = f"{year}1231"

        out_file = weather_dir / f"nasa_power_{year}.json"
        res = fetch_nasa_power_year(
            lat=lat,
            lon=lon,
            start_date=y_start,
            end_date=y_end,
            parameters=params,
            output_path=out_file,
            api_url=api_url,
            session=session,
        )

        manifest_entries[f"nasa_power_weather_{year}"] = {
            "path": str(out_file.relative_to(root)),
            "source_url": api_url,
            "query_window": f"{y_start}-{y_end}",
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "sha256": res["sha256"],
            "bytes": res["bytes"],
            "row_count": res["hourly_records"],
            "parameters": params,
            "units": {
                "ALLSKY_SFC_SW_DWN": "Wh/m^2",
                "T2M": "C",
                "RH2M": "%",
                "WS10M": "m/s",
            },
            "time_standard": "UTC",
            "licence": config["sources"]["nasa_power"]["licence"],
        }

    return manifest_entries


def write_manifest(
    manifest_path: Path,
    entries: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write an immutable, traceable manifest JSON documenting all raw assets.

    Args:
        manifest_path: Destination path for manifest.json.
        entries: Artifact entries mapping asset name to asset metadata.
        metadata: High-level dataset metadata.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_version": "1.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_metadata": metadata or {},
        "artifacts": entries,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved data manifest to %s (%d artifacts recorded)", manifest_path, len(entries))


def run_ingestion(
    config_path: Path,
    repo_root: Path | None = None,
    session: requests.Session | None = None,
) -> Path:
    """Execute complete end-to-end data ingestion pipeline.

    Args:
        config_path: Path to configs/data.yaml.
        repo_root: Base repository root directory.
        session: Optional HTTP session for network mocking.

    Returns:
        Path to the generated data manifest file.
    """
    root = repo_root or Path.cwd()
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Starting SolarCast ingestion for Site %s...", config.get("site_id"))

    # 1. Register / verify PVDAQ generation data
    pvdaq_entries = register_pvdaq_files(config, repo_root=root)

    # 2. Fetch NASA POWER weather across all years
    weather_entries = fetch_all_weather(config, repo_root=root, session=session)

    # 3. Compile and write manifest
    all_artifacts = {**pvdaq_entries, **weather_entries}
    manifest_path = root / config["paths"]["manifest_path"]

    dataset_meta = {
        "site_id": config.get("site_id"),
        "system_id": config.get("system_id"),
        "public_name": config.get("public_name"),
        "latitude": config.get("latitude"),
        "longitude": config.get("longitude"),
        "elevation_m": config.get("elevation_m"),
        "capacity_dc_kw": config.get("capacity_dc_kw"),
        "capacity_ac_nominal_kw": config.get("capacity_ac_nominal_kw"),
        "capacity_ac_max_observed_kw": config.get("capacity_ac_max_observed_kw"),
        "start_date": config.get("start_date"),
        "end_date": config.get("end_date"),
        "timezone": config.get("timezone"),
        "simulated_weather_lag_days": config.get("simulated_weather_lag_days"),
    }

    write_manifest(manifest_path, all_artifacts, dataset_meta)
    return manifest_path


def main() -> None:
    """Command-line entrypoint for data ingestion."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="SolarCast Data Ingestion")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to data configuration YAML file",
    )
    args = parser.parse_args()
    run_ingestion(args.config)


if __name__ == "__main__":
    main()
