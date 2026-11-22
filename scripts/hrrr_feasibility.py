"""NOAA HRRR Weather Forecast Feasibility Audit for SolarCast (Task M1.1b).

Audits spatial extraction, forecast semantics, publication latency, archive coverage
(HRRR v2, v3, v4), correlation with measured PV production (System 2107), and retrieval cost.
Produces reproducible evidence, preview CSV, correlation plots, and data feasibility report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
import pyproj
import s3fs
import yaml
import zarr


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Great Circle distance between two points in km."""
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return float(2.0 * r * np.arcsin(np.sqrt(a)))


def get_grid_mapping(
    lat: float,
    lon: float,
    fs: s3fs.S3FileSystem,
) -> dict[str, Any]:
    """Find nearest HRRR grid point and chunk coordinates for a given lat/lon."""
    store = s3fs.S3Map(root="hrrrzarr/grid/HRRR_chunk_index.zarr", s3=fs, check=False)
    root = zarr.open(store, mode="r")

    with fs.open("hrrrzarr/grid/projparams.json") as f:
        proj_params = json.loads(f.read().decode("utf-8"))

    p = pyproj.Proj(proj_params)
    x_pts = root["x"][:]
    y_pts = root["y"][:]

    x_target, y_target = p(lon, lat)
    ix = int(np.argmin(np.abs(x_pts - x_target)))
    iy = int(np.argmin(np.abs(y_pts - y_target)))

    nearest_x = float(x_pts[ix])
    nearest_y = float(y_pts[iy])
    nearest_lon, nearest_lat = p(nearest_x, nearest_y, inverse=True)

    dist_km = haversine_km(lat, lon, nearest_lat, nearest_lon)

    chunk_x = int(root["chunk_x"][ix])
    chunk_y = int(root["chunk_y"][iy])
    in_chunk_x = int(root["in_chunk_x"][ix])
    in_chunk_y = int(root["in_chunk_y"][iy])
    chunk_id = f"{chunk_x}.{chunk_y}"

    return {
        "index_x": ix,
        "index_y": iy,
        "nearest_lat": nearest_lat,
        "nearest_lon": nearest_lon,
        "distance_km": dist_km,
        "chunk_id": chunk_id,
        "chunk_x": chunk_x,
        "chunk_y": chunk_y,
        "in_chunk_x": in_chunk_x,
        "in_chunk_y": in_chunk_y,
    }


def audit_grib2_idx(
    fs: s3fs.S3FileSystem,
    bucket: str = "noaa-hrrr-bdp-pds",
    run_date: str = "20220613",
    run_hour: str = "00",
    lead: str = "01",
) -> dict[str, str]:
    """Fetch sample GRIB2 .idx file and extract semantics for TMP and DSWRF."""
    idx_path = f"{bucket}/hrrr.{run_date}/conus/hrrr.t{run_hour}z.wrfsfcf{lead}.grib2.idx"
    content = fs.cat(idx_path).decode("utf-8")
    lines = content.splitlines()

    results: dict[str, str] = {}
    for line in lines:
        if "TMP:2 m above ground" in line:
            results["TMP"] = line
        elif "DSWRF:surface" in line:
            results["DSWRF"] = line

    return results


def run_coverage_probes(
    fs: s3fs.S3FileSystem,
    probe_dates: list[str],
) -> list[dict[str, Any]]:
    """Probe the Zarr archive for presence of forecast and analysis runs across model versions."""
    records = []
    for d in probe_dates:
        p = f"hrrrzarr/sfc/{d}"
        try:
            items = fs.ls(p)
            fcst_runs = [i.split("/")[-1] for i in items if i.endswith("_fcst.zarr")]
            anl_runs = [i.split("/")[-1] for i in items if i.endswith("_anl.zarr")]
            missing_fcst = 24 - len(fcst_runs)

            # Check model version
            date_int = int(d)
            if date_int < 20180712:
                model_ver = "HRRR v2"
            elif date_int < 20201202:
                model_ver = "HRRR v3"
            else:
                model_ver = "HRRR v4"

            # Check forecast shape if available
            shape_info = "N/A"
            if fcst_runs:
                sample_run = sorted(fcst_runs)[0]
                arr_path = f"{p}/{sample_run}/2m_above_ground/TMP/2m_above_ground/TMP"
                try:
                    store = s3fs.S3Map(root=arr_path, s3=fs, check=False)
                    arr = zarr.open(store, mode="r")
                    shape_info = f"{arr.shape}"
                except Exception:
                    shape_info = "read error"

            records.append(
                {
                    "date": f"{d[:4]}-{d[4:6]}-{d[6:]}",
                    "model_version": model_ver,
                    "fcst_runs_found": len(fcst_runs),
                    "anl_runs_found": len(anl_runs),
                    "missing_fcst_runs": missing_fcst,
                    "sample_fcst_shape": shape_info,
                }
            )
        except Exception as e:
            records.append(
                {
                    "date": f"{d[:4]}-{d[4:6]}-{d[6:]}",
                    "model_version": "Unknown",
                    "fcst_runs_found": 0,
                    "anl_runs_found": 0,
                    "missing_fcst_runs": 24,
                    "sample_fcst_shape": f"Error: {e}",
                }
            )
    return records


def extract_sample_week(
    hourly_df: pd.DataFrame,
    fs: s3fs.S3FileSystem,
    grid_info: dict[str, Any],
    site_lat: float,
    site_lon: float,
    lag_hours: int = 2,
    sensitivity_lag_hours: int = 3,
    prod_delay_min: int = 15,
    sample_start: str = "2022-06-13T00:00:00Z",
    sample_end: str = "2022-06-20T00:00:00Z",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract forecast and analysis data for sample week daylight hours."""
    ix, iy = grid_info["index_x"], grid_info["index_y"]

    # Filter sample week
    week_df = hourly_df[
        (hourly_df["interval_start_utc"] >= sample_start)
        & (hourly_df["interval_start_utc"] < sample_end)
    ].copy()

    # Filter daylight
    daylight_df = week_df[week_df["is_daylight"]].copy()

    location = pvlib.location.Location(latitude=site_lat, longitude=site_lon, altitude=10.0)

    rows = []
    lag_diff_count = 0
    total_daylight = len(daylight_df)

    # Cache open Zarr stores across iterations to save requests
    zarr_cache: dict[str, Any] = {}

    for _, row in daylight_df.iterrows():
        target_start = row["interval_start_utc"]
        target_end = row["interval_end_utc"]
        issue_time = target_start  # H UTC

        # Determine eligible run with lag_hours
        init_time = issue_time - pd.Timedelta(hours=lag_hours)
        init_date_str = init_time.strftime("%Y%m%d")
        init_hh_str = f"{init_time.hour:02d}"
        run_name = f"{init_date_str}_{init_hh_str}z_fcst"
        zarr_path = f"hrrrzarr/sfc/{init_date_str}/{run_name}.zarr"

        weather_avail_time = init_time + pd.Timedelta(hours=lag_hours)
        avail_evidence = (
            "Assumed from documented operational latency (NCEP HRRR dissemination "
            "completes ~50-80 min post-cycle; 2-hour lag ensures availability before issue time; "
            "NWS SCN 18-18 & 20-91)"
        )

        # Leads covering [H, H+1): F02 and F03
        lead_start = lag_hours  # e.g. 2
        lead_end = lag_hours + 1  # e.g. 3
        valid_desc = (
            f"Leads F{lead_start:02d} to F{lead_end:02d} "
            f"covering [{target_start.strftime('%Y-%m-%d %H:%M')}, "
            f"{target_end.strftime('%H:%M')}) UTC"
        )

        # Extract TMP
        tmp_path = f"{zarr_path}/2m_above_ground/TMP/2m_above_ground/TMP"
        fcst_temp_c = np.nan
        q_flags = [str(row["quality_flag"])]

        try:
            if tmp_path not in zarr_cache:
                store_tmp = s3fs.S3Map(root=tmp_path, s3=fs, check=False)
                zarr_cache[tmp_path] = zarr.open(store_tmp, mode="r")
            arr_tmp = zarr_cache[tmp_path]
            t_f1 = float(arr_tmp[lead_start - 1, iy, ix]) - 273.15
            t_f2 = float(arr_tmp[lead_end - 1, iy, ix]) - 273.15
            fcst_temp_c = float(np.round((t_f1 + t_f2) / 2.0, 2))
        except Exception:
            q_flags.append("HRRR_TMP_FETCH_FAILED")

        # Sensitivity check with lag = 3
        init_sens = issue_time - pd.Timedelta(hours=sensitivity_lag_hours)
        sens_date_str = init_sens.strftime("%Y%m%d")
        sens_hh_str = f"{init_sens.hour:02d}"
        sens_path = (
            f"hrrrzarr/sfc/{sens_date_str}/{sens_date_str}_{sens_hh_str}z_fcst.zarr"
            f"/2m_above_ground/TMP/2m_above_ground/TMP"
        )
        try:
            if sens_path not in zarr_cache:
                store_sens = s3fs.S3Map(root=sens_path, s3=fs, check=False)
                zarr_cache[sens_path] = zarr.open(store_sens, mode="r")
            arr_sens = zarr_cache[sens_path]
            sens_t_f1 = float(arr_sens[sensitivity_lag_hours - 1, iy, ix]) - 273.15
            sens_t_f2 = float(arr_sens[sensitivity_lag_hours, iy, ix]) - 273.15
            sens_temp = float((sens_t_f1 + sens_t_f2) / 2.0)
            if not np.isclose(fcst_temp_c, sens_temp, atol=0.1):
                lag_diff_count += 1
        except Exception:
            pass

        # Check DSWRF in Zarr forecast (known iris-grib #265 omission)
        fcst_irr_wm2 = np.nan
        q_flags.append("HRRR_DSWRF_MISSING_ZARR_BUG")

        # Analysis DSWRF for plausibility benchmark
        anl_date_str = target_start.strftime("%Y%m%d")
        anl_hh_str = f"{target_start.hour:02d}"
        anl_path = (
            f"hrrrzarr/sfc/{anl_date_str}/{anl_date_str}_{anl_hh_str}z_anl.zarr"
            f"/surface/DSWRF/surface/DSWRF"
        )
        anl_dswrf = np.nan
        try:
            if anl_path not in zarr_cache:
                store_anl = s3fs.S3Map(root=anl_path, s3=fs, check=False)
                zarr_cache[anl_path] = zarr.open(store_anl, mode="r")
            arr_anl = zarr_cache[anl_path]
            val = float(arr_anl[iy, ix])
            if val >= 0:
                anl_dswrf = val
        except Exception:
            pass

        # Clear sky GHI via pvlib
        mid_time = target_start + pd.Timedelta(minutes=30)
        times = pd.DatetimeIndex([mid_time])
        cs = location.get_clearsky(times)
        clearsky_ghi = float(cs["ghi"].iloc[0])

        # Production timing
        prod_cutoff = issue_time - pd.Timedelta(minutes=prod_delay_min)
        prod_avail = issue_time
        actual_avail = target_end + pd.Timedelta(minutes=prod_delay_min)

        anl_dswrf_rounded = (
            float(np.round(anl_dswrf, 1)) if not np.isnan(anl_dswrf) else np.nan
        )

        rows.append(
            {
                "site_id": str(row["site_id"]),
                "prediction_issue_time_utc": issue_time.isoformat(),
                "target_start_utc": target_start.isoformat(),
                "target_end_utc": target_end.isoformat(),
                "weather_run_init_utc": init_time.isoformat(),
                "weather_available_at_utc": weather_avail_time.isoformat(),
                "availability_evidence_or_assumption": avail_evidence,
                "weather_valid_time_or_interval": valid_desc,
                "forecast_irradiance_wm2": fcst_irr_wm2,
                "forecast_temperature_c": fcst_temp_c,
                "production_feature_cutoff_utc": prod_cutoff.isoformat(),
                "production_feature_available_at_utc": prod_avail.isoformat(),
                "actual_energy_kwh": float(np.round(row["target_kwh"], 2)),
                "actual_available_at_utc": actual_avail.isoformat(),
                "quality_flags": ";".join(q_flags),
                # Supporting plausibility fields (kept for audit & plot)
                "clearsky_ghi_wm2": float(np.round(clearsky_ghi, 1)),
                "anl_dswrf_wm2": anl_dswrf_rounded,
                "solar_elevation_deg": float(np.round(row["solar_elevation_deg"], 2)),
            }
        )

    preview_df = pd.DataFrame(rows)
    stats = {
        "total_daylight_hours": total_daylight,
        "lag_sensitivity_diff_count": lag_diff_count,
        "lag_sensitivity_pct": (lag_diff_count / total_daylight) * 100 if total_daylight else 0.0,
    }
    return preview_df, stats


def generate_plots(df: pd.DataFrame, output_path: Path) -> dict[str, float]:
    """Generate correlation plots between measured PV production and radiation/weather."""
    # Exclude any non-finite rows for correlations
    valid_mask = ~df["actual_energy_kwh"].isna() & (df["actual_energy_kwh"] >= 0)
    sub = df[valid_mask].copy()

    # Correlations
    corr_cs = float(sub["actual_energy_kwh"].corr(sub["clearsky_ghi_wm2"]))

    sub_anl = sub.dropna(subset=["anl_dswrf_wm2"])
    corr_anl = float(sub_anl["actual_energy_kwh"].corr(sub_anl["anl_dswrf_wm2"]))

    corr_tmp = float(sub["actual_energy_kwh"].corr(sub["forecast_temperature_c"]))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)

    # Plot 1: Clear-Sky GHI vs PV production
    ax = axes[0]
    ax.scatter(
        sub["clearsky_ghi_wm2"],
        sub["actual_energy_kwh"],
        color="#1f77b4",
        alpha=0.7,
        edgecolors="none",
    )
    ax.set_title(f"Target Energy vs Clear-Sky GHI\n(r = {corr_cs:.3f}, N = {len(sub)})")
    ax.set_xlabel("pvlib Clear-Sky GHI (W/m²)")
    ax.set_ylabel("Actual AC Energy (kWh)")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Plot 2: HRRR Analysis DSWRF vs PV production
    ax = axes[1]
    ax.scatter(
        sub_anl["anl_dswrf_wm2"],
        sub_anl["actual_energy_kwh"],
        color="#2ca02c",
        alpha=0.7,
        edgecolors="none",
    )
    ax.set_title(
        f"Target Energy vs HRRR Analysis DSWRF (F00)\n(r = {corr_anl:.3f}, N = {len(sub_anl)})"
    )
    ax.set_xlabel("HRRR Analysis DSWRF (W/m²)")
    ax.set_ylabel("Actual AC Energy (kWh)")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Plot 3: HRRR Forecast Temperature vs PV production
    ax = axes[2]
    ax.scatter(
        sub["forecast_temperature_c"],
        sub["actual_energy_kwh"],
        color="#d62728",
        alpha=0.7,
        edgecolors="none",
    )
    ax.set_title(f"Target Energy vs HRRR Forecast TMP (2m)\n(r = {corr_tmp:.3f}, N = {len(sub)})")
    ax.set_xlabel("HRRR Forecast 2m Temperature (°C)")
    ax.set_ylabel("Actual AC Energy (kWh)")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    return {
        "corr_clearsky_ghi": corr_cs,
        "corr_anl_dswrf": corr_anl,
        "corr_forecast_tmp": corr_tmp,
    }


def main() -> None:
    """Execute complete HRRR feasibility audit."""
    base_dir = Path("c:/projects/SolarCast")
    configs_path = base_dir / "configs/data.yaml"
    hourly_path = base_dir / "data/processed/hourly.parquet"
    preview_csv_path = base_dir / "docs/data_feasibility_preview.csv"
    plot_path = base_dir / "docs/hrrr_pv_correlation.png"

    with open(configs_path) as f:
        config = yaml.safe_load(f)

    site_lat = float(config["latitude"])
    site_lon = float(config["longitude"])
    lag_hours = int(config.get("weather_availability_lag_hours", 2))
    sens_lag = int(config.get("weather_lag_sensitivity_hours", 3))
    prod_delay = int(config.get("production_available_delay_minutes", 15))

    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-1"})

    print("=== Step 1: Spatial Grid Mapping ===")
    grid_info = get_grid_mapping(site_lat, site_lon, fs)
    print(
        f"Nearest HRRR grid point: index_x={grid_info['index_x']}, index_y={grid_info['index_y']}"
    )
    print(f"Distance to plant: {grid_info['distance_km']:.3f} km")
    print(
        f"Chunk ID: {grid_info['chunk_id']} "
        f"(in_chunk_x={grid_info['in_chunk_x']}, in_chunk_y={grid_info['in_chunk_y']})"
    )

    print("\n=== Step 2: Semantics Audit (GRIB2 .idx) ===")
    grib_sem = audit_grib2_idx(fs)
    for k, v in grib_sem.items():
        print(f"  {k}: {v}")

    print("\n=== Step 4: Coverage Probes Across Versions ===")
    probe_days = [
        "20170115",
        "20170715",
        "20180115",
        "20180715",
        "20190615",
        "20200615",
        "20210615",
        "20221215",
        "20230615",
    ]
    probes = run_coverage_probes(fs, probe_days)
    for p in probes:
        print(
            f"  {p['date']} ({p['model_version']}): fcst={p['fcst_runs_found']}/24, "
            f"anl={p['anl_runs_found']}/24, shape={p['sample_fcst_shape']}"
        )

    print("\n=== Step 3 & 5: Sample Week Extraction & Correlations ===")
    hourly_df = pd.read_parquet(hourly_path)
    full_sample_df, stats = extract_sample_week(
        hourly_df=hourly_df,
        fs=fs,
        grid_info=grid_info,
        site_lat=site_lat,
        site_lon=site_lon,
        lag_hours=lag_hours,
        sensitivity_lag_hours=sens_lag,
        prod_delay_min=prod_delay,
    )

    # Save exact 15-column preview CSV for ~48 daylight rows (first 48 daylight rows)
    exact_cols = [
        "site_id",
        "prediction_issue_time_utc",
        "target_start_utc",
        "target_end_utc",
        "weather_run_init_utc",
        "weather_available_at_utc",
        "availability_evidence_or_assumption",
        "weather_valid_time_or_interval",
        "forecast_irradiance_wm2",
        "forecast_temperature_c",
        "production_feature_cutoff_utc",
        "production_feature_available_at_utc",
        "actual_energy_kwh",
        "actual_available_at_utc",
        "quality_flags",
    ]
    preview_48 = full_sample_df.head(48)[exact_cols].copy()
    preview_48.to_csv(preview_csv_path, index=False)
    print(f"Saved {len(preview_48)} rows to {preview_csv_path}")

    # Generate plots
    corrs = generate_plots(full_sample_df, plot_path)
    print(f"Saved correlation plots to {plot_path}")
    print(
        f"Correlations: Clear-Sky GHI r={corrs['corr_clearsky_ghi']:.3f}, "
        f"Anl DSWRF r={corrs['corr_anl_dswrf']:.3f}, TMP r={corrs['corr_forecast_tmp']:.3f}"
    )
    print(
        f"Sensitivity check (lag 2 vs 3): {stats['lag_sensitivity_diff_count']} "
        f"differing values ({stats['lag_sensitivity_pct']:.1f}%)"
    )


if __name__ == "__main__":
    main()
