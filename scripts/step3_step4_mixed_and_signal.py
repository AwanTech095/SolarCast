"""Mixed-Weather Extraction & Cloud Cover Signal Check (Steps 3 & 4).

Selects mixed-weather week (2022-12-26 to 2023-01-01) where all hourly rows are VALID
and clear-sky index proxy variance is maximal.
Extracts HRRR Zarr forecast TMP (2m) and TCDC (entire_atmosphere) at the plant
with weather_availability_lag_hours=2 (averaging leads F02 and F03).
Computes Spearman rank correlation between clear-sky index proxy and forecast TCDC
for VALID high_sun hours across both the mixed winter week and the June sample week.
Generates scatter plots in docs/tcdc_clearsky_proxy_correlation.png.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
import s3fs
import zarr
from scipy import stats

# Configuration & coordinates
SITE_LAT = 38.996306
SITE_LON = -122.134111
PLANT_IX = 198
PLANT_IY = 642
LAG_HOURS = 2

HOURLY_FILE = Path("data/processed/hourly.parquet")
OUT_DIR = Path("data/processed")
DOCS_DIR = Path("docs")
PLOT_FILE = DOCS_DIR / "tcdc_clearsky_proxy_correlation.png"


def extract_week_forecast(
    df_week: pd.DataFrame,
    fs: s3fs.S3FileSystem,
    location: pvlib.location.Location,
    p95_norm: float,
) -> pd.DataFrame:
    """Extract forecast TMP, TCDC and calculate clear-sky proxy for a 1-week dataframe."""
    zarr_cache: dict[str, Any] = {}
    rows = []

    for _, row in df_week.iterrows():
        t_start = pd.to_datetime(row["interval_start_utc"], utc=True)
        t_end = pd.to_datetime(row["interval_end_utc"], utc=True)
        issue_time = t_start

        # Initialization time: H - 2 hours
        init_time = issue_time - pd.Timedelta(hours=LAG_HOURS)
        init_date_str = init_time.strftime("%Y%m%d")
        init_hh_str = f"{init_time.hour:02d}"
        run_name = f"{init_date_str}_{init_hh_str}z_fcst"
        base_zarr = f"hrrrzarr/sfc/{init_date_str}/{run_name}.zarr"

        lead_start = LAG_HOURS      # F02 -> index 1
        lead_end = LAG_HOURS + 1    # F03 -> index 2

        # Extract TMP
        tmp_path = f"{base_zarr}/2m_above_ground/TMP/2m_above_ground/TMP"
        fcst_tmp_c = np.nan
        try:
            if tmp_path not in zarr_cache:
                store_tmp = s3fs.S3Map(root=tmp_path, s3=fs, check=False)
                zarr_cache[tmp_path] = zarr.open(store_tmp, mode="r")
            arr_tmp = zarr_cache[tmp_path]
            t_f1 = float(arr_tmp[lead_start - 1, PLANT_IY, PLANT_IX]) - 273.15
            t_f2 = float(arr_tmp[lead_end - 1, PLANT_IY, PLANT_IX]) - 273.15
            fcst_tmp_c = float(np.round((t_f1 + t_f2) / 2.0, 2))
        except Exception:
            pass

        # Extract TCDC
        tcdc_path = f"{base_zarr}/entire_atmosphere/TCDC/entire_atmosphere/TCDC"
        fcst_tcdc_pct = np.nan
        try:
            if tcdc_path not in zarr_cache:
                store_tcdc = s3fs.S3Map(root=tcdc_path, s3=fs, check=False)
                zarr_cache[tcdc_path] = zarr.open(store_tcdc, mode="r")
            arr_tcdc = zarr_cache[tcdc_path]
            c_f1 = float(arr_tcdc[lead_start - 1, PLANT_IY, PLANT_IX])
            c_f2 = float(arr_tcdc[lead_end - 1, PLANT_IY, PLANT_IX])
            fcst_tcdc_pct = float(np.round((c_f1 + c_f2) / 2.0, 1))
        except Exception:
            pass

        # Clear-sky GHI at midpoint
        mid_time = pd.DatetimeIndex([t_start + pd.Timedelta(minutes=30)])
        cs = location.get_clearsky(mid_time)
        clearsky_ghi = float(cs["ghi"].iloc[0])

        target_kwh = float(row["target_kwh"])
        is_high_sun = bool(row["high_sun"])
        is_valid = (row["quality_flag"] == "VALID")

        # Clear-sky proxy: (target_kwh / clearsky_ghi) / p95
        kt_proxy = np.nan
        if is_valid and is_high_sun and clearsky_ghi > 10.0:
            kt_proxy = (target_kwh / clearsky_ghi) / p95_norm

        rows.append({
            "site_id": row["site_id"],
            "interval_start_utc": t_start,
            "interval_end_utc": t_end,
            "weather_run_init_utc": init_time,
            "forecast_tmp_c": fcst_tmp_c,
            "forecast_tcdc_pct": fcst_tcdc_pct,
            "clearsky_ghi": clearsky_ghi,
            "target_kwh": target_kwh,
            "kt_proxy": kt_proxy,
            "is_daylight": bool(row["is_daylight"]),
            "high_sun": is_high_sun,
            "quality_flag": str(row["quality_flag"]),
        })

    return pd.DataFrame(rows)


def main() -> None:
    """Execute Step 3 mixed-weather extraction and Step 4 signal check."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load hourly dataset
    df_all = pd.read_parquet(HOURLY_FILE)
    df_all["interval_start_utc"] = pd.to_datetime(df_all["interval_start_utc"], utc=True)
    df_all["interval_end_utc"] = pd.to_datetime(df_all["interval_end_utc"], utc=True)

    location = pvlib.location.Location(latitude=SITE_LAT, longitude=SITE_LON, altitude=10.0)

    # Calculate global 95th percentile of target_kwh / clearsky_ghi over all VALID high_sun hours
    mid_times = pd.DatetimeIndex(df_all["interval_start_utc"] + pd.Timedelta(minutes=30))
    all_cs = location.get_clearsky(mid_times)
    df_all["clearsky_ghi"] = all_cs["ghi"].values

    mask_high = (
        (df_all["quality_flag"] == "VALID")
        & (df_all["high_sun"])
        & (df_all["clearsky_ghi"] > 10.0)
    )
    ratios = df_all.loc[mask_high, "target_kwh"] / df_all.loc[mask_high, "clearsky_ghi"]
    p95 = float(np.percentile(ratios.dropna(), 95))
    print(f"Normalisation factor p95 (target_kwh / clearsky_ghi): {p95:.6f}")

    # 2. Extract Mixed Week (2022-12-26 to 2023-01-01)
    print("\n--- Extracting Mixed-Weather Week (2022-12-26 to 2023-01-01) ---")
    week_mixed_raw = df_all[
        (df_all["interval_start_utc"] >= "2022-12-26 00:00:00Z")
        & (df_all["interval_start_utc"] < "2023-01-02 00:00:00Z")
    ].copy()
    assert len(week_mixed_raw) == 168, f"Expected 168 rows, got {len(week_mixed_raw)}"
    assert (week_mixed_raw["quality_flag"] == "VALID").all(), "All rows must be VALID"

    fs = s3fs.S3FileSystem(
        anon=True, client_kwargs={"region_name": "us-west-1"}, default_cache_type="none"
    )
    df_mixed = extract_week_forecast(week_mixed_raw, fs, location, p95)
    df_mixed.to_parquet(OUT_DIR / "mixed_week_extracted.parquet", index=False)
    print(f"Extracted {len(df_mixed)} hours for mixed week.")

    # 3. Extract June Sample Week (2022-06-13 to 2022-06-19)
    print("\n--- Extracting June Sample Week (2022-06-13 to 2022-06-20) ---")
    week_june_raw = df_all[
        (df_all["interval_start_utc"] >= "2022-06-13 00:00:00Z")
        & (df_all["interval_start_utc"] < "2022-06-20 00:00:00Z")
    ].copy()
    assert len(week_june_raw) == 168, f"Expected 168 rows, got {len(week_june_raw)}"
    assert (week_june_raw["quality_flag"] == "VALID").all(), "All rows must be VALID"

    df_june = extract_week_forecast(week_june_raw, fs, location, p95)
    df_june.to_parquet(OUT_DIR / "june_week_extracted.parquet", index=False)
    print(f"Extracted {len(df_june)} hours for June week.")

    # 4. Correlation Analysis for VALID high_sun hours
    mixed_hs = df_mixed[df_mixed["high_sun"] & df_mixed["kt_proxy"].notna()].copy()
    june_hs = df_june[df_june["high_sun"] & df_june["kt_proxy"].notna()].copy()

    # Drop any NaNs in forecast variables if present
    mixed_clean = mixed_hs.dropna(subset=["kt_proxy", "forecast_tcdc_pct"])
    june_clean = june_hs.dropna(subset=["kt_proxy", "forecast_tcdc_pct"])

    rho_mixed, p_mixed = stats.spearmanr(mixed_clean["forecast_tcdc_pct"], mixed_clean["kt_proxy"])
    r_mixed, pr_mixed = stats.pearsonr(mixed_clean["forecast_tcdc_pct"], mixed_clean["kt_proxy"])

    rho_june, p_june = stats.spearmanr(june_clean["forecast_tcdc_pct"], june_clean["kt_proxy"])
    r_june, pr_june = stats.pearsonr(june_clean["forecast_tcdc_pct"], june_clean["kt_proxy"])

    print("\n=== Signal Check Results (VALID high_sun hours) ===")
    print(f"Mixed Winter Week (2022-12-26 to 2023-01-01, N = {len(mixed_clean)}):")
    print(f"  Spearman rho = {rho_mixed:.4f} (p-value = {p_mixed:.4e})")
    print(f"  Pearson r    = {r_mixed:.4f} (p-value = {pr_mixed:.4e})")
    kt_mean_m = mixed_clean["kt_proxy"].mean()
    kt_std_m = mixed_clean["kt_proxy"].std()
    print(f"  kt_proxy mean = {kt_mean_m:.3f}, std = {kt_std_m:.3f}")
    tcdc_mean_m = mixed_clean["forecast_tcdc_pct"].mean()
    tcdc_std_m = mixed_clean["forecast_tcdc_pct"].std()
    print(f"  TCDC mean     = {tcdc_mean_m:.1f}%, std = {tcdc_std_m:.1f}%")

    print(f"\nJune Summer Week (2022-06-13 to 2022-06-19, N = {len(june_clean)}):")
    print(f"  Spearman rho = {rho_june:.4f} (p-value = {p_june:.4e})")
    print(f"  Pearson r    = {r_june:.4f} (p-value = {pr_june:.4e})")
    kt_mean_j = june_clean["kt_proxy"].mean()
    kt_std_j = june_clean["kt_proxy"].std()
    print(f"  kt_proxy mean = {kt_mean_j:.3f}, std = {kt_std_j:.3f}")
    tcdc_mean_j = june_clean["forecast_tcdc_pct"].mean()
    tcdc_std_j = june_clean["forecast_tcdc_pct"].std()
    print(f"  TCDC mean     = {tcdc_mean_j:.1f}%, std = {tcdc_std_j:.1f}%")

    # 5. Scatter Plot Generation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Left: Mixed Week
    ax1 = axes[0]
    sc1 = ax1.scatter(
        mixed_clean["forecast_tcdc_pct"],
        mixed_clean["kt_proxy"],
        c=mixed_clean["clearsky_ghi"],
        cmap="viridis",
        s=45,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.5,
    )
    # Regression line for visual trend
    m1, b1 = np.polyfit(mixed_clean["forecast_tcdc_pct"], mixed_clean["kt_proxy"], 1)
    x_grid1 = np.linspace(0, 100, 100)
    ax1.plot(
        x_grid1,
        m1 * x_grid1 + b1,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label="Linear Trend",
    )
    ax1.set_title(
        f"Mixed Winter Week (2022-12-26 to 2023-01-01)\n"
        f"Spearman $r_s = {rho_mixed:.3f}$ ($p = {p_mixed:.2e}$), $N = {len(mixed_clean)}$",
        fontsize=11,
    )
    ax1.set_xlabel("HRRR Forecast Total Cloud Cover TCDC (%) [Lag=2h]", fontsize=10)
    ax1.set_ylabel(
        r"Clear-Sky Index Proxy ($k_t = \frac{\mathrm{target}}{\mathrm{GHI}_{cs}} / p95$)",
        fontsize=10,
    )
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(-0.05, 1.45)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label("Clear-Sky GHI (W/m²)", fontsize=9)

    # Right: June Week
    ax2 = axes[1]
    sc2 = ax2.scatter(
        june_clean["forecast_tcdc_pct"],
        june_clean["kt_proxy"],
        c=june_clean["clearsky_ghi"],
        cmap="viridis",
        s=45,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.5,
    )
    if len(june_clean["forecast_tcdc_pct"].unique()) > 1:
        m2, b2 = np.polyfit(june_clean["forecast_tcdc_pct"], june_clean["kt_proxy"], 1)
        x_grid2 = np.linspace(0, 100, 100)
        ax2.plot(
            x_grid2,
            m2 * x_grid2 + b2,
            color="crimson",
            linestyle="--",
            linewidth=1.5,
            label="Linear Trend",
        )
    ax2.set_title(
        f"June Sample Week (2022-06-13 to 2022-06-19)\n"
        f"Spearman $r_s = {rho_june:.3f}$ ($p = {p_june:.2e}$), $N = {len(june_clean)}$",
        fontsize=11,
    )
    ax2.set_xlabel("HRRR Forecast Total Cloud Cover TCDC (%) [Lag=2h]", fontsize=10)
    ax2.set_xlim(-5, 105)
    ax2.set_ylim(-0.05, 1.45)
    ax2.grid(True, linestyle=":", alpha=0.6)
    if len(june_clean["forecast_tcdc_pct"].unique()) > 1:
        ax2.legend(loc="upper right")
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label("Clear-Sky GHI (W/m²)", fontsize=9)

    plt.suptitle(
        "Forecast Cloud Cover vs Clear-Sky Index Proxy (VALID high_sun hours)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=180)
    plt.close()
    print(f"Saved scatter plot to {PLOT_FILE}")


if __name__ == "__main__":
    main()
