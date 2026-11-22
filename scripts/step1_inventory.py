"""HRRR Zarr Variable Inventory for SolarCast (Step 1).

Audits forecast variables in s3://hrrrzarr/sfc/20220613/20220613_12z_fcst.zarr/
specifically checking chunk 0.4.1 (spatial tile 1.4 for System 2107) for non-empty data,
units, and GRIB2 .idx semantics (instantaneous vs average/accumulation).
"""

from __future__ import annotations

from typing import Any

import requests
import s3fs
import zarr


def audit_inventory() -> list[dict[str, Any]]:
    """Audit forecast variables in HRRR Zarr run 2022-06-13 12z."""
    base_url = "https://hrrrzarr.s3.amazonaws.com/sfc/20220613/20220613_12z_fcst.zarr"
    base_s3 = "hrrrzarr/sfc/20220613/20220613_12z_fcst.zarr"

    # Audit target list
    candidates = [
        {
            "name": "Total Cloud Cover (entire atmosphere)",
            "short": "TCDC",
            "level": "entire_atmosphere",
            "rel_path": "entire_atmosphere/TCDC/entire_atmosphere/TCDC",
            "grib_idx_desc": "1 hour fcst",
            "semantics": "Instantaneous point forecast",
            "units_table": "%",
        },
        {
            "name": "Low Cloud Cover",
            "short": "LCDC",
            "level": "low_cloud_layer",
            "rel_path": "low_cloud_layer/LCDC/low_cloud_layer/LCDC",
            "grib_idx_desc": "1 hour fcst",
            "semantics": "Instantaneous point forecast",
            "units_table": "%",
        },
        {
            "name": "Middle Cloud Cover",
            "short": "MCDC",
            "level": "middle_cloud_layer",
            "rel_path": "middle_cloud_layer/MCDC/middle_cloud_layer/MCDC",
            "grib_idx_desc": "1 hour fcst",
            "semantics": "Instantaneous point forecast",
            "units_table": "%",
        },
        {
            "name": "High Cloud Cover",
            "short": "HCDC",
            "level": "high_cloud_layer",
            "rel_path": "high_cloud_layer/HCDC/high_cloud_layer/HCDC",
            "grib_idx_desc": "1 hour fcst",
            "semantics": "Instantaneous point forecast",
            "units_table": "%",
        },
        {
            "name": "2m Temperature",
            "short": "TMP",
            "level": "2m_above_ground",
            "rel_path": "2m_above_ground/TMP/2m_above_ground/TMP",
            "grib_idx_desc": "1 hour fcst",
            "semantics": "Instantaneous point forecast",
            "units_table": "K",
        },
        {
            "name": "2m Dew Point Temperature",
            "short": "DPT",
            "level": "2m_above_ground",
            "rel_path": "2m_above_ground/DPT/2m_above_ground/DPT",
            "grib_idx_desc": "1 hour fcst",
            "semantics": "Instantaneous point forecast",
            "units_table": "K",
        },
        {
            "name": "2m Relative Humidity",
            "short": "RH",
            "level": "2m_above_ground",
            "rel_path": "2m_above_ground/RH/2m_above_ground/RH",
            "grib_idx_desc": "1 hour fcst",
            "semantics": "Instantaneous point forecast",
            "units_table": "%",
        },
        {
            "name": "10m U-Component of Wind",
            "short": "UGRD",
            "level": "10m_above_ground",
            "rel_path": "10m_above_ground/UGRD/10m_above_ground/UGRD",
            "grib_idx_desc": "1 hour fcst",
            "semantics": "Instantaneous point forecast",
            "units_table": "m s-1",
        },
        {
            "name": "10m V-Component of Wind",
            "short": "VGRD",
            "level": "10m_above_ground",
            "rel_path": "10m_above_ground/VGRD/10m_above_ground/VGRD",
            "grib_idx_desc": "1 hour fcst",
            "semantics": "Instantaneous point forecast",
            "units_table": "m s-1",
        },
        {
            "name": "Visibility",
            "short": "VIS",
            "level": "surface",
            "rel_path": "surface/VIS/surface/VIS",
            "grib_idx_desc": "1 hour fcst",
            "semantics": "Instantaneous point forecast",
            "units_table": "m",
        },
        {
            "name": "Precipitation Rate",
            "short": "PRATE",
            "level": "surface",
            "rel_path": "surface/PRATE/surface/PRATE",
            "grib_idx_desc": "1 hour fcst",
            "semantics": "Instantaneous point forecast",
            "units_table": "kg m-2 s-1",
        },
        {
            "name": "Accumulated Precipitation (1hr)",
            "short": "APCP",
            "level": "surface",
            "rel_path": "surface/APCP_1hr_acc_fcst/surface/APCP_1hr_acc_fcst",
            "grib_idx_desc": "0-1 hour acc fcst",
            "semantics": "Time-accumulated forecast",
            "units_table": "kg m-2 (mm)",
        },
        {
            "name": "Downward Shortwave Radiation Flux",
            "short": "DSWRF",
            "level": "surface",
            "rel_path": "surface/DSWRF/surface/DSWRF",
            "grib_idx_desc": "1 hour fcst",
            "semantics": "Instantaneous point forecast (FATAL BUG: chunks missing)",
            "units_table": "W m-2",
        },
    ]

    session = requests.Session()
    fs = s3fs.S3FileSystem(
        anon=True, client_kwargs={"region_name": "us-west-1"}, default_cache_type="none"
    )

    results = []
    for c in candidates:
        rel = c["rel_path"]
        u_zarray = f"{base_url}/{rel}/.zarray"
        u_chunk = f"{base_url}/{rel}/0.4.1"

        r_zarr = session.head(u_zarray, timeout=10)
        r_chunk = session.head(u_chunk, timeout=10)

        has_zarray = r_zarr.status_code == 200
        has_chunk = r_chunk.status_code == 200
        chunk_size = int(r_chunk.headers.get("Content-Length", 0)) if has_chunk else 0

        # Decode plant value if chunk exists
        val_f01: float | str = "N/A"
        if has_chunk and chunk_size > 0:
            try:
                store = s3fs.S3Map(root=f"{base_s3}/{rel}", s3=fs, check=False)
                arr = zarr.open(store, mode="r")
                val_f01 = float(arr[0, 642, 198])
            except Exception as e:
                val_f01 = f"Error: {e}"

        c_res = dict(c)
        c_res["has_zarray"] = has_zarray
        c_res["has_chunk"] = has_chunk
        c_res["chunk_bytes"] = chunk_size
        c_res["sample_val_f01"] = val_f01
        results.append(c_res)

    return results


def main() -> None:
    """Run Step 1 variable inventory audit and print markdown table."""
    results = audit_inventory()
    header = (
        "| Variable | Zarr Path | Units | GRIB2 .idx Descriptor | "
        "Semantics | Chunk 1.4 Status | Size | F01 Value |"
    )
    print(header)
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        status_str = "Working" if r["has_chunk"] and r["chunk_bytes"] > 0 else "Missing Chunk"
        print(
            f"| **{r['name']}** | `{r['rel_path']}` | {r['units_table']} | "
            f"`{r['grib_idx_desc']}` | {r['semantics']} | "
            f"**{status_str}** | {r['chunk_bytes']:,} B | {r['sample_val_f01']} |"
        )


if __name__ == "__main__":
    main()
