"""Data validation schemas and quarantine logic using Pandera."""

from pathlib import Path

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

# -----------------------------------------------------------------------------
# Pandera Schemas
# -----------------------------------------------------------------------------

# Raw 15-minute generation schema
raw_generation_schema = DataFrameSchema(
    columns={
        "measured_on": Column(
            pa.String,
            nullable=False,
            description="Local civil timestamp string (YYYY-MM-DD HH:MM:SS)",
        ),
        "meter_revenue_grade_ac_output_meter_149578": Column(
            pa.Float64,
            nullable=False,
            coerce=True,
            description="15-minute average AC active power in kW",
        ),
    },
    strict=False,
    coerce=True,
)

# Quarantined records schema
quarantine_schema = DataFrameSchema(
    columns={
        "measured_on": Column(pa.String, nullable=False),
        "power_kw": Column(pa.Float64, nullable=True),
        "reason": Column(
            pa.String,
            checks=Check.isin([
                "duplicate_timestamp",
                "above_893_kw",
                "negative_power",
                "non_finite",
            ]),
            nullable=False,
        ),
    },
    strict=False,
)

# Aggregated hourly dataset schema
hourly_schema = DataFrameSchema(
    columns={
        "site_id": Column(pa.String, nullable=False),
        "interval_start_utc": Column("datetime64[ns, UTC]", nullable=False),
        "interval_end_utc": Column("datetime64[ns, UTC]", nullable=False),
        "target_kwh": Column(
            pa.Float64,
            checks=[
                Check.greater_than_or_equal_to(0.0),
                Check.less_than_or_equal_to(893.0),
            ],
            nullable=True,
        ),
        "capacity_kw": Column(pa.Float64, checks=Check.greater_than(0.0), nullable=False),
        "capacity_ac_observed_kw": Column(
            pa.Float64, checks=Check.greater_than(0.0), nullable=False
        ),
        "latitude": Column(pa.Float64, checks=Check.in_range(-90.0, 90.0), nullable=False),
        "longitude": Column(pa.Float64, checks=Check.in_range(-180.0, 180.0), nullable=False),
        "solar_elevation_deg": Column(
            pa.Float64, checks=Check.in_range(-90.0, 90.0), nullable=False
        ),
        "is_daylight": Column(pa.Bool, nullable=False),
        "high_sun": Column(pa.Bool, nullable=False),
        "quality_flag": Column(
            pa.String,
            checks=Check.isin([
                "VALID",
                "INCOMPLETE_INTERVALS",
                "MISSING_DATA",
                "SUSPECT_ZERO",
            ]),
            nullable=False,
        ),
    },
    strict=True,
    coerce=True,
)


# -----------------------------------------------------------------------------
# Validation and Quarantine Logic
# -----------------------------------------------------------------------------

def validate_and_quarantine_raw_generation(
    df: pd.DataFrame,
    timestamp_col: str = "measured_on",
    power_col: str = "meter_revenue_grade_ac_output_meter_149578",
    max_capacity_kw: float = 893.0,
    quarantine_output_path: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate raw 15-minute generation data and quarantine anomalous records.

    Rules enforced:
    - Duplicate timestamps are flagged (reason='duplicate_timestamp').
    - Values above nameplate DC capacity (893 kW) are quarantined (reason='above_893_kw').
    - Negative power values are quarantined (reason='negative_power').
    - Non-finite or missing values are quarantined (reason='non_finite').

    No silent clipping is performed. Quarantined records are isolated with an
    explanatory reason column and optionally written to parquet.

    Args:
        df: Raw generation DataFrame.
        timestamp_col: Name of the timestamp column.
        power_col: Name of the AC power column in kW.
        max_capacity_kw: Maximum plausible capacity in kW (default 893.0 kW DC nameplate).
        quarantine_output_path: Optional path to save quarantine.parquet.

    Returns:
        Tuple of (clean_df, quarantine_df).
    """
    df_work = df.copy()

    # Priority 1: Non-finite power
    is_non_finite = (
        df_work[power_col].isna()
        | np.isinf(pd.to_numeric(df_work[power_col], errors="coerce"))
    )

    # Priority 2: Negative power
    numeric_power = pd.to_numeric(df_work[power_col], errors="coerce")
    is_negative = (~is_non_finite) & (numeric_power < 0.0)

    # Priority 3: Exceeds DC nameplate capacity
    is_over_capacity = (~is_non_finite) & (numeric_power > max_capacity_kw)

    # Priority 4: Duplicate timestamps
    is_duplicate = df_work[timestamp_col].duplicated(keep="first")

    # Construct quarantine masks with priority
    reasons = pd.Series(index=df_work.index, dtype="object")
    reasons[is_non_finite] = "non_finite"
    reasons[is_negative] = "negative_power"
    reasons[is_over_capacity] = "above_893_kw"
    # Only label as duplicate if not already quarantined by value checks
    reasons[is_duplicate & reasons.isna()] = "duplicate_timestamp"

    quarantine_mask = reasons.notna()

    quarantine_df = pd.DataFrame({
        "measured_on": df_work.loc[quarantine_mask, timestamp_col].astype(str),
        "power_kw": numeric_power.loc[quarantine_mask],
        "reason": reasons.loc[quarantine_mask],
    })

    clean_df = df_work.loc[~quarantine_mask].copy()
    clean_df[power_col] = numeric_power.loc[~quarantine_mask]

    # Validate quarantine schema if any rows exist
    if not quarantine_df.empty:
        quarantine_df = quarantine_schema.validate(quarantine_df)

    if quarantine_output_path:
        out_path = Path(quarantine_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        quarantine_df.to_parquet(out_path, index=False)

    return clean_df, quarantine_df
