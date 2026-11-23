# Weather Data Quality Report: HRRR Forecast Features

**Dataset File:** `data/processed/hrrr_forecast_features.parquet`  
**Site ID:** `2107`  
**Evaluation Window:** 2018-07-12 00:00:00 UTC to 2023-11-09 23:00:00 UTC  
**Date of Audit:** 2026-11-23  

---

## 1. Summary Statistics

| Metric | Value |
|---|---|
| **Total Rows** | 23,532 |
| **First Issue Time (UTC)** | 2018-07-12 00:00:00+00:00 |
| **Last Issue Time (UTC)** | 2023-11-09 23:00:00+00:00 |
| **First Target Start (UTC)** | 2018-07-12 00:00:00+00:00 |
| **Last Target Start (UTC)** | 2023-11-09 23:00:00+00:00 |
| **Weather Available (`tmp_c`, `tcdc_pct` non-null)** | 22,570 (95.91%) |
| **Weather Missing (`tmp_c`, `tcdc_pct` NaN)** | 962 (4.09%) |
| **Layer Cloud Cover (`lcdc_pct`, `mcdc_pct`, `hcdc_pct`)** | 23,532 NaN (100.00% unextracted / NaN by design) |

---

## 2. Extraction Status Breakdown

| Extraction Status | Row Count | Percentage | Description |
|---|---|---|---|
| `SUCCESS:tmp,tcdc` | 22,131 | 94.05% | Nominal run found at lag = 2 h; both `tmp` and `tcdc` extracted successfully. |
| `FALLBACK_SUCCESS:tmp,tcdc` | 439 | 1.87% | Nominal run was missing; successfully fell back to an earlier available run (age 3–8 h). |
| `FAILED_NO_RUN:tmp,tcdc` | 961 | 4.08% | No valid HRRR run was found within the fallback window (up to 8 h age). |
| `FAILED_READ:tmp,tcdc` | 1 | 0.004% | A single run chunk failed during chunk retrieval (issue time `2020-07-03 02:00:00 UTC`). |
| **Total** | **23,532** | **100.00%** | |

### Weather Run Age Distribution (Available Weather Rows, $N = 22,570$)
- **2 hours (nominal):** 22,131 (98.05% of available runs)
- **3 hours (fallback):** 234 (1.04%)
- **4 hours (fallback):** 84 (0.37%)
- **5 hours (fallback):** 42 (0.19%)
- **6 hours (fallback):** 33 (0.15%)
- **7 hours (fallback):** 28 (0.12%)
- **8 hours (fallback):** 18 (0.08%)

---

## 3. Join Coverage Against Daylight Hours

Comparing `data/processed/hrrr_forecast_features.parquet` with the validated meter dataset `data/processed/hourly.parquet` across the evaluation period:
- **Total hourly rows in evaluation window:** 46,728
- **Daylight hourly rows (`is_daylight == True`):** 23,532
- **Daylight rows joined with HRRR features:** 23,532 / 23,532 (**100.00% join coverage**)
- **Daylight rows with weather data present:** 22,570 / 23,532 (**95.91%**)
- **Daylight rows with missing weather:** 962 / 23,532 (**4.09%**)

---

## 4. Operational Leakage and Availability Verification

For all 22,570 rows where forecast weather was retrieved:
1. **Availability constraint verification (`weather_available_at_utc <= issue_time_utc`):**
   - Result: **PASSED (100% compliant, 0 violations)**.
   - Every weather forecast ingested was strictly released and available at or before the forecast issue time $H$.
2. **Minimum run age verification (`run_age_hours >= 2`):**
   - Result: **PASSED (100% compliant, 0 violations)**.
   - The minimum run age is 2.0 hours, respecting the 2-hour NOAA HRRR ingest and post-processing latency SLA.

---

## 5. Missing Weather Analysis

### 5.1 Monthly Distribution of Missing Weather (Daylight Hours)

| Month | Daylight Hours | Available | Missing | Missing % |
|---|---|---|---|---|
| 2018-07 | 281 | 195 | 86 | 30.60% |
| 2018-08 | 421 | 417 | 4 | 0.95% |
| 2018-09 | 364 | 364 | 0 | 0.00% |
| 2018-10 | 344 | 324 | 20 | 5.81% |
| 2018-11 | 300 | 300 | 0 | 0.00% |
| 2018-12 | 302 | 302 | 0 | 0.00% |
| 2019-01 | 294 | 280 | 14 | 4.76% |
| 2019-02 | 302 | 302 | 0 | 0.00% |
| 2019-03 | 361 | 361 | 0 | 0.00% |
| 2019-04 | 397 | 343 | 54 | 13.60% |
| 2019-05 | 434 | 434 | 0 | 0.00% |
| 2019-06 | 437 | 418 | 19 | 4.35% |
| 2019-07 | 447 | 447 | 0 | 0.00% |
| 2019-08 | 422 | 402 | 20 | 4.74% |
| 2019-09 | 365 | 365 | 0 | 0.00% |
| 2019-10 | 345 | 326 | 19 | 5.51% |
| 2019-11 | 300 | 269 | 31 | 10.33% |
| 2019-12 | 302 | 302 | 0 | 0.00% |
| 2020-01 | 293 | 293 | 0 | 0.00% |
| 2020-02 | 313 | 290 | 23 | 7.35% |
| 2020-03 | 362 | 361 | 1 | 0.28% |
| 2020-04 | 399 | 399 | 0 | 0.00% |
| 2020-05 | 434 | 434 | 0 | 0.00% |
| 2020-06 | 437 | 365 | 72 | 16.48% |
| 2020-07 | 446 | 445 | 1 | 0.22% |
| 2020-08 | 421 | 407 | 14 | 3.33% |
| 2020-09 | 364 | 354 | 10 | 2.75% |
| 2020-10 | 343 | 314 | 29 | 8.45% |
| 2020-11 | 300 | 281 | 19 | 6.33% |
| 2020-12 | 301 | 301 | 0 | 0.00% |
| 2021-01 | 294 | 294 | 0 | 0.00% |
| 2021-02 | 303 | 303 | 0 | 0.00% |
| 2021-03 | 362 | 362 | 0 | 0.00% |
| 2021-04 | 398 | 223 | 175 | 43.97% |
| 2021-05 | 434 | 420 | 14 | 3.23% |
| 2021-06 | 437 | 437 | 0 | 0.00% |
| 2021-07 | 446 | 429 | 17 | 3.81% |
| 2021-08 | 421 | 421 | 0 | 0.00% |
| 2021-09 | 364 | 364 | 0 | 0.00% |
| 2021-10 | 343 | 309 | 34 | 9.91% |
| 2021-11 | 300 | 300 | 0 | 0.00% |
| 2021-12 | 302 | 302 | 0 | 0.00% |
| 2022-01 | 294 | 252 | 42 | 14.29% |
| 2022-02 | 303 | 293 | 10 | 3.30% |
| 2022-03 | 361 | 361 | 0 | 0.00% |
| 2022-04 | 397 | 397 | 0 | 0.00% |
| 2022-05 | 434 | 434 | 0 | 0.00% |
| 2022-06 | 437 | 437 | 0 | 0.00% |
| 2022-07 | 447 | 447 | 0 | 0.00% |
| 2022-08 | 421 | 415 | 6 | 1.43% |
| 2022-09 | 364 | 350 | 14 | 3.85% |
| 2022-10 | 344 | 344 | 0 | 0.00% |
| 2022-11 | 300 | 300 | 0 | 0.00% |
| 2022-12 | 302 | 302 | 0 | 0.00% |
| 2023-01 | 294 | 294 | 0 | 0.00% |
| 2023-02 | 302 | 291 | 11 | 3.64% |
| 2023-03 | 361 | 339 | 22 | 6.09% |
| 2023-04 | 397 | 356 | 41 | 10.33% |
| 2023-05 | 434 | 416 | 18 | 4.15% |
| 2023-06 | 437 | 437 | 0 | 0.00% |
| 2023-07 | 447 | 447 | 0 | 0.00% |
| 2023-08 | 421 | 320 | 101 | 23.99% |
| 2023-09 | 365 | 364 | 1 | 0.27% |
| 2023-10 | 345 | 336 | 9 | 2.61% |
| 2023-11 | 90 | 79 | 11 | 12.22% |

### 5.2 Contiguous Missing Outages ($\ge 10$ Daylight Hours)

The largest contiguous missing blocks in the daylight series are:
1. **2021-04-03 18:00 UTC to 2021-04-16 23:00 UTC:** 175 daylight hours missing (~13.2 calendar days).
2. **2023-08-16 15:00 UTC to 2023-08-23 22:00 UTC:** 101 daylight hours missing (~7.3 calendar days).
3. **2018-07-12 00:00 UTC to 2018-07-18 00:00 UTC:** 86 daylight hours missing (~6 calendar days, at dataset initiation).
4. **2020-06-25 16:00 UTC to 2020-06-30 03:00 UTC:** 72 daylight hours missing (~4.5 calendar days).
5. **2023-03-30 14:00 UTC to 2023-04-04 16:00 UTC:** 63 daylight hours missing (~5.1 calendar days).
6. **2019-04-18 14:00 UTC to 2019-04-22 02:00 UTC:** 54 daylight hours missing (~3.5 calendar days).
7. **2022-01-17 15:00 UTC to 2022-01-21 00:00 UTC:** 40 daylight hours missing (~3.4 calendar days).
8. **2021-10-05 15:00 UTC to 2021-10-08 00:00 UTC:** 34 daylight hours missing (~2.4 calendar days).
9. **2020-10-12 14:00 UTC to 2020-10-14 20:00 UTC:** 29 daylight hours missing (~2.3 calendar days).
10. **2020-02-04 15:00 UTC to 2020-02-06 17:00 UTC:** 23 daylight hours missing (~2.1 calendar days).
11. **2019-08-03 13:00 UTC to 2019-08-04 18:00 UTC:** 20 daylight hours missing.
12. **2018-10-28 15:00 UTC to 2018-10-30 00:00 UTC:** 20 daylight hours missing.
13. **2023-10-31 15:00 UTC to 2023-11-02 00:00 UTC:** 20 daylight hours missing.
14. **2020-11-17 19:00 UTC to 2020-11-19 17:00 UTC:** 19 daylight hours missing.
15. **2019-10-12 00:00 UTC to 2019-10-13 18:00 UTC:** 19 daylight hours missing.
16. **2019-06-28 20:00 UTC to 2019-06-29 23:00 UTC:** 19 daylight hours missing.
17. **2023-05-08 21:00 UTC to 2023-05-10 00:00 UTC:** 18 daylight hours missing.
18. **2019-11-16 17:00 UTC to 2019-11-18 00:00 UTC:** 18 daylight hours missing.
19. **2021-07-28 23:00 UTC to 2021-07-30 01:00 UTC:** 17 daylight hours missing.
20. **2021-05-21 13:00 UTC to 2021-05-22 02:00 UTC:** 14 daylight hours missing.
21. **2020-08-03 13:00 UTC to 2020-08-04 02:00 UTC:** 14 daylight hours missing.
22. **2022-09-14 00:00 UTC to 2022-09-15 01:00 UTC:** 14 daylight hours missing.
23. **2019-01-05 16:00 UTC to 2019-01-06 20:00 UTC:** 14 daylight hours missing.
24. **2023-02-23 15:00 UTC to 2023-02-24 01:00 UTC:** 11 daylight hours missing.
25. **2019-11-27 15:00 UTC to 2019-11-28 15:00 UTC:** 11 daylight hours missing.
26. **2020-09-07 16:00 UTC to 2020-09-08 01:00 UTC:** 10 daylight hours missing.
27. **2022-02-24 15:00 UTC to 2022-02-25 00:00 UTC:** 10 daylight hours missing.

### 5.3 Root Cause of Gaps
> [!NOTE]
> The exact root cause for missing runs in the public archive (`s3://hrrrzarr`) is **unknown**. There are no upstream incident logs or error manifests embedded in the Zarr store. Missing runs correspond to either missing model output chunks during the archive ingestion pipeline by the University of Utah / AWS Open Data program, or NOAA operational model outages at the time of run generation.

Because feature extraction gracefully tolerates missing weather (marking `extraction_status = FAILED_NO_RUN` and returning `NaN`), downstream modeling and serving pipelines must explicitly support `NaN` weather inputs (e.g., via median imputation in linear models, native NaN routing in tree ensembles, and the `weather_missing` binary indicator feature).

---

## 6. Value Range Validation and Upstream Fill Value Anomaly

### 6.1 Range Bounds
Physical and meteorological validation thresholds for HRRR forecast features:
- **Total Cloud Cover (`tcdc_pct`):** $[0.0, 100.0]\,\%$
- **2-Meter Air Temperature (`tmp_c`):** $[-40.0, 60.0]\,^\circ\text{C}$

### 6.2 Upstream Fill Value Anomaly (-10,000)
During data auditing, exactly one record was found with an out-of-range value:
- **Issue Time:** `2020-07-29 21:00:00+00:00` (target hour `21:00:00` to `22:00:00` UTC)
- **Extracted Values:** `tmp_c = 34.1` $^\circ\text{C}$, `tcdc_pct = -10000.0` %, `extraction_status = SUCCESS:tmp,tcdc`
- **Cause:** Upstream missing/fill value sentinel (`-10000`) embedded in the MesoWest HRRR Zarr array for entire atmosphere total cloud cover.

### 6.3 Handling in Feature Engineering Pipeline
The feature extraction code in `src/features/preprocessing.py` treats any `tcdc_pct` outside $[0, 100]$ or `tmp_c` outside $[-40, 60]$ as `NaN`:
1. The out-of-bounds `tcdc_pct = -10000.0` is converted to `NaN`.
2. The binary indicator `weather_missing` is set to `1`.
3. `clearsky_ghi_cloud_adj_wm2` is set to `NaN` (avoiding false clear-sky scaling from clipping negative values).
4. `weather_fallback_used` is forced to `0`.
5. Downstream modeling routes `NaN` via native tree branching (`HistGradientBoostingRegressor`) or median imputation (`Ridge`).

After range filtering, the effective usable weather count across the evaluation period is **22,569 / 23,532 (95.91%)** daylight hours. This behavior is covered by unit test `test_weather_valid_range_filtering` in `tests/test_features.py`.
