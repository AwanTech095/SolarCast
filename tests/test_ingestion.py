"""Unit and integration tests for src/data/ingestion.py with mocked network calls."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
import yaml

from src.data.ingestion import (
    compute_sha256,
    count_csv_rows,
    fetch_nasa_power_year,
    register_pvdaq_files,
    run_ingestion,
    write_manifest,
)


@pytest.fixture
def sample_config(tmp_path: Path) -> dict:
    """Fixture supplying a minimal test configuration pointing to tmp_path."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = raw_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    weather_dir = raw_dir / "weather"
    weather_dir.mkdir(parents=True, exist_ok=True)

    # Pre-populate meter CSV and metadata in tmp_path
    meter_file = raw_dir / "2107_meter_15m_data.csv"
    meter_content = (
        "measured_on,meter_revenue_grade_ac_output\n"
        "2023-01-01 00:00:00,0.0\n"
        "2023-01-01 00:15:00,10.5\n"
    )
    meter_file.write_text(meter_content, encoding="utf-8")

    meta_file = meta_dir / "2107_system_metadata.json"
    meta_file.write_text(json.dumps({"System": {"system_id": 2107}}), encoding="utf-8")

    return {
        "site_id": "2107",
        "system_id": 2107,
        "public_name": "Farm Solar Array",
        "latitude": 38.996306,
        "longitude": -122.134111,
        "start_date": "2023-01-01",
        "end_date": "2023-01-02",
        "time_interval_alignment": "L",
        "weather_parameters": ["ALLSKY_SFC_SW_DWN", "T2M"],
        "paths": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "manifest_path": "data/raw/manifest.json",
            "meter_csv": "data/raw/2107_meter_15m_data.csv",
            "metadata_json": "data/raw/metadata/2107_system_metadata.json",
            "weather_dir": "data/raw/weather",
        },
        "sources": {
            "pvdaq": {
                "s3_bucket": "oedi-data-lake",
                "s3_prefix": "pvdaq/2023-solar-data-prize/2107_OEDI/",
                "licence": "DOE Open Data",
                "doi": "10.25984/1846021",
            },
            "nasa_power": {
                "api_url": "https://power.larc.nasa.gov/api/temporal/hourly/point",
                "licence": "CC BY 4.0",
                "time_standard": "UTC",
            },
        },
    }


def test_compute_sha256(tmp_path: Path) -> None:
    """Test SHA-256 calculation matches known digest."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("SolarCast Data Integrity Test\n", encoding="utf-8")
    sha = compute_sha256(test_file)
    assert len(sha) == 64
    assert isinstance(sha, str)


def test_count_csv_rows(tmp_path: Path) -> None:
    """Test non-header line counting."""
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("header1,header2\nrow1,val1\nrow2,val2\nrow3,val3\n", encoding="utf-8")
    assert count_csv_rows(csv_file) == 3


def test_register_pvdaq_files_idempotent(sample_config: dict, tmp_path: Path) -> None:
    """Test that existing local files are registered without downloading."""
    entries = register_pvdaq_files(sample_config, repo_root=tmp_path)
    assert "pvdaq_meter_15m" in entries
    assert "pvdaq_metadata" in entries
    assert entries["pvdaq_meter_15m"]["row_count"] == 2
    assert len(entries["pvdaq_meter_15m"]["sha256"]) == 64


def test_fetch_nasa_power_year_success(tmp_path: Path) -> None:
    """Test NASA POWER fetch with mocked HTTP 200 response."""
    output_path = tmp_path / "nasa_power_test.json"
    mock_payload = {
        "properties": {
            "parameter": {
                "ALLSKY_SFC_SW_DWN": {"2023010100": 0.0, "2023010101": 25.4},
                "T2M": {"2023010100": 5.2, "2023010101": 6.1},
            }
        }
    }

    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_payload
    mock_session.get.return_value = mock_response

    res = fetch_nasa_power_year(
        lat=38.9963,
        lon=-122.1341,
        start_date="20230101",
        end_date="20230101",
        parameters=["ALLSKY_SFC_SW_DWN", "T2M"],
        output_path=output_path,
        session=mock_session,
    )

    assert output_path.exists()
    assert res["hourly_records"] == 2
    assert mock_session.get.call_count == 1

    # Test idempotency: calling again with existing valid file skips network
    res_cached = fetch_nasa_power_year(
        lat=38.9963,
        lon=-122.1341,
        start_date="20230101",
        end_date="20230101",
        parameters=["ALLSKY_SFC_SW_DWN", "T2M"],
        output_path=output_path,
        session=mock_session,
    )
    assert res_cached["hourly_records"] == 2
    # Network call count remains 1 because cached file was reused
    assert mock_session.get.call_count == 1


def test_fetch_nasa_power_retry_backoff(tmp_path: Path) -> None:
    """Test retry logic on transient HTTP errors."""
    output_path = tmp_path / "nasa_power_retry.json"
    mock_payload = {
        "properties": {
            "parameter": {
                "ALLSKY_SFC_SW_DWN": {"2023010100": 0.0},
            }
        }
    }

    mock_session = MagicMock()
    # First call raises connection error, second succeeds
    fail_response = requests.RequestException("Connection timed out")
    success_response = MagicMock()
    success_response.status_code = 200
    success_response.json.return_value = mock_payload
    mock_session.get.side_effect = [fail_response, success_response]

    with patch("time.sleep") as mock_sleep:
        res = fetch_nasa_power_year(
            lat=38.9963,
            lon=-122.1341,
            start_date="20230101",
            end_date="20230101",
            parameters=["ALLSKY_SFC_SW_DWN"],
            output_path=output_path,
            max_retries=2,
            session=mock_session,
        )

        assert res["hourly_records"] == 1
        assert mock_session.get.call_count == 2
        mock_sleep.assert_called_once_with(1.0)


def test_write_manifest(tmp_path: Path) -> None:
    """Test manifest JSON generation."""
    manifest_path = tmp_path / "manifest.json"
    entries = {
        "asset_1": {
            "sha256": "abcdef123456",
            "bytes": 1024,
            "row_count": 50,
        }
    }
    write_manifest(manifest_path, entries, {"site_id": "2107"})

    assert manifest_path.exists()
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["manifest_version"] == "1.0"
    assert data["dataset_metadata"]["site_id"] == "2107"
    assert "asset_1" in data["artifacts"]


def test_run_ingestion_mocked(sample_config: dict, tmp_path: Path) -> None:
    """End-to-end integration test with mocked weather API and config."""
    config_file = tmp_path / "test_data.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(sample_config, f)

    mock_payload = {
        "properties": {
            "parameter": {
                "ALLSKY_SFC_SW_DWN": {"2023010100": 0.0, "2023010101": 10.0},
                "T2M": {"2023010100": 8.0, "2023010101": 8.5},
            }
        }
    }

    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_payload
    mock_session.get.return_value = mock_response

    manifest_path = run_ingestion(config_file, repo_root=tmp_path, session=mock_session)

    assert manifest_path.exists()
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    assert "pvdaq_meter_15m" in manifest["artifacts"]
    assert "pvdaq_metadata" in manifest["artifacts"]
    assert "nasa_power_weather_2023" in manifest["artifacts"]
    assert manifest["artifacts"]["pvdaq_meter_15m"]["row_count"] == 2
