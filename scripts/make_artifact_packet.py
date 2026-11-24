"""Artifact Packet Generator for SolarCast (Member 1 to Member 2 Handoff).

Creates a standalone, clean artifact packet outside the repository containing
all raw and processed datasets, model bundles, manifests, and evaluation audits
required by Member 2 to reproduce pipelines, configure DVC, and run CI/CD.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("make_artifact_packet")

# Processed paths specifically excluded per specification
PROCESSED_EXCLUSIONS = {
    "hrrr_parts": (
        "Intermediate monthly chunk files prior to consolidation; "
        "superseded by hrrr_forecast_features.parquet"
    ),
    "june_week_extracted.parquet": (
        "Exploratory sample from early M1.1b feasibility check, "
        "not used in documented pipeline"
    ),
    "mixed_week_extracted.parquet": (
        "Exploratory sample from early M1.1b feasibility check, "
        "not used in documented pipeline"
    ),
    ".gitkeep": "Git placeholder file",
}

# Convenience files to include
CONVENIENCE_FILES = [
    Path("reports/evaluation.json"),
    Path("reports/final_test_audit.json"),
    Path("configs/data_manifest.json"),
]

MODEL_FILES = [
    Path("models/solarcast_f1_bundle.joblib"),
]


def collect_packet_files(repo_dir: Path) -> tuple[list[Path], dict[str, str]]:
    """Identify files to include in the artifact packet and track excluded paths.

    Args:
        repo_dir: Root directory of the SolarCast repository.

    Returns:
        A tuple of (included_files, exclusions_dict).
    """
    included: list[Path] = []
    exclusions: dict[str, str] = {}

    # 1. Everything under data/raw/ (except .gitkeep)
    raw_dir = repo_dir / "data" / "raw"
    if raw_dir.exists():
        for path in sorted(raw_dir.rglob("*")):
            if path.is_file():
                if path.name == ".gitkeep":
                    exclusions[str(path.relative_to(repo_dir)).replace("\\", "/")] = (
                        "Git placeholder file"
                    )
                else:
                    included.append(path)

    # 2. Everything under data/processed/ with specific exclusions
    processed_dir = repo_dir / "data" / "processed"
    if processed_dir.exists():
        for path in sorted(processed_dir.rglob("*")):
            if not path.is_file():
                continue
            rel_proc = path.relative_to(processed_dir).as_posix()
            parts = rel_proc.split("/")
            top_part = parts[0]

            if top_part in PROCESSED_EXCLUSIONS:
                rel_repo = str(path.relative_to(repo_dir)).replace("\\", "/")
                exclusions[rel_repo] = PROCESSED_EXCLUSIONS[top_part]
                continue
            if rel_proc in PROCESSED_EXCLUSIONS:
                rel_repo = str(path.relative_to(repo_dir)).replace("\\", "/")
                exclusions[rel_repo] = PROCESSED_EXCLUSIONS[rel_proc]
                continue
            if path.name == ".gitkeep":
                exclusions[str(path.relative_to(repo_dir)).replace("\\", "/")] = (
                    "Git placeholder file"
                )
                continue

            included.append(path)

    # 3. Model bundle
    for model_path in MODEL_FILES:
        full_path = repo_dir / model_path
        if full_path.exists():
            included.append(full_path)
        else:
            logger.warning("Expected model file not found: %s", full_path)

    # 4. Reports and config convenience files
    for conv_path in CONVENIENCE_FILES:
        full_path = repo_dir / conv_path
        if full_path.exists():
            included.append(full_path)
        else:
            logger.warning("Expected convenience file not found: %s", full_path)

    return sorted(included), exclusions


def copy_packet_files(repo_dir: Path, out_dir: Path, files: list[Path]) -> None:
    """Copy collected files to the destination directory preserving relative paths.

    Args:
        repo_dir: Root directory of the source repository.
        out_dir: Destination root directory for the packet.
        files: List of file paths to copy.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for src_path in files:
        rel_path = src_path.relative_to(repo_dir)
        dest_path = out_dir / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)


def main() -> None:
    """Main CLI entrypoint for Step 1 packet generation."""
    parser = argparse.ArgumentParser(description="Create SolarCast artifact packet for Member 2")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("D:/SolarCast_packet"),
        help="Destination directory for artifact packet (default: D:/SolarCast_packet)",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="SolarCast repository root directory",
    )
    args = parser.parse_args()

    repo_dir: Path = args.repo_dir.resolve()
    out_dir: Path = args.out.resolve()

    logger.info("Source repository: %s", repo_dir)
    logger.info("Destination packet: %s", out_dir)

    # Safety checks
    if str(out_dir).startswith(str(repo_dir)):
        logger.error("Destination must be OUTSIDE the repository directory: %s", out_dir)
        sys.exit(1)

    included_files, exclusions = collect_packet_files(repo_dir)

    print("\n" + "=" * 80)
    print("SOLARCAST ARTIFACT PACKET INCLUSION REPORT")
    print("=" * 80)
    total_bytes = 0
    for p in included_files:
        rel = p.relative_to(repo_dir).as_posix()
        sz = p.stat().st_size
        total_bytes += sz
        print(f"  [INCLUDE] {rel:<55} ({sz:>10,d} bytes)")

    print("\n" + "=" * 80)
    print("SOLARCAST ARTIFACT PACKET EXCLUSION REPORT (WITH REASONS)")
    print("=" * 80)
    for path_str, reason in sorted(exclusions.items()):
        print(f"  [EXCLUDE] {path_str:<55} -> Reason: {reason}")

    print("\n" + "=" * 80)
    print(f"Total files to copy: {len(included_files)}")
    print(f"Total size: {total_bytes:,} bytes ({total_bytes / (1024 * 1024):.2f} MB)")
    print("=" * 80)

    logger.info("Copying files to %s ...", out_dir)
    copy_packet_files(repo_dir, out_dir, included_files)
    logger.info("Step 1 complete: %d files successfully copied to %s", len(included_files), out_dir)


if __name__ == "__main__":
    main()
