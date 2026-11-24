"""Artifact Packet Verifier and Installer for SolarCast.

Verifies the integrity of every file in the SolarCast artifact packet against
CHECKSUMS.sha256. Optionally installs files into a destination repository checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("verify_artifact_packet")


def sha256_of_file(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_packet(packet_dir: Path) -> tuple[bool, list[str]]:
    """Verify all files listed in CHECKSUMS.sha256.

    Args:
        packet_dir: Path to the artifact packet directory.

    Returns:
        (success, error_messages)
    """
    checksums_file = packet_dir / "CHECKSUMS.sha256"
    if not checksums_file.exists():
        return False, [f"Checksums file not found: {checksums_file}"]

    errors: list[str] = []
    lines = checksums_file.read_text(encoding="utf-8").splitlines()

    verified_count = 0
    print("\n" + "=" * 80)
    print(f"VERIFYING SOLARCAST ARTIFACT PACKET: {packet_dir}")
    print("=" * 80)

    for line_idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            errors.append(f"Line {line_idx}: Invalid line format: '{line}'")
            continue

        expected_sha, rel_path_str = parts[0].strip(), parts[1].strip()
        file_path = packet_dir / rel_path_str

        if not file_path.exists():
            msg = f"MISSING FILE: {rel_path_str}"
            logger.error("  [FAIL] %s", msg)
            errors.append(msg)
            continue

        actual_sha = sha256_of_file(file_path)
        if actual_sha.lower() != expected_sha.lower():
            msg = (
                f"CORRUPTION DETECTED: {rel_path_str}\n"
                f"    Expected SHA-256: {expected_sha}\n"
                f"    Actual SHA-256:   {actual_sha}"
            )
            logger.error("  [FAIL] %s", msg)
            errors.append(msg)
        else:
            print(f"  [OK]   {rel_path_str}")
            verified_count += 1

    print("=" * 80)
    print(f"Verification summary: {verified_count} passed, {len(errors)} failed.")
    print("=" * 80)

    return len(errors) == 0, errors


def install_packet(packet_dir: Path, repo_dir: Path) -> None:
    """Install verified files from the packet into a repository checkout.

    Args:
        packet_dir: Path to the verified packet directory.
        repo_dir: Path to target repository root.
    """
    checksums_file = packet_dir / "CHECKSUMS.sha256"
    lines = checksums_file.read_text(encoding="utf-8").splitlines()

    installed_count = 0
    print("\n" + "=" * 80)
    print(f"INSTALLING ARTIFACT PACKET INTO: {repo_dir}")
    print("=" * 80)

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        _, rel_path_str = line.split(maxsplit=1)
        src_path = packet_dir / rel_path_str
        dest_path = repo_dir / rel_path_str

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)
        print(f"  [INSTALLED] {rel_path_str}")
        installed_count += 1

    print("=" * 80)
    logger.info("Successfully installed %d files to %s", installed_count, repo_dir)


def main() -> None:
    """Main CLI entrypoint for packet verification and installation."""
    parser = argparse.ArgumentParser(
        description="Verify and install SolarCast artifact packet for Member 2"
    )
    parser.add_argument(
        "packet_dir",
        type=Path,
        help="Path to artifact packet directory (e.g. D:/SolarCast_packet)",
    )
    parser.add_argument(
        "--install",
        type=Path,
        default=None,
        metavar="REPO_DIR",
        help="Optional path to repo checkout to copy data/ and models/ into",
    )
    args = parser.parse_args()

    packet_dir = args.packet_dir.resolve()
    if not packet_dir.exists() or not packet_dir.is_dir():
        logger.error("Packet directory does not exist: %s", packet_dir)
        sys.exit(1)

    success, errors = verify_packet(packet_dir)
    if not success:
        logger.error("Packet verification FAILED with %d error(s):", len(errors))
        for err in errors:
            print(f"  ERROR: {err}", file=sys.stderr)
        sys.exit(1)

    logger.info("Packet verification PASSED: all files matched CHECKSUMS.sha256")

    if args.install:
        repo_dir = args.install.resolve()
        if not repo_dir.exists():
            logger.error("Target repository directory does not exist: %s", repo_dir)
            sys.exit(1)
        install_packet(packet_dir, repo_dir)


if __name__ == "__main__":
    main()
