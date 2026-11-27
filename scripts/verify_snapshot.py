"""Compare restored SolarCast artifacts with the team's SHA-256 receipt."""

import argparse
import hashlib
import json
from pathlib import Path


def verify_snapshot(root: Path) -> int:
    """Verify every receipt entry and return the number of checked files."""
    receipt = root / "reports/packet_receipt_20261126.json"
    entries = json.loads(receipt.read_text(encoding="utf-8"))["verified_files"]
    if not entries:
        raise SystemExit("Artifact receipt contains no files.")
    for entry in entries:
        path = root / entry["path"]
        if not path.is_file():
            raise SystemExit(f"Missing snapshot file: {entry['path']}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise SystemExit(f"SHA-256 mismatch: {entry['path']}")
    return len(entries)


def main() -> None:
    """Verify the artifact snapshot at the specified checkout root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    count = verify_snapshot(parser.parse_args().root)
    print(f"Verified {count} files against the artifact receipt.")


if __name__ == "__main__":
    main()
