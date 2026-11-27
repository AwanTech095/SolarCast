"""Container entrypoints for the SolarCast historical candidate pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from scripts.verify_snapshot import verify_snapshot

TARGETS = ("data/raw.dvc", "data/processed.dvc", "models/solarcast_f1_bundle.joblib.dvc")
INPUTS = ("hourly.parquet", "quarter_hourly.parquet", "hrrr_forecast_features.parquet")
TRACKING_URI = "https://dagshub.com/AwanTech095/SolarCast.mlflow"


def restore_snapshot(output: Path, dvc: str = "dvc") -> None:
    """Restore and verify the packet before exporting training inputs."""
    subprocess.run([dvc, "pull", "-r", "team", *TARGETS], check=True)
    verify_snapshot(Path.cwd())
    output.mkdir(parents=True, exist_ok=True)
    for name in INPUTS:
        shutil.copyfile(Path("data/processed") / name, output / name)


def train_candidate(inputs: Path, output: Path) -> None:
    """Train on the snapshot and package the newly produced candidate directory."""
    from src.models.train import run_training_pipeline

    if os.environ.get("MLFLOW_TRACKING_URI") != TRACKING_URI:
        raise ValueError("Set MLFLOW_TRACKING_URI to the SolarCast shared endpoint.")
    if not all((inputs / name).is_file() for name in INPUTS):
        raise FileNotFoundError("The restored training input artifact is incomplete.")
    output.mkdir(parents=True, exist_ok=True)
    result = run_training_pipeline(
        hourly_path=inputs / INPUTS[0],
        quarter_hourly_path=inputs / INPUTS[1],
        weather_path=inputs / INPUTS[2],
        models_dir=output / "models",
        reports_dir=output / "reports",
        register_candidate=False,
    )
    manifest = Path(result["candidate_manifest_path"]).resolve()
    relative = manifest.relative_to(output.resolve())
    (output / "candidate_path.txt").write_text(relative.as_posix(), encoding="utf-8")


def verify_candidate(candidate: Path, output: Path, eligibility: Path) -> None:
    """Re-evaluate exact candidate files and publish a typed gate result."""
    from src.models.candidates import evaluate_candidate_artifact

    root = candidate.resolve()
    manifest = (root / (root / "candidate_path.txt").read_text(encoding="utf-8").strip()).resolve()
    manifest.relative_to(root)
    result = evaluate_candidate_artifact(manifest)
    eligible = result["eligible"] is True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    eligibility.parent.mkdir(parents=True, exist_ok=True)
    eligibility.write_text(json.dumps(eligible), encoding="utf-8")
    print(f"Candidate gate: {'eligible' if eligible else 'rejected'}")


def main() -> None:
    """Execute one pipeline stage with explicit artifact paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    restore = stages.add_parser("restore")
    restore.add_argument("--output", type=Path, required=True)
    restore.add_argument("--dvc", default="dvc")
    train = stages.add_parser("train")
    train.add_argument("--inputs", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    verify = stages.add_parser("verify")
    verify.add_argument("--candidate", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--eligibility", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "restore":
        restore_snapshot(args.output, args.dvc)
    elif args.stage == "train":
        train_candidate(args.inputs, args.output)
    else:
        verify_candidate(args.candidate, args.output, args.eligibility)


if __name__ == "__main__":
    main()
