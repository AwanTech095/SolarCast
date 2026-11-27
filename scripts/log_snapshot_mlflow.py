"""Log and verify the historical SolarCast snapshot in a shared MLflow run."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from scripts.verify_snapshot import verify_snapshot
from src.models.lineage import get_lineage_tags

EXPERIMENT = "solarcast-snapshot-verification"
BUNDLE = "models/solarcast_f1_bundle.joblib"
RECEIPT = "reports/packet_receipt_20261126.json"
EVALUATION = "reports/evaluation.json"
EVIDENCE = (
    RECEIPT, EVALUATION, "reports/final_test_audit.json",
    "data/raw.dvc", "data/processed.dvc", f"{BUNDLE}.dvc",
)


def main() -> None:
    """Verify the local packet and round-trip its evidence through MLflow."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri.startswith("https://"):
        raise SystemExit("Set MLFLOW_TRACKING_URI to the team's HTTPS tracking endpoint.")
    root = Path(__file__).resolve().parents[1]
    count = verify_snapshot(root)
    receipt = json.loads((root / RECEIPT).read_text(encoding="utf-8"))
    evaluation = json.loads((root / EVALUATION).read_text(encoding="utf-8"))
    tags = get_lineage_tags(root)
    tags.update({
        "run_purpose": "historical_snapshot_verification",
        "source_model_run_id": receipt["bundle_checks"]["mlflow_run_id"],
        "model_bundle_dvc_path": f"{BUNDLE}.dvc",
    })
    if tags["git_commit"] == "unknown_commit":
        raise SystemExit("Could not identify the source Git revision.")
    metrics = {
        f"historical_cv_{key}": float(value)
        for key, value in evaluation["candidate_metrics"].items()
    }
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="verified-historical-snapshot") as run:
        mlflow.set_tags(tags)
        mlflow.log_param("verified_receipt_files", count)
        mlflow.log_metrics(metrics)
        for relative_path in EVIDENCE:
            path = Path(relative_path)
            mlflow.log_artifact(str(root / path), artifact_path=f"evidence/{path.parent.as_posix()}")

        client = MlflowClient()
        stored = client.get_run(run.info.run_id).data
        if any(stored.tags.get(key) != value for key, value in tags.items()):
            raise RuntimeError("MLflow lineage tags did not round-trip.")
        if any(stored.metrics.get(key) != value for key, value in metrics.items()):
            raise RuntimeError("MLflow historical metrics did not round-trip.")
        with tempfile.TemporaryDirectory() as temp:
            for relative_path in EVIDENCE:
                downloaded = client.download_artifacts(
                    run.info.run_id, f"evidence/{relative_path}", dst_path=temp,
                )
                expected = hashlib.sha256((root / relative_path).read_bytes()).digest()
                if hashlib.sha256(Path(downloaded).read_bytes()).digest() != expected:
                    raise RuntimeError(f"MLflow evidence differs from source: {relative_path}")
        run_id = run.info.run_id
    if client.get_run(run_id).info.status != "FINISHED":
        raise RuntimeError("MLflow did not record the run as finished.")
    print(f"Verified shared MLflow run: {run_id}")


if __name__ == "__main__":
    main()
