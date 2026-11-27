"""Airflow 3 DAG for manually submitting and tracking a Kubeflow candidate run."""

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from airflow.sdk import dag, get_current_context, task


@dag(
    dag_id="solarcast_candidate",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule=None,
    catchup=False,
    max_active_runs=1,
)
def solarcast_candidate() -> None:
    """Validate deployment settings, submit one run and wait for its terminal state."""

    @task
    def check_settings() -> dict[str, str]:
        keys = ("SOLARCAST_KFP_HOST", "SOLARCAST_KFP_NAMESPACE", "SOLARCAST_PIPELINE_PACKAGE")
        settings = {key: os.environ[key] for key in keys}
        if not settings["SOLARCAST_KFP_HOST"].startswith(("http://", "https://")):
            raise ValueError("SOLARCAST_KFP_HOST must be a URL.")
        if not Path(settings["SOLARCAST_PIPELINE_PACKAGE"]).is_file():
            raise FileNotFoundError("Mount the compiled pipeline package on the Airflow worker.")
        return settings

    @task(retries=0)
    def submit(settings: dict[str, str]) -> str:
        from kfp import Client

        client = Client(
            host=settings["SOLARCAST_KFP_HOST"], namespace=settings["SOLARCAST_KFP_NAMESPACE"],
            existing_token=os.environ.get("SOLARCAST_KFP_TOKEN"),
        )
        name = f"solarcast-{get_current_context()['run_id']}"
        run = client.create_run_from_pipeline_package(
            settings["SOLARCAST_PIPELINE_PACKAGE"], arguments={}, run_name=name,
            experiment_name="SolarCast", namespace=settings["SOLARCAST_KFP_NAMESPACE"],
        )
        print(f"Kubeflow run ID: {run.run_id}")
        return run.run_id

    @task(execution_timeout=timedelta(hours=3))
    def wait_for_completion(settings: dict[str, str], run_id: str) -> None:
        from kfp import Client

        client = Client(
            host=settings["SOLARCAST_KFP_HOST"], namespace=settings["SOLARCAST_KFP_NAMESPACE"],
            existing_token=os.environ.get("SOLARCAST_KFP_TOKEN"),
        )
        run = client.wait_for_run_completion(run_id, timeout=10_200, sleep_duration=30)
        if run.state != "SUCCEEDED":
            raise RuntimeError(f"Kubeflow run {run_id} ended in {run.state}")
        print(f"See Kubeflow run {run_id} for the eligibility output and gate evidence.")

    settings = check_settings()
    wait_for_completion(settings, submit(settings))


candidate_dag = solarcast_candidate()
