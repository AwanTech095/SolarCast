"""Compile the SolarCast Kubeflow Pipelines v2 graph."""

import argparse
import re

from kfp import compiler, dsl, kubernetes

from orchestration.stages import TRACKING_URI


def compile_pipeline(image: str, output: str) -> None:
    """Compile with a fixed container digest and stage-specific secret references."""
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("Use a published worker image pinned by its sha256 digest.")

    @dsl.container_component
    def restore(snapshot: dsl.Output[dsl.Dataset]) -> dsl.ContainerSpec:
        return dsl.ContainerSpec(
            image=image, command=["python", "-m", "orchestration.stages", "restore"],
            args=["--output", snapshot.path, "--dvc", "/opt/dvc/bin/dvc"],
        )

    @dsl.container_component
    def train(snapshot: dsl.Input[dsl.Dataset], candidate: dsl.Output[dsl.Artifact]
              ) -> dsl.ContainerSpec:
        return dsl.ContainerSpec(
            image=image, command=["python", "-m", "orchestration.stages", "train"],
            args=["--inputs", snapshot.path, "--output", candidate.path],
        )

    @dsl.container_component
    def verify(candidate: dsl.Input[dsl.Artifact], gate: dsl.Output[dsl.Artifact],
               eligible: dsl.OutputPath(bool)) -> dsl.ContainerSpec:
        return dsl.ContainerSpec(
            image=image, command=["python", "-m", "orchestration.stages", "verify"],
            args=["--candidate", candidate.path, "--output", gate.path,
                  "--eligibility", eligible],
        )

    @dsl.pipeline(name="solarcast-historical-candidate")
    def pipeline() -> bool:
        restored = restore().set_caching_options(False)
        kubernetes.use_secret_as_env(
            restored, secret_name="solarcast-dvc",
            secret_key_to_env={
                "access-key-id": "AWS_ACCESS_KEY_ID", "secret-access-key": "AWS_SECRET_ACCESS_KEY",
            },
        )
        trained = train(snapshot=restored.outputs["snapshot"]).set_caching_options(False)
        trained.set_env_variable("MLFLOW_TRACKING_URI", TRACKING_URI)
        kubernetes.use_secret_as_env(
            trained, secret_name="solarcast-mlflow",
            secret_key_to_env={"username": "MLFLOW_TRACKING_USERNAME",
                               "token": "MLFLOW_TRACKING_PASSWORD"},
        )
        checked = verify(candidate=trained.outputs["candidate"]).set_caching_options(False)
        return checked.outputs["eligible"]

    compiler.Compiler().compile(pipeline, output)


def main() -> None:
    """Compile an importable KFP pipeline spec."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    compile_pipeline(args.image, args.output)


if __name__ == "__main__":
    main()
