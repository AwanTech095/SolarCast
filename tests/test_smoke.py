"""Smoke tests for Member 1 scaffolding, package structure, and configs."""

import importlib
import sys
import unittest
from pathlib import Path

# Ensure repository root is in sys.path when run directly or by test runners
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class TestMember1Scaffold(unittest.TestCase):
    """Smoke tests verifying Member 1 directory layout and package imports."""

    def test_src_imports(self) -> None:
        """Verify that all Member 1 src packages can be imported."""
        packages = ["src", "src.data", "src.features", "src.models"]
        for pkg in packages:
            mod = importlib.import_module(pkg)
            self.assertIsNotNone(mod)

    def test_member1_directories_exist(self) -> None:
        """Verify that expected Member 1 directories exist."""
        expected_dirs = [
            _REPO_ROOT / "configs",
            _REPO_ROOT / "data" / "raw",
            _REPO_ROOT / "data" / "processed",
            _REPO_ROOT / "src" / "data",
            _REPO_ROOT / "src" / "features",
            _REPO_ROOT / "src" / "models",
            _REPO_ROOT / "feature_store" / "feature_repo",
            _REPO_ROOT / "tests" / "fixtures",
            _REPO_ROOT / "docs",
            _REPO_ROOT / "evidence",
        ]
        for directory in expected_dirs:
            self.assertTrue(directory.exists(), f"Directory missing: {directory}")
            self.assertTrue(
                directory.is_dir(), f"Expected directory but found file: {directory}"
            )

    def test_env_example_contains_required_keys(self) -> None:
        """Verify that .env.example defines required MLflow keys."""
        env_example = _REPO_ROOT / ".env.example"
        self.assertTrue(env_example.exists(), ".env.example does not exist")

        content = env_example.read_text(encoding="utf-8")
        required_keys = [
            "MLFLOW_TRACKING_URI",
            "MLFLOW_TRACKING_USERNAME",
            "MLFLOW_TRACKING_PASSWORD",
        ]
        for key in required_keys:
            self.assertIn(key, content, f"Missing {key} in .env.example")

    def test_pyproject_ruff_config(self) -> None:
        """Verify that pyproject.toml contains ruff line-length 100."""
        pyproject = _REPO_ROOT / "pyproject.toml"
        self.assertTrue(pyproject.exists(), "pyproject.toml does not exist")

        content = pyproject.read_text(encoding="utf-8")
        self.assertIn("line-length = 100", content, "Missing line-length = 100 in pyproject.toml")


if __name__ == "__main__":
    unittest.main()
