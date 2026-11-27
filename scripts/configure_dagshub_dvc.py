"""Store a DagsHub token in the Git-ignored DVC configuration."""

import warnings
from getpass import GetPassWarning, getpass
from pathlib import Path

from dvc.config import Config


def main() -> None:
    """Prompt without echo and preserve any other local remote settings."""
    config = Config(str(Path(__file__).resolve().parents[1] / ".dvc"))
    if "team" not in config.get("remote", {}):
        raise SystemExit("The DVC remote 'team' is missing.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", GetPassWarning)
        token = getpass("DagsHub access token (input hidden): ").strip()
    if not token:
        raise SystemExit("No token entered.")
    with config.edit(level="local") as local:
        remote = local.setdefault("remote", {}).setdefault("team", {})
        remote["access_key_id"] = token
        remote["secret_access_key"] = token
    print("Saved team remote credentials in .dvc/config.local.")


if __name__ == "__main__":
    main()
