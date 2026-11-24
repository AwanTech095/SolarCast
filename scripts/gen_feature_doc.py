"""Generate Markdown feature table documentation from configs/feature_schema.json.

Usage:
    python scripts/gen_feature_doc.py [--set B] [--output-file path]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SCHEMA_PATH = Path("configs/feature_schema.json")


def generate_feature_markdown_table(
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
    feature_set: str = "B",
) -> str:
    """Generate Markdown table of features from feature_schema.json."""
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)

    if "feature_sets" in schema and feature_set in schema["feature_sets"]:
        features = schema["feature_sets"][feature_set]["features"]
    elif "features" in schema:
        features = schema["features"]
    else:
        raise KeyError(f"Feature set '{feature_set}' not found in {schema_path}")

    lines: list[str] = [
        f"### Feature Schema v{schema.get('schema_version', '1.1.0')} (Feature Set {feature_set}, {len(features)} Features)",
        "",
        "| # | Name | Data Type | Units | Description |",
        "|---|---|:---:|:---:|---|",
    ]

    for i, feat in enumerate(features, 1):
        name = f"`{feat['name']}`"
        dtype = f"`{feat['dtype']}`"
        units = f"`{feat['units']}`"
        desc = feat["description"].replace("|", "\\|")
        lines.append(f"| {i} | {name} | {dtype} | {units} | {desc} |")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate feature table documentation")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH), help="Path to feature_schema.json")
    parser.add_argument("--set", default="B", help="Feature set to generate table for (A or B)")
    parser.add_argument("--output", default=None, help="Optional output path")
    args = parser.parse_args()

    table_md = generate_feature_markdown_table(args.schema, feature_set=args.set)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(table_md)
        print(f"Generated feature documentation table -> {out_path}")
    else:
        print(table_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
