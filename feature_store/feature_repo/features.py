"""Feature definitions and schema mapping for SolarCast Feature Set B."""

from __future__ import annotations

import json
from pathlib import Path

from feast import Field
from feast.types import Float64, Int64

SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "feature_schema.json"


def get_feature_fields() -> list[Field]:
    """Load ordered Feature Set B field definitions from configs/feature_schema.json."""
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)

    fields: list[Field] = []
    for item in schema["feature_sets"]["B"]["features"]:
        dtype = Int64 if item["dtype"] == "int64" else Float64
        tags = {"units": item.get("units", "dimensionless")}
        fields.append(
            Field(
                name=item["name"],
                dtype=dtype,
                description=item.get("description", ""),
                tags=tags,
            )
        )
    return fields
