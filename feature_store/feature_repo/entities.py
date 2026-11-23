"""Feast entity definitions for SolarCast."""

from feast import Entity, ValueType

# Entity for PV generation site
site_entity = Entity(
    name="site",
    join_keys=["site_id"],
    value_type=ValueType.STRING,
    description="Solar photovoltaic generation facility (PVDAQ Site 2107, Arbuckle, CA)",
)
