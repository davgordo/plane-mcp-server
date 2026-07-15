"""Shared serialization helpers for MCP tool responses."""

from collections.abc import Iterable, Mapping
from typing import Any


def parse_requested_fields(fields: str | None) -> set[str] | None:
    """Parse a comma-separated sparse fieldset string."""
    if fields is None:
        return None

    requested_fields = {field.strip() for field in fields.split(",") if field.strip()}
    if not requested_fields:
        raise ValueError("fields must contain at least one field name")
    return requested_fields


def serialize_resource(resource: Any, fields: str | None = None) -> Any:
    """Serialize a resource, applying a sparse top-level fieldset when requested."""
    requested_fields = parse_requested_fields(fields)

    if hasattr(resource, "model_dump"):
        if requested_fields is None:
            return resource.model_dump()
        return resource.model_dump(include=requested_fields)

    if isinstance(resource, Mapping):
        if requested_fields is None:
            return dict(resource)
        return {field: resource[field] for field in requested_fields if field in resource}

    return resource


def serialize_resources(resources: Iterable[Any], fields: str | None = None) -> list[Any]:
    """Serialize a sequence of resources, applying sparse fieldsets item-by-item."""
    return [serialize_resource(resource, fields=fields) for resource in resources]
