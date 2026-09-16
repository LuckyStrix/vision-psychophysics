"""Tests for vpsych.data.migrations: the schema-version registry and migrate() framework."""

from __future__ import annotations

from typing import Any

import pytest

from vpsych.data import migrations


def test_migrate_identity_same_version() -> None:
    obj = {"a": 1}
    result = migrations.migrate(obj, "TestSummary", "1.0.0", "1.0.0")
    assert result == obj
    assert result is not obj  # shallow copy, not the same object


def test_migrate_unknown_path_raises() -> None:
    with pytest.raises(migrations.MigrationError):
        migrations.migrate({"a": 1}, "NoSuchSchema", "1.0.0", "2.0.0")


def test_register_and_apply_mock_migration() -> None:
    schema_name = "MockThing-" + str(id(object()))  # unique per test run

    @migrations.register_migration(schema_name, "1.0.0", "1.1.0")
    def _add_field(obj: dict[str, Any]) -> dict[str, Any]:
        new_obj = dict(obj)
        new_obj["new_field"] = "default"
        return new_obj

    old_doc = {"name": "widget", "value": 42}
    migrated = migrations.migrate(old_doc, schema_name, "1.0.0", "1.1.0")

    assert migrated == {"name": "widget", "value": 42, "new_field": "default"}
    # The original object is untouched.
    assert old_doc == {"name": "widget", "value": 42}


def test_register_duplicate_edge_raises() -> None:
    schema_name = "DupMock-" + str(id(object()))

    @migrations.register_migration(schema_name, "1.0.0", "1.1.0")
    def _first(obj: dict[str, Any]) -> dict[str, Any]:
        return obj

    with pytest.raises(ValueError, match="already registered"):

        @migrations.register_migration(schema_name, "1.0.0", "1.1.0")
        def _second(obj: dict[str, Any]) -> dict[str, Any]:
            return obj


def test_migrate_chains_multiple_registered_edges() -> None:
    schema_name = "ChainMock-" + str(id(object()))

    @migrations.register_migration(schema_name, "1.0.0", "1.1.0")
    def _step_one(obj: dict[str, Any]) -> dict[str, Any]:
        return {**obj, "step_one": True}

    @migrations.register_migration(schema_name, "1.1.0", "1.2.0")
    def _step_two(obj: dict[str, Any]) -> dict[str, Any]:
        return {**obj, "step_two": True}

    result = migrations.migrate({"original": True}, schema_name, "1.0.0", "1.2.0")
    assert result == {"original": True, "step_one": True, "step_two": True}


def test_known_schema_versions_includes_current() -> None:
    from vpsych.data.schemas import SCHEMA_VERSION

    assert SCHEMA_VERSION in migrations.KNOWN_SCHEMA_VERSIONS
