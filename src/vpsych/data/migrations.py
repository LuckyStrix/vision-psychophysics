"""Schema-version registry and migration framework for on-disk JSON documents.

Every JSON file vpsych writes is tagged, at the dataset level, with the
`vpsych.data.schemas.SCHEMA_VERSION` it was written under (see
`dataset_description.json`). When that version is bumped for a breaking
change, old datasets must still be readable: a migration is a pure function
`dict -> dict` registered for one `(schema_name, from_version, to_version)`
edge, and `migrate` walks a chain of registered edges (via breadth-first
search) to get from an arbitrary `from_version` to `to_version`, so a
multi-step upgrade path doesn't need every pairwise migration registered
explicitly.

`schema_name` is a free-text label identifying *which* document shape is
being migrated (e.g. `"SessionInfo"`, `"TestSummary"`, `"Participant"`,
`"DatasetDescription"`) -- migrations for different document shapes are
independent of one another.

As of `SCHEMA_VERSION = "1.0.0"` there is exactly one version, so `migrate`
with `from_version == to_version == "1.0.0"` is always the identity
transform; this module exists so that changes when the schema does version
have somewhere to register. See `tests/data/test_migrations.py` for a mock
migration demonstrating the framework end to end.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]

#: Every schema version vpsych has ever written to disk, oldest first.
#: Extend this whenever `vpsych.data.schemas.SCHEMA_VERSION` is bumped.
KNOWN_SCHEMA_VERSIONS: list[str] = ["1.0.0"]

_MIGRATIONS: dict[str, dict[tuple[str, str], MigrationFn]] = {}


class MigrationError(ValueError):
    """Raised when no migration path exists between two schema versions."""


def register_migration(
    schema_name: str, from_version: str, to_version: str
) -> Callable[[MigrationFn], MigrationFn]:
    """Decorator that registers a migration function for one version edge.

    Args:
        schema_name: Which document shape this migrates (e.g.
            `"TestSummary"`).
        from_version: The schema version the input `dict` is shaped for.
        to_version: The schema version the returned `dict` is shaped for.

    Returns:
        A decorator that registers the wrapped function and returns it
        unchanged.

    Raises:
        ValueError: If an edge is already registered for
            `(schema_name, from_version, to_version)`.
    """

    def decorator(fn: MigrationFn) -> MigrationFn:
        edges = _MIGRATIONS.setdefault(schema_name, {})
        key = (from_version, to_version)
        if key in edges:
            raise ValueError(
                f"A migration for {schema_name!r} {from_version!r} -> {to_version!r} "
                "is already registered."
            )
        edges[key] = fn
        return fn

    return decorator


def _find_path(schema_name: str, from_version: str, to_version: str) -> list[MigrationFn]:
    edges = _MIGRATIONS.get(schema_name, {})
    if from_version == to_version:
        return []

    # BFS over the directed graph of registered (from, to) edges for this schema.
    parents: dict[str, tuple[str, MigrationFn]] = {}
    queue: deque[str] = deque([from_version])
    visited = {from_version}
    while queue:
        current = queue.popleft()
        for (edge_from, edge_to), fn in edges.items():
            if edge_from != current or edge_to in visited:
                continue
            parents[edge_to] = (current, fn)
            if edge_to == to_version:
                queue.clear()
                break
            visited.add(edge_to)
            queue.append(edge_to)

    if to_version not in parents and to_version != from_version:
        raise MigrationError(
            f"No migration path for {schema_name!r} from {from_version!r} to {to_version!r}."
        )

    path: list[MigrationFn] = []
    node = to_version
    while node != from_version:
        prev, fn = parents[node]
        path.append(fn)
        node = prev
    path.reverse()
    return path


def migrate(
    obj: dict[str, Any], schema_name: str, from_version: str, to_version: str
) -> dict[str, Any]:
    """Migrate a raw JSON document (already `json.load`ed into a `dict`) between schema versions.

    Args:
        obj: The document to migrate, as a plain `dict` (not yet validated
            against a pydantic model -- migration happens before/instead of
            that, since the whole point is that `obj` may not validate
            against the *current* model).
        schema_name: Which document shape `obj` is (e.g. `"TestSummary"`).
        from_version: The schema version `obj` was written under.
        to_version: The schema version to migrate `obj` to.

    Returns:
        A new `dict`, migrated to `to_version`. If `from_version ==
        to_version`, returns a shallow copy of `obj` unchanged (identity
        migration).

    Raises:
        MigrationError: If no chain of registered migrations connects
            `from_version` to `to_version` for `schema_name`.
    """
    if from_version == to_version:
        return dict(obj)
    result = dict(obj)
    for fn in _find_path(schema_name, from_version, to_version):
        result = fn(result)
    return result
