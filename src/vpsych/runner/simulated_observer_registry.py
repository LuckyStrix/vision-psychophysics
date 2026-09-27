"""Registry of `--simulate`/`--simulate-config` observer "kind" extensions.

Split out of `vpsych.runner.__main__` so that a test package (e.g.
`color_discrimination`) can register its own kind without importing
`vpsych.runner.__main__` itself. That module is executed as `__main__` when
the runner is launched via `python -m vpsych.runner` (the `build_runner_command`
default -- see `vpsych.app.runner_process`), which makes it a *different*
module object, with its own separate module-level state, from the
`vpsych.runner.__main__` that `import`/`from ... import` reaches by name
(the same file, loaded twice under two different names is standard, if
surprising, Python module-identity behavior for anything literally named
`__main__.py`). A registration side effect against one copy is invisible to
`run_session` running as the other, so `python -m vpsych.runner --simulate
trivector:...` used to fail with "Unknown simulated observer kind" even
though `vpsych-run --simulate trivector:...` (the console-script entry
point, which never executes this file as `__main__`) worked fine. Neither
this module nor anything it imports is ever run as `__main__`, so it has
only one identity regardless of how the runner was launched, and both
`vpsych.runner.__main__` and a test package's own `observer.py` should
import the registry from here, not from `vpsych.runner.__main__`.
"""

from __future__ import annotations

from collections.abc import Callable

from vpsych.core.observers import SimulatedObserver

#: See `register_simulated_observer_kind`.
REGISTERED_SIMULATED_OBSERVER_KINDS: dict[str, Callable[[dict[str, float]], SimulatedObserver]] = {}


def register_simulated_observer_kind(
    kind: str, builder: Callable[[dict[str, float]], SimulatedObserver]
) -> None:
    """Register a `build_simulated_observer` "kind" extension.

    Args:
        kind: The `--simulate kind:...`/`{"kind": ...}` name this builder
            handles, e.g. `"trivector"`. Must not collide with a built-in
            kind (`"psychometric"`, `"csf"`) or an already-registered one.
        builder: Called with the parsed `dict[str, float]` params (see
            `vpsych.runner.__main__._parse_kv_params`) and must return a
            constructed `SimulatedObserver`, raising `ValueError` for a
            missing/invalid parameter -- the same contract
            `build_simulated_observer`'s built-in kinds follow.

    Raises:
        ValueError: If `kind` is a built-in kind or already registered.
    """
    if kind in ("psychometric", "csf"):
        raise ValueError(f"Simulated observer kind {kind!r} is a built-in kind, cannot register.")
    if kind in REGISTERED_SIMULATED_OBSERVER_KINDS:
        raise ValueError(f"Simulated observer kind {kind!r} is already registered.")
    REGISTERED_SIMULATED_OBSERVER_KINDS[kind] = builder
