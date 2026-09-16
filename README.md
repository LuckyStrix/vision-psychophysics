# vpsych — vision-psychophysics

A local, calibrated visual psychophysics test suite for measuring visual
acuity, contrast sensitivity, color discrimination, motion coherence, and
temporal and hyperacuity thresholds, using standard adaptive psychophysical
methods (QUEST+, weighted staircases, method of constant stimuli).

**Status: pre-alpha.** Interfaces are still being frozen and most test
implementations are stubs. Expect breaking changes.

## Not a medical device

This software is research tooling. It is **not** a medical device, is
**not** FDA-cleared or CE-marked, and is **not** intended to diagnose, treat,
or screen for any medical condition. Results are only as reliable as your
display calibration and testing environment. Do not use it to make clinical
decisions.

## Why local, not a browser app

Calibration and frame-accurate timing require direct access to the display
and its refresh cycle. Running locally with PsychoPy (pyglet backend) and
a psychtoolbox-backed keyboard lets the runner trust its own timing in a way
a browser tab cannot guarantee.

## Install

This project targets **Python 3.10** (pinned — PsychoPy wheels target 3.10)
and is managed with [`uv`](https://docs.astral.sh/uv/).

```bash
uv python install 3.10
uv sync
uv run pytest
```

Run the desktop app (once implemented):

```bash
uv run vpsych
```

Run a session non-interactively (used by the app, and directly for
headless/simulated end-to-end runs):

```bash
uv run vpsych-run --session-plan plan.json --status-file status.json --data-root ~/vpsych-data
```

See `docs/SETUP_LINUX.md` for Linux display/audio setup notes.

## Layout

```
src/vpsych/
  app/            PySide6 desktop UI
  runner/         Subprocess entry point that runs a session and streams trials to disk
  core/
    display.py    Screen geometry: cm <-> px <-> deg, Nyquist/refresh limits
    calibration/  Geometry, gamma, color primaries, grading
    timing.py     Frame-count durations, dropped-frame accounting
    trial.py      Generic trial record and timeline (fixation -> stimulus -> response -> ITI)
    procedures/   Adaptive procedures: QUEST+, weighted staircase, constant stimuli, qCSF
    psychometric.py  Psychometric function fitting (MLE), bootstrap CIs, goodness of fit
    observers.py  Simulated observers for validating procedures against known thresholds
    rng.py        Seeded, logged random number generation
  tests_catalog/  One plugin per perceptual test (acuity, CSF, color, motion, ...)
  data/           Schemas, BIDS-like path layout, session writer, export
  reports/        HTML session reports and longitudinal plots
tests/            Unit tests and simulated-observer recovery tests
tools/            timing_check.py, gamma_measure.py — standalone calibration/diagnostic scripts
docs/             METHODS.md, DATA_FORMAT.md, CALIBRATION.md, SETUP_LINUX.md
```

Session data is **not** stored in this repository. It lives under
`~/vpsych-data` by default (override with `VPSYCH_DATA_ROOT`), in a
BIDS-behavioral-inspired layout with pseudonymous participant IDs only.

## License

GPL-3.0-or-later (see `LICENSE`). PsychoPy, a core dependency, is GPL-3
licensed.
