# Linux setup

Notes on setting up a Linux machine to run vpsych for real (non-headless)
testing: display server/compositor settings for reliable frame timing,
disabling compositing/vsync tearing effects, audio backend notes, and
psychtoolbox keyboard backend requirements. This is written for a
Debian/Ubuntu-family desktop; adjust package manager commands for other
distributions.

None of this matters for running the test suite (`uv run pytest`), which is
fully headless -- it only matters for actually presenting stimuli and
collecting real keypresses on a test machine.

## Python and uv

The project pins Python 3.10 (`.python-version`) because the PsychoPy
wheels this project depends on target 3.10. `uv` handles the interpreter
and virtualenv automatically:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv isn't already installed
uv sync            # installs the pinned Python 3.10 and all dependencies
uv run pytest      # headless test suite
uv run vpsych       # the PySide6 app
```

You do not need a system Python 3.10 install; `uv python install 3.10`
(run automatically by `uv sync`) fetches one into uv's own toolchain
directory, isolated from your system Python.

## Display/compositor settings

Frame-timing accuracy (the whole point of running this suite as a native
app instead of in a browser) depends on `win.flip()` calls landing exactly
on the display's vertical blank, with no compositor in the way reordering
or delaying them.

- **Disable desktop compositing while testing.** A compositing window
  manager (GNOME Shell/Mutter, KDE Plasma/KWin, etc.) inserts its own
  present pipeline between your app and the display, which can add a frame
  or more of latency and, worse, inconsistent latency (jitter) from frame
  to frame. Options, roughly in order of how much control they give you:
  - Run the session under a **minimal window manager** with no compositor
    at all (e.g. a bare `Xorg` session with no WM, or a lightweight
    non-compositing WM) for the machine dedicated to running sessions.
  - On KDE Plasma: *System Settings -> Display and Monitor -> Compositor ->
    disable*, or toggle it on the fly with `Alt+Shift+F12`.
  - On GNOME: `gsettings set org.gnome.mutter workarounds-compositor-disable
    true` (may not fully disable on Wayland; prefer an Xorg session, see
    below).
  - Confirm the window manager is actually letting PsychoPy's pyglet
    backend take over full-screen exclusive-ish presentation; running
    `tools/timing_check.py` and checking the reported jitter/dropped-frame
    numbers is the real test, not any particular settings toggle.
- **Prefer Xorg over Wayland for now.** PsychoPy's pyglet backend and vsync
  behavior are best understood on X11; Wayland compositors vary widely in
  how (or whether) they expose reliable present timing to a plain
  application window. If your desktop environment offers an "Xorg" login
  session option, use it for real sessions.
- **Disable screen blanking / DPMS / screensaver** for the duration of a
  session (`xset s off; xset -dpms` in an Xorg session), so the display
  doesn't power-save mid-session.
- **Single monitor, or explicitly pick the target monitor.** Multi-monitor
  setups with mismatched refresh rates can cause the compositor (if any is
  still active) to pick an inconsistent update rate; run
  `tools/timing_check.py` on the actual monitor/output the session will use.

## Disabling OS color adjustments (night light, HDR)

Both of these silently change the mapping from pixel value to displayed
light and will invalidate a gamma/color calibration if left on, or toggled
mid-session -- they must be off for both calibration and every real
session, per the environment checklist in docs/CALIBRATION.md.

- **Night light / blue-light filter:**
  - GNOME: *Settings -> Displays -> Night Light -> off*, or
    `gsettings set org.gnome.settings-daemon.plugins.color night-light-enabled false`.
  - KDE Plasma: *System Settings -> Display and Monitor -> Night Color ->
    off*.
  - Check for a distro-specific `redshift`/`gammastep` service too
    (`systemctl --user status redshift gammastep 2>/dev/null`) -- these run
    independently of the desktop environment's own night-light toggle and
    will fight with a gamma calibration just as much.
- **HDR / adaptive brightness / auto-contrast:**
  - Most Linux desktops don't enable HDR by default; if yours does (recent
    GNOME/KDE on supported hardware), turn it off in the same Display
    settings panel -- HDR tone-mapping is exactly the kind of nonlinear,
    content-dependent input-to-luminance mapping a gamma calibration
    assumes does not exist.
  - Check for laptop-panel "adaptive brightness" / ambient-light-sensor
    auto-brightness (`power-profiles-daemon`, GNOME's *Automatic Brightness*
    setting) and disable it; a display that changes brightness based on
    room lighting mid-session breaks the luminance calibration exactly like
    HDR does.
  - Disable GPU-driver-level dynamic contrast/gamma "enhancement" features
    if your card's control panel exposes them (uncommon to have these on by
    default on Linux, but AMD/NVIDIA proprietary control panels sometimes
    do).

## Keyboard input backend (psychtoolbox)

`psychopy.hardware.keyboard.Keyboard` uses the `psychtoolbox` package's
low-level keyboard hook for sub-millisecond response timestamps, which on
Linux needs elevated scheduling/input permissions to work well:

- **Real-time scheduling priority.** For the lowest, most consistent
  latency, the process needs permission to raise its scheduling priority.
  Add your user to a realtime-capable group and set limits via
  `/etc/security/limits.d/`:

  ```sh
  sudo groupadd -f realtime
  sudo usermod -aG realtime "$USER"
  ```

  Then create `/etc/security/limits.d/99-realtime.conf`:

  ```
  @realtime   -  rtprio     99
  @realtime   -  memlock    unlimited
  @realtime   -  nice       -20
  ```

  Log out and back in (group membership changes require a new login
  session) before the limits take effect.

- **Input device group permissions.** Reading raw keyboard events (as
  opposed to going through the window system's normal input path) generally
  requires membership in the `input` group:

  ```sh
  sudo usermod -aG input "$USER"
  ```

  Again, log out and back in for this to take effect. Without it,
  `psychtoolbox`'s keyboard backend may silently fall back to a
  higher-latency path, or fail to initialize -- if response-time
  measurements look implausibly slow or noisy, check group membership
  first.

- **Verify** with `groups` after logging back in that both `realtime` and
  `input` are listed, and that a simple PsychoPy keyboard demo reports
  plausible (sub-5ms class) latencies before trusting real session data for
  timing-sensitive tests (e.g. anything measuring RT itself, or CFF).

## Monitor warm-up

Not a Linux-specific setting, but worth repeating here since it interacts
with *when* you run the steps above: power the monitor on and let it run
for **at least 15 minutes** before calibrating (see docs/CALIBRATION.md's
environment checklist) -- LCD backlights measurably drift in luminance (and
sometimes color) during their first several minutes after power-on, so
calibrating a cold display produces numbers that don't hold for the
session that follows.

## Troubleshooting

- **`tools/timing_check.py` reports >1% dropped frames:** compositor is
  likely still active, or the window isn't getting an exclusive-ish
  fullscreen path -- revisit the "Display/compositor settings" section
  above, and confirm you're on Xorg rather than Wayland.
- **Session aborts immediately with `REFRESH_MISMATCH`:** the OS-reported
  refresh rate (baked into your calibration's `geometry.refresh_hz`) no
  longer matches what the display is actually running at (a driver update,
  a cable swap, or a resolution/refresh change since calibration). Re-run
  geometry calibration, or explicitly set the display's refresh rate back
  to what the calibration expects.
- **Keyboard responses look high-latency or jittery:** check `input`/
  `realtime` group membership (above) and that you logged out/in after
  adding them; also confirm no other process is grabbing the keyboard
  device exclusively.
- **Colors look tinted even with Night Light off in the DE settings:**
  check for a separate `redshift`/`gammastep` systemd user service running
  independently of the desktop environment's own toggle (see above).
- **PsychoPy window won't go fullscreen / appears on the wrong monitor:**
  explicitly select the target screen/output in your window manager or via
  PsychoPy's screen-index option, and disable any "join displays as one
  virtual screen" multi-monitor mode while testing.
