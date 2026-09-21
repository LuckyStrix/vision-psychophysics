"""Qt screens for the vpsych desktop app.

Each module here holds one `QWidget` screen; the business logic each screen
displays lives in `vpsych.app.viewmodels` and is unit-tested there without
Qt. Screens only add layout/widget glue and call into `vpsych.data`/
`vpsych.core` for real values -- never inventing placeholder data.
"""

from __future__ import annotations
