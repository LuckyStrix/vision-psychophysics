"""Non-Qt view-model logic for the desktop app.

Every module in this package is pure Python (no PySide6/Qt import) so it is
directly unit-testable without a display: view-model construction, display
string formatting, plan/request building, and status-to-text mapping. Qt
screens under `vpsych.app.screens` import these modules and only add the
widget/layout glue.
"""

from __future__ import annotations
