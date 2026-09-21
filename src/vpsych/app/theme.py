"""Calm, clinical visual design tokens and the app-wide Qt stylesheet.

One accent color, a system font stack, generous spacing, and a consistent
grid -- this is a measurement instrument, not a game. Severity is always
paired with text/markers elsewhere (see
`vpsych.app.viewmodels.results.severity_marker`); the colors below are a
secondary cue only, never the sole signal.
"""

from __future__ import annotations

# Neutral, low-chroma palette plus a single desaturated blue accent.
BACKGROUND = "#f5f6f7"
SURFACE = "#ffffff"
BORDER = "#d7dbe0"
TEXT_PRIMARY = "#1c2126"
TEXT_SECONDARY = "#5a6472"
TEXT_DISABLED = "#9aa3ad"
ACCENT = "#2f5d8a"
ACCENT_TEXT = "#ffffff"
FOCUS_RING = "#2f5d8a"

SEVERITY_INFO = "#3b6ea5"
SEVERITY_WARNING = "#9a6b1c"
SEVERITY_CRITICAL = "#a33a3a"
SEVERITY_SUCCESS = "#2f7d5c"

SPACING_UNIT = 8
FONT_FAMILY = (
    '-apple-system, "Segoe UI", "Helvetica Neue", "Noto Sans", "Liberation Sans", sans-serif'
)

STYLESHEET = f"""
* {{
    font-family: {FONT_FAMILY};
}}

QMainWindow, QWidget {{
    background-color: {BACKGROUND};
    color: {TEXT_PRIMARY};
    font-size: 14px;
}}

QLabel[role="heading"] {{
    font-size: 20px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
}}

QLabel[role="subheading"] {{
    font-size: 15px;
    font-weight: 600;
    color: {TEXT_SECONDARY};
}}

QLabel[role="caption"] {{
    font-size: 12px;
    color: {TEXT_SECONDARY};
}}

QFrame[role="card"] {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}

QFrame[role="card-disabled"] {{
    background-color: {BACKGROUND};
    border: 1px dashed {BORDER};
    border-radius: 8px;
}}

QPushButton {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 16px;
    color: {TEXT_PRIMARY};
}}

QPushButton:hover {{
    border-color: {ACCENT};
}}

QPushButton:disabled {{
    color: {TEXT_DISABLED};
    background-color: {BACKGROUND};
}}

QPushButton[role="primary"] {{
    background-color: {ACCENT};
    color: {ACCENT_TEXT};
    border: 1px solid {ACCENT};
    font-weight: 600;
}}

QPushButton[role="primary"]:disabled {{
    background-color: {BORDER};
    border-color: {BORDER};
    color: {SURFACE};
}}

QPushButton:focus, QLineEdit:focus, QComboBox:focus, QListWidget:focus, QTabBar::tab:focus {{
    outline: none;
    border: 2px solid {FOCUS_RING};
}}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
}}

QListWidget, QTreeWidget, QTableWidget {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}

QProgressBar {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    background-color: {SURFACE};
    text-align: center;
    height: 18px;
}}

QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 3px;
}}
"""


def severity_color(severity: str) -> str:
    """Secondary color cue for a severity level; always paired with text/marker elsewhere."""
    return {
        "info": SEVERITY_INFO,
        "warning": SEVERITY_WARNING,
        "critical": SEVERITY_CRITICAL,
    }.get(severity, TEXT_SECONDARY)
