"""Render a self-contained HTML session report with inline figures and data.

Produces a single .html file suitable for offline viewing, containing all
figures as base64-encoded PNGs and comprehensive session/test metadata.
"""

from __future__ import annotations

import base64
import contextlib
import io
import math
from datetime import datetime
from pathlib import Path

from matplotlib.figure import Figure

from vpsych.data import dataset
from vpsych.data.schemas import SessionInfo, TestSummary
from vpsych.data.tsv import read_trials_tsv
from vpsych.reports.figures import psychometric_figure, quality_badges


def _figure_to_base64_png(fig: Figure) -> str:
    """Encode a matplotlib Figure as a base64-encoded PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def _get_calibration_info(
    session: SessionInfo, data_root: Path
) -> tuple[str, str | None, str | None, str]:
    """Retrieve calibration grade info for a session.

    Returns:
        (calibration_hash, luminance_grade, color_grade, staleness_text)
    """
    calibration_hash = session.calibration_hash
    luminance_grade = None
    color_grade = None
    staleness_text = "not available"

    try:
        calibration = dataset.load_calibration(calibration_hash, data_root)
        luminance_grade = calibration.luminance_grade
        color_grade = calibration.color_grade

        # Check staleness
        staleness_text = "stale (>30 days)" if calibration.is_stale(max_age_days=30) else "current"
    except (FileNotFoundError, ValueError):
        staleness_text = "not found"

    return calibration_hash, luminance_grade, color_grade, staleness_text


def _html_escape(text: str) -> str:
    """Escape text for safe HTML inclusion."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _format_timestamp(ts: datetime | None) -> str:
    """Format a datetime for display."""
    if ts is None:
        return "not available"
    return ts.strftime("%Y-%m-%d %H:%M:%S %Z")


def _quality_badge_html(code: str, severity: str, message: str) -> str:
    """Generate HTML for a quality flag badge."""
    severity_colors = {
        "info": "#4A90E2",
        "warning": "#F5A623",
        "critical": "#E74C3C",
    }
    color = severity_colors.get(severity, "#999")
    return (
        f'<span style="display:inline-block; margin:4px 4px 4px 0; padding:4px 8px; '
        f'background-color:{color}; color:white; border-radius:4px; font-size:12px;" '
        f'title="{_html_escape(message)}">{_html_escape(code)}</span>'
    )


def render_session_report(session_dir: Path, out_path: Path | None = None) -> Path:
    """Render a self-contained HTML session report.

    Reads session.json, all _summary.json and _trials.tsv files from the
    session, generates figures, and writes a standalone HTML file containing
    all data and figures as inlined base64 PNGs (no external assets).

    Args:
        session_dir: Path to a session directory (sub-XXXX/ses-YYYYMMDDTHHMMSS/).
        out_path: Where to write the HTML report, or None to write
            `{session_dir}/report.html`.

    Returns:
        The path to the written HTML file.

    Raises:
        FileNotFoundError: If session.json is not found in session_dir.
        ValueError: If session.json cannot be parsed.
    """
    # Load session.json
    session_json_path = session_dir / "session.json"
    if not session_json_path.exists():
        raise FileNotFoundError(f"session.json not found in {session_dir}")

    session = SessionInfo.model_validate_json(session_json_path.read_text(encoding="utf-8"))
    participant_id = session.participant_id
    session_id = session.session_id

    # Infer data root from session directory structure
    # session_dir is .../sub-XXXX/ses-YYYYMMDDTHHMMSS, so parent.parent is data_root
    data_root = session_dir.parent.parent

    # Get calibration info
    cal_hash, lum_grade, color_grade, staleness = _get_calibration_info(session, data_root)

    # Load and render summaries
    beh_dir = session_dir / "beh"
    summaries_html = []

    if beh_dir.exists():
        for summary_path in sorted(beh_dir.glob("*_summary.json")):
            try:
                summary = TestSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
            except Exception as e:
                summaries_html.append(
                    f'<div class="test-result"><p>Error loading {summary_path.name}: {e}</p></div>'
                )
                continue

            # Load corresponding trials for psychometric figure
            trials_path = summary_path.parent / summary_path.name.replace(
                "_summary.json", "_trials.tsv"
            )
            trials_df = None
            if trials_path.exists():
                with contextlib.suppress(Exception):
                    trials_df = read_trials_tsv(trials_path)

            # Build HTML for this test result
            test_html = f'<div class="test-result"><h3>{summary.task_id} (Eye: {summary.eye}, Run: {summary.run})</h3>'

            # Threshold and CI
            if summary.estimate.value is not None and not math.isnan(summary.estimate.value):
                test_html += (
                    f"<p><strong>Threshold:</strong> {summary.estimate.value:.4f} "
                    f"[{summary.estimate.ci_low:.4f}, {summary.estimate.ci_high:.4f}] "
                    f"{summary.estimate.units}</p>"
                )
            else:
                test_html += "<p><strong>Threshold:</strong> not available</p>"

            # Quality flags
            badges = quality_badges(summary)
            if badges:
                test_html += '<div style="margin:8px 0;"><strong>Quality:</strong> '
                for code, severity, message in badges:
                    test_html += _quality_badge_html(code, severity, message)
                test_html += "</div>"

            # Trial counts
            test_html += (
                f"<p><strong>Trials:</strong> {summary.n_trials} main, {summary.n_catch} catch"
            )
            if summary.n_catch > 0:
                test_html += f" (lapse rate: {summary.catch_lapse_rate:.2%})"
            test_html += "</p>"

            # Psychometric figure
            if trials_df is not None and len(trials_df) > 0:
                try:
                    from vpsych.reports.figures import matplotlib_use_agg

                    matplotlib_use_agg()
                    fig = psychometric_figure(summary, trials_df)
                    fig_uri = _figure_to_base64_png(fig)
                    test_html += f'<img src="{fig_uri}" alt="Psychometric function" style="max-width:800px; height:auto;" />'
                    import matplotlib.pyplot as plt

                    plt.close(fig)
                except Exception as e:
                    test_html += f"<p><em>Could not render psychometric figure: {e}</em></p>"

            test_html += "</div>"
            summaries_html.append(test_html)

    # Build complete HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vision Psychophysics Report - {participant_id} {session_id}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}
        h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2em;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 1.5em;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
            margin-bottom: 12px;
            font-size: 1.2em;
        }}
        p {{
            margin-bottom: 10px;
        }}
        .session-info {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
            border-left: 4px solid #0072B2;
        }}
        .session-info p {{
            margin-bottom: 8px;
            font-size: 0.95em;
        }}
        .session-info strong {{
            color: #2c3e50;
        }}
        .calibration-info {{
            background-color: #fff3cd;
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 20px;
            border-left: 4px solid #F5A623;
        }}
        .test-result {{
            background-color: #fafafa;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
            border-left: 4px solid #009E73;
        }}
        .test-result img {{
            margin-top: 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .disclaimer {{
            background-color: #ffe6e6;
            color: #c0392b;
            padding: 12px;
            border-radius: 4px;
            margin-top: 30px;
            border-left: 4px solid #c0392b;
            font-weight: bold;
        }}
        .not-available {{
            color: #7f8c8d;
            font-style: italic;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            text-align: left;
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #ecf0f1;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        @media (prefers-color-scheme: dark) {{
            body {{
                background-color: #1e1e1e;
                color: #e0e0e0;
            }}
            .container {{
                background-color: #2d2d2d;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
            }}
            h1, h2, h3 {{
                color: #e0e0e0;
            }}
            .session-info, .test-result {{
                background-color: #333;
            }}
            .calibration-info {{
                background-color: #4d3a1a;
                color: #f0e68c;
            }}
            .disclaimer {{
                background-color: #3a1515;
                color: #ff9999;
            }}
            table {{
                border-color: #444;
            }}
            th {{
                background-color: #444;
            }}
            tr:hover {{
                background-color: #3a3a3a;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Vision Psychophysics Session Report</h1>

        <div class="session-info">
            <h2>Session Information</h2>
            <table>
                <tr>
                    <th>Field</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Participant ID</td>
                    <td><code>{_html_escape(participant_id)}</code></td>
                </tr>
                <tr>
                    <td>Session ID</td>
                    <td><code>{_html_escape(session_id)}</code></td>
                </tr>
                <tr>
                    <td>Started</td>
                    <td>{_format_timestamp(session.started_utc)}</td>
                </tr>
                <tr>
                    <td>Ended</td>
                    <td>{_format_timestamp(session.ended_utc)}</td>
                </tr>
                <tr>
                    <td>Status</td>
                    <td>{_html_escape(session.status)}</td>
                </tr>
                <tr>
                    <td>Software Version</td>
                    <td>{_html_escape(session.software_version)}</td>
                </tr>
                <tr>
                    <td>Git Commit</td>
                    <td><code>{_html_escape(session.git_commit or "not available")}</code></td>
                </tr>
            </table>
        </div>

        <div class="calibration-info">
            <h2>Calibration Information</h2>
            <table>
                <tr>
                    <th>Field</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Calibration Hash</td>
                    <td><code>{_html_escape(cal_hash[:16])}...</code></td>
                </tr>
                <tr>
                    <td>Luminance Grade</td>
                    <td>{lum_grade or "not available"}</td>
                </tr>
                <tr>
                    <td>Color Grade</td>
                    <td>{color_grade or "not available"}</td>
                </tr>
                <tr>
                    <td>Status</td>
                    <td>{_html_escape(staleness)}</td>
                </tr>
            </table>
        </div>

        <div class="session-info">
            <h2>Display Information</h2>
            <table>
                <tr>
                    <th>Field</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Resolution</td>
                    <td>{session.display.width_px} x {session.display.height_px} px</td>
                </tr>
                <tr>
                    <td>Physical Size</td>
                    <td>{session.display.width_cm:.1f} x {session.display.height_cm:.1f} cm</td>
                </tr>
                <tr>
                    <td>Viewing Distance</td>
                    <td>{session.display.viewing_distance_cm:.1f} cm</td>
                </tr>
                <tr>
                    <td>Refresh Rate</td>
                    <td>{session.display.refresh_hz:.1f} Hz</td>
                </tr>
            </table>
        </div>

        <h2>Test Results</h2>
        {chr(10).join(summaries_html) if summaries_html else '<p class="not-available">No test results available</p>'}

        <div class="disclaimer">
            <p>⚠️ DISCLAIMER: This is not a medical device and should not be used for clinical diagnosis or monitoring.</p>
        </div>
    </div>
</body>
</html>
"""

    # Write HTML
    out_path = session_dir / "report.html" if out_path is None else Path(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding="utf-8")

    return out_path
