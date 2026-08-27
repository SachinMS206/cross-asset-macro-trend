"""
update_readme.py

Called by .github/workflows/refresh-live-data.yml after a live run of
scripts/run_demo.py. Reads reports/metrics_table.md (real results from that
run) and splices it into README.md between two markers, replacing whatever
was there before (synthetic-demo numbers on the very first run, or last
week's live numbers on every run after that).

This keeps README.md as the single source of truth a reader sees, while
guaranteeing the numbers in it were produced by an actual, reproducible
--live run rather than typed in by hand.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

START_MARKER = "<!-- LIVE_RESULTS_START -->"
END_MARKER = "<!-- LIVE_RESULTS_END -->"


def main():
    readme_path = Path("README.md")
    metrics_path = Path("reports/metrics_table.md")

    if not metrics_path.exists():
        raise SystemExit("reports/metrics_table.md not found -- run scripts/run_demo.py --live first.")

    metrics_md = metrics_path.read_text().strip()
    readme = readme_path.read_text()

    if START_MARKER not in readme or END_MARKER not in readme:
        raise SystemExit(
            f"README.md is missing the {START_MARKER} / {END_MARKER} markers -- "
            "add them around the results table so this script knows what to replace."
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    block = (
        f"{START_MARKER}\n"
        f"*Last refreshed automatically from live FRED/Yahoo Finance data: {timestamp}. "
        f"See `.github/workflows/refresh-live-data.yml`.*\n\n"
        f"{metrics_md}\n\n"
        f"![Performance](reports/performance.png)\n"
        f"{END_MARKER}"
    )

    before = readme.split(START_MARKER)[0]
    after = readme.split(END_MARKER)[1]
    new_readme = before + block + after

    readme_path.write_text(new_readme)
    print(f"README.md updated with live results as of {timestamp}")


if __name__ == "__main__":
    main()
