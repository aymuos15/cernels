"""View saved profiles from analysis/<host>/profile/*.json (read-only, local ok).

Usage: uv run --no-sync python -m profiling.view [<model>]
"""

import json
import sys
from pathlib import Path

from profiling import report


def main(which=None):
    paths = sorted(Path("analysis").glob("*/profile/*/profile.json"))
    if which:
        paths = [p for p in paths if p.parent.name == which]
    if not paths:
        print("no profiles in analysis/*/profile/ — run profiling.main first")
        return
    for p in paths:
        print(report.build_report(json.loads(p.read_text())))
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
