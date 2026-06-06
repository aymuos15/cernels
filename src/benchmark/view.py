"""View saved benchmark results from analysis/<host>/<config>.json (one row per machine)."""

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()
WORKLOADS = ["eager", "compile", "lib", "custom"]
COLS = [
    "config",
    "gpu",
    "arch",
    *(f"{w}(ms)" for w in WORKLOADS),
    "lib vs compile",
    "custom vs compile",
    "lib ✓",
    "custom ✓",
]


def short_gpu(name):
    return name.replace("NVIDIA ", "").replace(" Laptop GPU", "").replace("GeForce ", "")


t = Table(title="benchmark results", header_style="bold cyan")
for c in COLS:
    t.add_column(c, justify="left" if c in ("config", "gpu", "arch", "lib ✓", "custom ✓") else "right")

rows = []
for path in Path("analysis").glob("*/*.json"):
    r = json.loads(path.read_text())
    mean = {w: d["mean_ms"] for w, d in r["workloads"].items()}
    # a workload is "defined" for the config (repo -> lib, custom_sha -> custom) even if it didn't run
    defined = {"lib": r["provenance"]["repo"] is not None, "custom": r["provenance"]["custom_sha"] is not None}
    verified = {w: r["workloads"].get(w, {}).get("verified") for w in ("lib", "custom")}
    rows.append((r["config"], r["machine"], mean, defined, verified))


def mark(defined, v):
    if not defined:  # config has no such workload at all
        return "-"
    return "✓" if v else ("✗" if v is False else "·")  # defined but unverified/skipped -> ·


for config, mach, mean, defined, verified in sorted(rows, key=lambda x: (x[0], x[1]["gpu"])):
    vs = {w: f"{mean['compile'] / mean[w]:.2f}x" if {"compile", w} <= mean.keys() else "-" for w in ("lib", "custom")}
    cells = (f"{mean[w]:.3f}" if w in mean else "-" for w in WORKLOADS)
    t.add_row(
        config,
        short_gpu(mach["gpu"]),
        mach["arch"],
        *cells,
        vs["lib"],
        vs["custom"],
        mark(defined["lib"], verified["lib"]),
        mark(defined["custom"], verified["custom"]),
    )

console.print(t if t.row_count else "[yellow]no results in analysis/ — run a benchmark first")
