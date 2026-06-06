"""View saved benchmark results from analysis/<host>/<config>.json (one row per machine)."""

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()
WORKLOADS = ["eager", "compile", "kernel", "custom"]
COLS = ["config", "gpu", "arch", *(f"{w}(ms)" for w in WORKLOADS), "hub vs compile", "custom vs compile", "match"]


def short_gpu(name):
    return name.replace("NVIDIA ", "").replace(" Laptop GPU", "").replace("GeForce ", "")


t = Table(title="benchmark results", header_style="bold cyan")
for c in COLS:
    t.add_column(c, justify="left" if c in ("config", "gpu", "arch", "match") else "right")

rows = []
for path in Path("analysis").glob("*/*.json"):
    r = json.loads(path.read_text())
    mean = {w: d["mean_ms"] for w, d in r["workloads"].items()}
    verified = r["workloads"].get("kernel", {}).get("verified")
    rows.append((r["config"], r["machine"], mean, verified))

for config, mach, mean, verified in sorted(rows, key=lambda x: (x[0], x[1]["gpu"])):
    vs = {
        w: f"{mean['compile'] / mean[w]:.2f}x" if {"compile", w} <= mean.keys() else "-" for w in ("kernel", "custom")
    }
    match = "✓" if verified else ("✗" if verified is False else "·")
    cells = (f"{mean[w]:.3f}" if w in mean else "-" for w in WORKLOADS)
    t.add_row(config, short_gpu(mach["gpu"]), mach["arch"], *cells, vs["kernel"], vs["custom"], match)

console.print(t if t.row_count else "[yellow]no results in analysis/ — run a benchmark first")
