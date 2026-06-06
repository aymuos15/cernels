"""View saved benchmark results from analysis/."""

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()
WORKLOADS = ["eager", "compile", "kernel", "custom"]
COLS = ["config", *(f"{w}(ms)" for w in WORKLOADS), "speedup", "match"]

t = Table(title="benchmark results", header_style="bold cyan")
for c in COLS:
    t.add_column(c, justify="left" if c in ("config", "match") else "right")

for path in sorted(Path("analysis").glob("*/benchmark.json")):
    data = json.loads(path.read_text())
    mean = {r["workload"]: r["timingResults"]["mean_ms"] for r in data["results"]}
    verified = next((r.get("verified") for r in data["results"] if r["workload"] == "kernel"), None)
    speedup = f"{mean['eager'] / mean['kernel']:.2f}x" if {"eager", "kernel"} <= mean.keys() else ""
    match = "✓" if verified else ("✗" if verified is False else "·")
    cells = (f"{mean[w]:.3f}" if w in mean else "-" for w in WORKLOADS)
    t.add_row(path.parent.name, *cells, speedup, match)

console.print(t if t.row_count else "[yellow]no results in analysis/ — run a benchmark first")
