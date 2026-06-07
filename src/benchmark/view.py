"""View saved benchmark results from analysis/<host>/<config>.json (one row per machine).

Columns: op_eager / op_compile (the reference, run eager + torch.compiled) and the
contenders hub / lib / custom. Speedups are vs the reference bar = op_compile, else
op_eager, else hub (when the Hub kernel is itself the reference).
"""

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()
WORKLOADS = ["op_eager", "op_compile", "hub", "lib", "custom"]
CONTENDERS = ["hub", "lib", "custom"]
DENOM_PRIORITY = ["op_compile", "op_eager", "hub"]  # the speedup bar, best available first
COLS = [
    "config",
    "gpu",
    "arch",
    *(f"{w}(ms)" for w in WORKLOADS),
    *(f"{w} vs ref" for w in CONTENDERS),
    *(f"{w} ✓" for w in CONTENDERS),
]
TEXT_COLS = {"config", "gpu", "arch", *(f"{w} ✓" for w in CONTENDERS)}


def short_gpu(name):
    return name.replace("NVIDIA ", "").replace(" Laptop GPU", "").replace("GeForce ", "")


def mark(declared, present, v):
    if not declared:  # config has no such workload at all
        return "-"
    if not present:  # declared but didn't run (errored / skipped)
        return "·"
    return "✓" if v else ("✗" if v is False else "·")  # ran but no verdict -> ·


t = Table(title="benchmark results", header_style="bold cyan")
for c in COLS:
    t.add_column(c, justify="left" if c in TEXT_COLS else "right")

rows = []
for path in Path("analysis").glob("*/*.json"):
    r = json.loads(path.read_text())
    mean = {w: d["mean_ms"] for w, d in r["workloads"].items()}
    ref_label = r.get("reference", "op_eager")
    declared = r.get("declared", {})
    verified = {w: r["workloads"].get(w, {}).get("verified") for w in CONTENDERS}
    rows.append((r["config"], r["machine"], mean, ref_label, declared, verified))


for config, mach, mean, ref_label, declared, verified in sorted(rows, key=lambda x: (x[0], x[1]["gpu"])):
    denom = next((mean[w] for w in DENOM_PRIORITY if w in mean), None)
    # a contender that is itself the reference (e.g. hub for megablocks) shows no speedup/verdict
    vs = {w: "-" if w == ref_label or denom is None or w not in mean else f"{denom / mean[w]:.2f}x" for w in CONTENDERS}
    marks = {w: "-" if w == ref_label else mark(declared.get(w, False), w in mean, verified[w]) for w in CONTENDERS}
    cells = (f"{mean[w]:.3f}" if w in mean else "-" for w in WORKLOADS)
    t.add_row(
        config,
        short_gpu(mach["gpu"]),
        mach["arch"],
        *cells,
        *(vs[w] for w in CONTENDERS),
        *(marks[w] for w in CONTENDERS),
    )

console.print(t if t.row_count else "[yellow]no results in analysis/ — run a benchmark first")
