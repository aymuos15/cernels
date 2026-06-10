"""View saved whole-model results from analysis/<host>/model/<model>.json (one row per variant)."""

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()
TEXT_COLS = {"model", "gpu", "variant", "greedy match"}
COLS = ["model", "gpu", "variant", "prefill_ms", "decode_tok_s", "generate_s", "greedy match", "logits Δmax"]

t = Table(title="whole-model results", header_style="bold cyan")
for c in COLS:
    t.add_column(c, justify="left" if c in TEXT_COLS else "right")

for path in sorted(Path("analysis").glob("*/model/*.json")):
    r = json.loads(path.read_text())
    for variant, v in r["variants"].items():
        t.add_row(
            r["model"],
            r["machine"]["gpu"].replace("NVIDIA ", ""),
            variant,
            f"{v['prefill_ms']:.1f}",
            f"{v['decode_tok_s']:.2f}",
            f"{v['generate_s']:.2f}",
            v["gate_token_match"],
            f"{v['last_logits_max_diff']:.3f}",
        )

console.print(t if t.row_count else "[yellow]no model results in analysis/*/model/ — run modeling.main first")
