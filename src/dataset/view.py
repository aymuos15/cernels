"""View the kernels dataset, interactively."""

import pandas as pd
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

console = Console()
df = pd.read_parquet("kernels.parquet")
COLS = ["id", "downloads", "likes", "n_variants", "backends"]


def cell(v):
    return ", ".join(map(str, v)) if hasattr(v, "__len__") and not isinstance(v, str) else str(v)


def table(rows, title):
    t = Table(title=title, header_style="bold cyan")
    for c in COLS:
        t.add_column(c, justify="right" if rows[c].dtype.kind in "if" else "left", overflow="fold")
    for _, r in rows.iterrows():
        t.add_row(*(cell(r[c]) for c in COLS))
    console.print(t)


def summary(_=None):
    console.print(
        f"[bold]{len(df)}[/] kernel repos · [bold]{df.org.nunique()}[/] orgs · "
        f"[bold]{int(df.downloads.sum()):,}[/] downloads"
    )
    console.print("backends:", df.backends.explode().value_counts().to_dict())
    console.print("torch:   ", df.torch_versions.explode().value_counts().to_dict())


def top():
    n = int(Prompt.ask("how many", default="15"))
    table(df.head(n), f"Top {n} by downloads")


def search():
    q = Prompt.ask("name/org contains").strip().lower()
    hit = df[df.id.str.lower().str.contains(q)]
    table(hit, f"matches for '{q}'") if len(hit) else console.print("[red]no matches")


def by_backend():
    b = Prompt.ask("backend", choices=["cuda", "rocm", "cpu"])
    hit = df[df.backends.apply(lambda bs: b in bs)]
    table(hit, f"{len(hit)} repos with {b} build")


def show():
    q = Prompt.ask("exact id")
    hit = df[df.id == q]
    console.print(hit.iloc[0].to_dict() if len(hit) else "[red]not found")


ACTIONS = {
    "summary": summary,
    "top": top,
    "search": search,
    "backend": by_backend,
    "show": show,
}

while True:
    choice = Prompt.ask("\n[bold]action[/]", choices=[*ACTIONS, "quit"], default="summary")
    if choice == "quit":
        break
    ACTIONS[choice]()
