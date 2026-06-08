"""Enforce RULES.md §1 (naming invariant) and §2 (kernel->config->loader chain).

Source of truth: each src/kops/<slug>/ directory name is the canonical slug. Every
other name must derive from it (kebab for build name + Hub repo, snake everywhere
else). Exits non-zero on any mismatch so pre-commit can block. Pure stdlib.
"""

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KOPS = ROOT / "src/kops"
REG = KOPS / "registry"
CFG = ROOT / "src/configs/registry"
NS = "aymuos15"

bad: list[str] = []


def kebab(s: str) -> str:
    return s.replace("_", "-")


kernel_dirs = [d for d in sorted(KOPS.iterdir()) if d.is_dir() and d.name not in {"registry", "__pycache__"}]
slugs = {d.name for d in kernel_dirs}

for d in kernel_dirs:
    s = d.name
    bt = d / "build.toml"
    if not bt.exists():
        bad.append(f"[{s}] missing build.toml")
        continue
    try:
        t = tomllib.loads(bt.read_text())
    except tomllib.TOMLDecodeError as e:
        bad.append(f"[{s}] build.toml parse error: {e}")
        continue
    name = t.get("general", {}).get("name")
    if name != kebab(s):
        bad.append(f'[{s}] build.toml name="{name}" != "{kebab(s)}"')
    repo = t.get("general", {}).get("hub", {}).get("repo-id")
    if repo != f"{NS}/{kebab(s)}":
        bad.append(f'[{s}] repo-id="{repo}" != "{NS}/{kebab(s)}"')
    kern = t.get("kernel", {})
    if s not in kern:
        bad.append(f"[{s}] missing [kernel.{s}] (have {list(kern)})")
    elif f"csrc/{s}.cu" not in kern[s].get("src", []):
        bad.append(f"[{s}] [kernel.{s}].src missing csrc/{s}.cu")
    if not (d / "csrc" / f"{s}.cu").exists():
        bad.append(f"[{s}] missing csrc/{s}.cu")
    if not (d / "torch-ext" / s / "__init__.py").exists():
        bad.append(f"[{s}] missing torch-ext/{s}/__init__.py")
    tb = d / "torch-ext" / "torch_binding.cpp"
    if not tb.exists():
        bad.append(f"[{s}] missing torch-ext/torch_binding.cpp")
    else:
        ops = set(re.findall(r'ops\.(?:def|impl)\(\s*"([^"(]+)', tb.read_text()))
        if s not in ops:
            bad.append(f'[{s}] torch_binding.cpp registers no op named "{s}" (found {sorted(ops)})')
    if not (d / "CARD.md").exists():
        bad.append(f"[{s}] missing CARD.md")
    if not (REG / f"{s}.py").exists():
        bad.append(f"[{s}] missing loader src/kops/registry/{s}.py")
    cf = CFG / f"{s}.py"
    if not cf.exists():
        bad.append(f"[{s}] missing config src/configs/registry/{s}.py")
    elif f'name = "{s}"' not in cf.read_text():
        bad.append(f'[{s}] config name != "{s}"')

init = (CFG / "__init__.py").read_text()
for f in sorted(REG.glob("*.py")):
    if not f.stem.startswith("_") and f.stem not in slugs:
        bad.append(f"[{f.stem}] orphan loader: no kops kernel dir")
for f in sorted(CFG.glob("*.py")):
    if f.stem in {"__init__"}:
        continue
    if f.stem not in slugs:
        bad.append(f"[{f.stem}] orphan config: no kops kernel dir")
    elif f"configs.registry.{f.stem} import" not in init:
        bad.append(f"[{f.stem}] config not registered in CONFIGS __init__.py")

if bad:
    print(f"check_naming: {len(bad)} violation(s)")
    for v in bad:
        print("  -", v)
    sys.exit(1)
print(f"check_naming: OK — {len(slugs)} kernels, names consistent")
