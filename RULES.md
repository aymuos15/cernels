# RULES.md — engineering constitution

The invariants this repo holds to; every change is reviewed against them. Operational "how to run" lives in [AGENTS.md](AGENTS.md); mechanics in [docs/guide/](docs/guide/); this file is the principles. When a rule and a convenience conflict, the rule wins.

## 1. Naming — one canonical slug per kernel

Each kernel has ONE canonical `snake_case` slug. Every other name derives from it mechanically — no judgement, enforced by [`scripts/check_naming.py`](scripts/check_naming.py).

| derived form | case | example |
|---|---|---|
| config name / CLI id | snake_case | `qwen3_next_moe_experts` |
| config class | CamelCase | `Qwen3NextMoeExperts` |
| kops dir | snake_case | `src/kops/qwen3_next_moe_experts/` |
| loader file | snake_case | `src/kops/registry/qwen3_next_moe_experts.py` |
| torch-ext pkg + `[kernel.x]` | snake_case | `torch-ext/qwen3_next_moe_experts/` |
| csrc file + op symbol | snake_case | `ops.qwen3_next_moe_experts` |
| build name + Hub repo | kebab-case | `aymuos15/qwen3-next-moe-experts` |

Model-specific kernels carry the model prefix; op-level kernels use the full op name. No abbreviations (`multi_scale_deformable_attention`, not `msda`).

## 2. No dead code

Nothing in the tree that isn't reachable and used.

- No unused imports, names, params, functions, classes, or files (ruff F4xx/F8xx).
- No commented-out code — git is the history.
- No unreachable branches; no single-use indirection that earns nothing.
- Every kops kernel is referenced by a config, every config is in `CONFIGS`, every loader is imported. `check_naming.py` asserts the full chain.
- TODOs live in `.issues/`, never in code.

## 3. Comments & docstrings — only what the code cannot say

Default to NONE. Good names, types, and small functions are the documentation. Add prose ONLY when it carries information the code cannot, namely one of: WHY (rationale / tradeoff) · a non-obvious constraint or invariant · a footgun · a reference (paper, issue, upstream line) · a unit/contract not in the types.

Never: restate the code, narrate structure (no banner comments), paraphrase a signature, or leave commented-out code or stale TODOs.

Docstrings:

- Module: one or two lines — what the file is + any non-obvious contract. Keep.
- Function/class: only when the contract isn't obvious from name + signature + types. A docstring that repeats the signature is deleted.

THE TEST: delete it; if a competent reader loses information they could not recover from the code, restore only the lost part. Otherwise it stays deleted.

Protected (these ARE load-bearing — keep): dtype/unit constraints ("reduce in fp32"), ABI gotchas ("int64_t — schema int maps to int64"), timed-path rationale ("router runs here to stay in the timed path"), magic constants with a source ("clamp(max=7) per gpt-oss"), and CUDA memory-layout / index / shared-mem-budget math. CUDA gets more latitude because that information isn't recoverable from the code.

## 4. Structure

A custom kernel = a self-contained kernel-builder repo `src/kops/<slug>/` (build.toml, csrc/, torch-ext/, flake.{nix,lock}, CARD.md) + a thin loader `src/kops/registry/<slug>.py` + a benchmark config `src/configs/registry/<slug>.py` in `CONFIGS`. Python prep lives in the loader; the CUDA op does only compute; `inputs()` builds raw tensors only. Kernels are Hub-ready at creation: real `aymuos15/<kebab>` repo-id + a filled CARD.md.

## 5. Baselines & benchmarks (honesty)

Never hand-write a reference — call the real library/Hub op. Keep the op's distinctive work in the timed path; precompute only raw inputs. Report the true bar (custom vs `op_compile`, or vs the `hub`/`op_eager` reference where that is the bar). Parity is reported as parity. Done = verifies ✓ on the GB10 Spark; never build or benchmark locally (AGENTS.md).

## 6. Commits

One logical change per commit, reviewable diff, prefixed by area (`kops:`, `configs:`, `docs:`, `scripts:`). Don't commit or push unless asked. Never commit secrets, `build/`, or `analysis/` (gitignored).

## 7. Enforcement

`scripts/check_naming.py` (the §1 invariant + the §2 kernel→config→loader chain), ruff (unused + format), pyrefly (types), and clang-format (CUDA/C++) all run in pre-commit and block the commit on any violation. The judgement parts of §2–§3 — is this indirection dead weight, does this comment earn its place — are review-enforced against the tests above, not a linter.
