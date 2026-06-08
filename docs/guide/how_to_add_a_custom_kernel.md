# How to add a custom kernel

Custom kernels are **HF kernel-builder kernels** (AOT, nix-built, Hub-publishable), living one-per-directory under `src/kops/<name>/`. Each registers a native `torch.library` op (so it composes with `torch.compile`) and is loaded at runtime via `get_local_kernel`. A thin `src/kops/registry/<name>.py` loader exposes `kernel(*inputs)` for the benchmark config. (We no longer use `load_inline`.)

`<name>` is the kernel's canonical `snake_case` slug; every other name (build name, op symbol, torch-ext pkg, loader, config, `aymuos15/<kebab>` repo) derives from it per [RULES.md §1](../../RULES.md) and is enforced by `scripts/check_naming.py`.

> **Spark-only**: kernels build via nix on the GB10 Spark — see [running benchmarks](running_benchmarks.md). The build needs the kernel dir to be a git repo (nix flakes); `build/` is gitignored (push to the Hub, never commit).

## 1. Lay out the kernel-builder repo

```
src/kops/<name>/
├── build.toml
├── csrc/<name>.cu          # the CUDA: a host fn `at::Tensor <fn>(...)`; #include <torch/all.h> (NOT torch/extension.h)
├── torch-ext/
│   ├── torch_binding.cpp    # TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops){ ops.def("<fn>(...) -> Tensor"); ops.impl("<fn>", torch::kCUDA, &<fn>); } REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
│   ├── torch_binding.h      # the <fn> declaration; #include <torch/torch.h>
│   └── <name>/__init__.py   # from ._ops import ops; def <fn>(...): return ops.<fn>(...)
├── flake.nix               # input github:huggingface/kernels ; lib.genKernelFlakeOutputs
├── flake.lock              # commit it (reproducibility)
└── CARD.md                 # Hub README (uploaded as repo README.md) — see §5
```

`build.toml`:
```toml
[general]
name = "my-kernel"          # MUST be lowercase + digits + DASHES only (no underscores); maps to torch-ext/my_kernel
version = 1
license = "Apache-2.0"
backends = ["cuda"]

[general.hub]
repo-id = "aymuos15/my-kernel"        # required; use the real project Hub namespace (aymuos15/<name>), NOT kernels-local

[torch]
src = ["torch-ext/torch_binding.cpp", "torch-ext/torch_binding.h"]

[kernel.my_kernel]          # section-key underscores are fine
backend = "cuda"
depends = ["torch"]
src = ["csrc/my_kernel.cu"]
```

`registration.h` and `_ops.py` are **auto-generated** by `build2cmake` — don't write them. Study `src/kops/rms_norm/` (simple), `src/kops/rotary_embedding/` (tuple return), `src/kops/gaussian_blur_2d/` (Python prep split), `src/kops/megablocks_moe/` (module input).

## 2. Hard rules (each caused a real build failure)

- **`[general] name` must use dashes, not underscores** (`silu-and-mul`, not `silu_and_mul`); the `torch-ext/<dir>` and section keys keep underscores.
- **Integer op args must be `int64_t` in C++** (the schema string still says `int`). `int topk` → `static_assert` failure; use `int64_t topk`.
- **`build.toml` needs `version` + `[general.hub] repo-id`** or build2cmake reports a misleading `TOML parse error at line 1`.
- **Keep Python prep out of the op.** The CUDA op is pure (e.g. `gblur(x, ky, kx)`); building the taps / argsort / MoE routing lives in the registry loader (`src/kops/registry/<name>.py`).

## 3. The registry loader

`src/kops/registry/<name>.py` exposes `kernel(*inputs)` for the config. Load the built kernel via the shared `load(slug)` helper (`src/kops/registry/_local.py`, memoized) — the op symbol equals the slug, so the call is `load("<name>").<name>(...)`:
```python
from kops.registry._local import load

def kernel(*inputs):
    return load("<name>").<name>(*inputs)   # + any Python prep
```
Keep any Python prep (router, reshape, dtype casts) here, not in the CUDA op. Cache static per-module conversions so they aren't re-done every timed iteration (see `gpt_oss_moe_experts.py`).

## 4. Build + benchmark (on the Spark)

```bash
scripts/transfer.sh spark
ssh spark 'cd ~/kernels && bash scripts/build_kernels.sh <name>'   # nix build-and-copy -> src/kops/<name>/build/
ssh spark 'cd ~/kernels && uv run --no-sync python -m benchmark.main <name>'
rsync spark:kernels/analysis/ analysis/ ; uv run --no-sync python -m benchmark.view
```

`scripts/build_kernels.sh` git-inits each kernel dir and runs `nix run .#build-and-copy -L`. Wire `custom = staticmethod(<name>_kernel)` in the config (see [how to add a config](how_to_add_a_config.md)); success = the contender verifies `✓` and beats `op_compile` (see [correctness](correctness.md)).

## 5. The Hub card (`CARD.md`)

Every kernel ships a `CARD.md` so it's publish-ready — kernel-builder fills any template placeholders at build time and uploads it as the repo's `README.md`. Front-matter (`tags: [kernel]`, `library_name: kernels`, `license: apache-2.0`) then: a one-line op description, the reference op, a benchmark table with the real `analysis/` numbers (op_eager / op_compile / hub / lib / custom mean ms + the custom-vs-`op_compile` speedup; say so honestly if it's parity), a `get_kernel("aymuos15/<name>", version=1)` usage snippet, and the aarch64 / sm_121 (GB10) build-target note. Copy an existing card (e.g. `src/kops/qwen3_next_moe_experts/CARD.md`) and adapt.

## 6. Publishing (optional — do not push unless asked)

The source tree is push-ready (real `repo-id` + `CARD.md` in place): `kernels upload src/kops/<name> --repo-id aymuos15/<name>`, or `nix build .#redistributable` then push `build/`. Note `build/` is **aarch64/sm_121-only** (built on the GB10 Spark) — non-GB10 consumers need the full build matrix. Once on the Hub, the loader can switch to `get_kernel(repo, version=...)`.
