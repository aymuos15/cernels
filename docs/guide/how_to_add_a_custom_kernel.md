# How to add a custom kernel

Custom kernels are **HF kernel-builder kernels** (AOT, nix-built, Hub-publishable), living one-per-directory under `src/kops/<name>/`. Each registers a native `torch.library` op (so it composes with `torch.compile`) and is loaded at runtime via `get_local_kernel`. A thin `src/kops/registry/<name>.py` loader exposes `kernel(*inputs)` for the benchmark config. (We no longer use `load_inline`.)

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
└── flake.lock              # commit it (reproducibility)
```

`build.toml`:
```toml
[general]
name = "my-kernel"          # MUST be lowercase + digits + DASHES only (no underscores); maps to torch-ext/my_kernel
version = 1
license = "Apache-2.0"
backends = ["cuda"]

[general.hub]
repo-id = "kernels-local/my-kernel"   # required (placeholder until real Hub publish)

[torch]
src = ["torch-ext/torch_binding.cpp", "torch-ext/torch_binding.h"]

[kernel.my_kernel]          # section-key underscores are fine
backend = "cuda"
depends = ["torch"]
src = ["csrc/my_kernel.cu"]
```

`registration.h` and `_ops.py` are **auto-generated** by `build2cmake` — don't write them. Study `src/kops/rmsnorm/` (simple), `src/kops/rope/` (tuple return), `src/kops/gaussian_blur/` (Python prep split), `src/kops/moe/` (module input).

## 2. Hard rules (each caused a real build failure)

- **`[general] name` must use dashes, not underscores** (`silu-mul`, not `silu_mul`); the `torch-ext/<dir>` and section keys keep underscores.
- **Integer op args must be `int64_t` in C++** (the schema string still says `int`). `int topk` → `static_assert` failure; use `int64_t topk`.
- **`build.toml` needs `version` + `[general.hub] repo-id`** or build2cmake reports a misleading `TOML parse error at line 1`.
- **Keep Python prep out of the op.** The CUDA op is pure (e.g. `gblur(x, ky, kx)`); building the taps / argsort / MoE routing lives in the registry loader (`src/kops/registry/<name>.py`).

## 3. The registry loader

`src/kops/registry/<name>.py` exposes `kernel(*inputs)` for the config:
```python
from pathlib import Path
from typing import Any
_mod: Any = None
_REPO = Path(__file__).resolve().parents[1] / "<name>"   # src/kops/<name>

def _module():
    global _mod
    if _mod is None:
        from kernels import get_local_kernel
        _mod = get_local_kernel(_REPO)
    return _mod

def kernel(*inputs):
    return _module().<fn>(*inputs)   # + any Python prep
```

## 4. Build + benchmark (on the Spark)

```bash
scripts/transfer.sh spark
ssh spark 'cd ~/kernels && bash scripts/build_kernels.sh <name>'   # nix build-and-copy -> src/kops/<name>/build/
ssh spark 'cd ~/kernels && uv run --no-sync python -m benchmark.main <name>'
rsync spark:kernels/analysis/ analysis/ ; uv run --no-sync python -m benchmark.view
```

`scripts/build_kernels.sh` git-inits each kernel dir and runs `nix run .#build-and-copy -L`. Wire `custom = staticmethod(<name>_kernel)` in the config (see [how to add a config](how_to_add_a_config.md)); success = the contender verifies `✓` and beats `op_compile` (see [correctness](correctness.md)).

## 5. Publishing (optional)

The source tree is push-ready: `kernels upload src/kops/<name> --repo-id <org>/<name>`, or `nix build .#redistributable` then push `build/`. Once on the Hub, the loader can switch to `get_kernel(repo, version=...)`.
