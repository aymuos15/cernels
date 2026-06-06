"""Benchmark an op from a config: eager vs compile vs lib (production op) vs custom.

Usage: uv run python -m benchmark.main <config>   # <config> = a Config.name in src/configs/
Writes analysis/<host>/<config>.{json,log}. Works for Hub kernels and non-Hub ops
(e.g. torchvision) alike — the config just provides eager/lib/custom callables.
"""

import sys

import torch
from huggingface_hub.utils import disable_progress_bars
from kernels.cli.benchmark import _print_results_table, get_kernel_sha_from_build_name
from tqdm import tqdm

from benchmark.monitor import capture, stats, time_ms
from benchmark.save import save
from configs.registry import CONFIGS

DEVICE = "cuda"
WARMUP = 10
ITERATIONS = 100


def _first(out):
    """Multi-output ops -> first tensor, for the correctness check."""
    return out[0] if isinstance(out, (tuple, list)) else out


def run(cfg):
    inputs = cfg.inputs(DEVICE, cfg.dtype)
    workloads = {"eager": cfg.baseline}
    if cfg.use_compile:
        workloads["compile"] = torch.compile(cfg.baseline)
    if cfg.lib is not None:  # ops whose eager reference *is* the production op set no separate lib
        workloads["lib"] = cfg.lib
    if cfg.custom is not None:
        workloads["custom"] = cfg.custom
    ref = _first(cfg.baseline(*inputs))

    results = {}
    eager_ms = None
    for name, fn in workloads.items():
        try:
            for _ in range(WARMUP):
                fn(*inputs)
            torch.cuda.synchronize()
            verified = None if name == "eager" else cfg.verify(_first(fn(*inputs)), ref)
            times = [time_ms(lambda: fn(*inputs)) for _ in tqdm(range(ITERATIONS), desc=name, file=sys.__stderr__)]
        except Exception as exc:  # a workload (e.g. compile of a data-dependent op) may fail
            print(f"  {name}: skipped ({type(exc).__name__}: {exc})", file=sys.__stderr__)
            continue
        results[name] = stats(times, verified, None if name == "eager" else eager_ms)
        if name == "eager":
            eager_ms = results[name].mean_ms
    return results


def main(name):
    disable_progress_bars()
    cfg = CONFIGS[name]()
    with capture() as log:
        results = run(cfg)
        _print_results_table(results)
    kernel = getattr(cfg, "kernel", None)
    sha = get_kernel_sha_from_build_name(kernel) if kernel is not None else None
    save(name, cfg, results, sha, log.getvalue())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m benchmark.main <config>  (a Config.name in src/configs/)")
    main(sys.argv[1])
