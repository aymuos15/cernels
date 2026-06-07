"""Benchmark an op from a config: op_eager / op_compile / hub / lib / custom.

Usage: uv run python -m benchmark.main <config>   # <config> = a Config.name in src/configs/
Writes analysis/<host>/<config>.{json,log}. Works for Hub kernels and non-Hub ops
(e.g. torchvision) alike — the config just provides baseline/hub/lib/custom callables.
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
    # The reference (baseline): timed as `hub` when it IS a Hub kernel, else as `op_eager`.
    ref_label = "hub" if cfg.reference_is_hub else "op_eager"
    workloads = {ref_label: cfg.baseline}
    if not cfg.reference_is_hub and cfg.use_compile:
        workloads["op_compile"] = torch.compile(cfg.baseline)
    if not cfg.reference_is_hub and cfg.hub is not None:  # separate Hub-kernel contender
        workloads["hub"] = cfg.hub
    if cfg.lib is not None:  # separate library contender
        workloads["lib"] = cfg.lib
    if cfg.custom is not None:
        workloads["custom"] = cfg.custom
    ref = _first(cfg.baseline(*inputs))

    results = {}
    ref_ms = None
    for name, fn in workloads.items():
        try:
            for _ in range(WARMUP):
                fn(*inputs)
            torch.cuda.synchronize()
            verified = None if name == ref_label else cfg.verify(_first(fn(*inputs)), ref)
            times = [time_ms(lambda: fn(*inputs)) for _ in tqdm(range(ITERATIONS), desc=name, file=sys.__stderr__)]
        except Exception as exc:  # a workload (e.g. compile of a data-dependent op) may fail
            print(f"  {name}: skipped ({type(exc).__name__}: {exc})", file=sys.__stderr__)
            continue
        results[name] = stats(times, verified, None if name == ref_label else ref_ms)
        if name == ref_label:
            ref_ms = results[name].mean_ms
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
