"""Benchmark a kernel from a Python config: eager vs compile vs kernel (+ custom).

Usage: uv run python -m benchmark.main <config>   # <config> = a Config.name in src/configs/
Writes analysis/<config>/benchmark.{json,log}.
"""

import sys
from typing import Any

import torch
from huggingface_hub.utils import disable_progress_bars
from kernels.benchmark import Benchmark
from kernels.cli.benchmark import _print_results_table, run_benchmark_class

from benchmark.monitor import capture, save
from configs.base import Config
from configs.registry import CONFIGS


class KernelBenchmark(Benchmark):
    seed = 42
    iterations = 100
    warmup = 10
    is_local = False
    cfg: Config  # a Config instance, injected per run

    def setup(self):
        self.inputs = self.cfg.inputs(self.device, self.cfg.dtype)
        self.compiled = torch.compile(self.cfg.baseline)
        self.buf = torch.empty_like(self.inputs[0]) if self.cfg.out_arg else None

    @staticmethod
    def first(out):
        """Multi-output ops -> first tensor, for the correctness check."""
        return out[0] if isinstance(out, (tuple, list)) else out

    def benchmark_eager(self):
        self.out = self.first(self.cfg.baseline(*self.inputs))

    def benchmark_compile(self):
        self.out = self.first(self.compiled(*self.inputs))

    def benchmark_kernel(self):
        fn = getattr(self.kernel, self.cfg.op)
        if self.buf is not None:
            fn(self.buf, *self.inputs)
            self.out = self.buf
        else:
            self.out = self.first(fn(*self.inputs))

    def verify_kernel(self):
        return self.first(self.cfg.baseline(*self.inputs))


# Injected only when a config sets `custom` (a callable, e.g. from src/kops/).
def _benchmark_custom(self):
    self.out = self.first(self.cfg.custom(*self.inputs))


def _verify_custom(self):
    return self.first(self.cfg.baseline(*self.inputs))


def main(name):
    disable_progress_bars()  # the cached-file scan / revision check is not a download
    cfg = CONFIGS[name]()
    attrs: dict[str, Any] = {"cfg": cfg}
    if cfg.custom is not None:
        attrs["benchmark_custom"] = _benchmark_custom
        attrs["verify_custom"] = _verify_custom
    cls = type("KernelBenchmark", (KernelBenchmark,), attrs)
    with capture() as log:
        results, sha = run_benchmark_class(
            cls,
            iterations=KernelBenchmark.iterations,
            warmup=KernelBenchmark.warmup,
            repo_id=cfg.repo,
            is_local=KernelBenchmark.is_local,
            revision=f"v{cfg.version}",
        )
        _print_results_table(results)
    save(name, results, sha, log.getvalue())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m benchmark.main <config>  (a Config.name in src/configs/)")
    main(sys.argv[1])
