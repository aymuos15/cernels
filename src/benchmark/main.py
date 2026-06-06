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
from tqdm import tqdm

from benchmark.monitor import capture, save
from configs.base import Config
from configs.registry import CONFIGS


def _bar(method):
    """Advance a per-workload tqdm on each call. Writes to the real stderr (sys.__stderr__)
    so it shows live even while monitor.capture() redirects sys.stderr for the log."""
    label = method.__name__.split("benchmark_")[-1]

    def wrapper(self):
        if getattr(self, "_bar", None) is None:
            if KernelBenchmark.active is not None:
                KernelBenchmark.active.close()  # close the previous workload's bar
            self._bar = KernelBenchmark.active = tqdm(desc=label, file=sys.__stderr__)
        method(self)
        self._bar.update(1)

    return wrapper


class KernelBenchmark(Benchmark):
    seed = 42
    iterations = 100
    warmup = 10
    is_local = False
    cfg: Config  # a Config instance, injected per run
    active: "tqdm | None" = None  # the currently-running workload's bar

    def setup(self):
        self.inputs = self.cfg.inputs(self.device, self.cfg.dtype)
        self.compiled = torch.compile(self.cfg.baseline)
        self.buf = torch.empty_like(self.inputs[0]) if self.cfg.out_arg else None

    @staticmethod
    def first(out):
        """Multi-output ops -> first tensor, for the correctness check."""
        return out[0] if isinstance(out, (tuple, list)) else out

    @_bar
    def benchmark_eager(self):
        self.out = self.first(self.cfg.baseline(*self.inputs))

    @_bar
    def benchmark_compile(self):
        self.out = self.first(self.compiled(*self.inputs))

    @_bar
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
@_bar
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
        if KernelBenchmark.active is not None:
            KernelBenchmark.active.close()  # close the last workload's bar
        _print_results_table(results)
    save(name, results, sha, log.getvalue())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m benchmark.main <config>  (a Config.name in src/configs/)")
    main(sys.argv[1])
