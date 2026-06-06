"""Benchmark a kernel from a YAML config: eager vs compile vs kernel.

Usage: uv run python src/benchmark/main.py <config>   # loads src/configs/<config>.yaml
Writes analysis/<config>/benchmark.{json,log}.
"""

import sys
from importlib.resources import files

import torch
import yaml
from huggingface_hub.utils import disable_progress_bars
from kernels.benchmark import Benchmark
from kernels.cli.benchmark import _print_results_table, run_benchmark_class

from benchmark.monitor import capture, save
from configs.helpers import resolve


class KernelBenchmark(Benchmark):
    seed = 42
    iterations = 100
    warmup = 10
    is_local = False
    cfg: dict = {}

    def setup(self):
        dtype = getattr(torch, self.cfg["dtype"])
        self.inputs = tuple(torch.randn(*s, device=self.device, dtype=dtype) for s in self.cfg["inputs"])
        self.ref = resolve(self.cfg["baseline"])
        self.compiled = torch.compile(self.ref)
        self.buf = torch.empty_like(self.inputs[0]) if self.cfg.get("out_arg") else None

    def benchmark_eager(self):
        self.out = self.ref(*self.inputs)

    def benchmark_compile(self):
        self.out = self.compiled(*self.inputs)

    def benchmark_kernel(self):
        fn = getattr(self.kernel, self.cfg["op"])
        if self.buf is not None:
            fn(self.buf, *self.inputs)
            self.out = self.buf
        else:
            self.out = fn(*self.inputs)

    def verify_kernel(self):
        return self.ref(*self.inputs)


def main(name):
    disable_progress_bars()  # the cached-file scan / revision check is not a download
    cfg = yaml.safe_load((files("configs") / f"{name}.yaml").read_text())
    cls = type("KernelBenchmark", (KernelBenchmark,), {"cfg": cfg})
    with capture() as log:
        results, sha = run_benchmark_class(
            cls,
            iterations=KernelBenchmark.iterations,
            warmup=KernelBenchmark.warmup,
            repo_id=cfg["repo"],
            is_local=KernelBenchmark.is_local,
            revision=f"v{cfg['version']}",
        )
        _print_results_table(results)
    save(name, results, sha, log.getvalue())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: main.py <config>  (name of a yaml in src/configs/)")
    main(sys.argv[1])
