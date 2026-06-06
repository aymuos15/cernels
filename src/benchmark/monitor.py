"""Capture and persist benchmark output to analysis/<config>/benchmark.{json,log}."""

import contextlib
import io
import json
import sys
from pathlib import Path

from kernels.cli.benchmark import BenchmarkResult, collect_machine_info

ANALYSIS = Path("analysis")


@contextlib.contextmanager
def capture():
    """Redirect stderr into a buffer for logging, then echo it back to the console."""
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        yield buf
    sys.stderr.write(buf.getvalue())


def save(name, results, sha, log):
    result = BenchmarkResult(
        timing_results=results,
        machine_info=collect_machine_info(),
        kernel_commit_sha=sha,
        benchmark_script_path="src/benchmark/main.py",
    )
    out = ANALYSIS / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(result.to_payload(), indent=2))
    (out / "benchmark.log").write_text(log)
