"""Persist benchmark output to analysis/<host>/<config>.{json,log} (latest run per machine)."""

import contextlib
import hashlib
import importlib
import io
import json
import platform
import socket
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

ANALYSIS = Path("analysis")


@contextlib.contextmanager
def capture():
    """Redirect stderr into a buffer for logging, then echo it back to the console."""
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        yield buf
    sys.stderr.write(buf.getvalue())


def _machine():
    cc = torch.cuda.get_device_capability()
    return {
        "gpu": torch.cuda.get_device_name(),
        "cc": f"{cc[0]}.{cc[1]}",
        "arch": platform.machine(),
        "backend": f"CUDA {torch.version.cuda}",
        "torch": torch.__version__,
        "os": f"{platform.system()} {platform.release()}",
        "host": socket.gethostname(),
    }


def _shapes(cfg):
    try:  # meta device gives shapes without allocating
        return [list(t.shape) for t in cfg.inputs("meta", cfg.dtype)]
    except Exception:
        return []


def _custom_sha(cfg):
    if cfg.custom is None:
        return None
    mod = importlib.import_module(cfg.custom.__module__)
    assert mod.__file__
    src = Path(mod.__file__)
    cu = src.with_suffix(".cu")  # the .cu is what determines perf; fall back to the .py
    return hashlib.sha256((cu if cu.exists() else src).read_bytes()).hexdigest()[:12]


def save(name, cfg, results, sha, log):
    machine = _machine()
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": name,
        "op": cfg.op,
        "dtype": str(cfg.dtype).removeprefix("torch."),
        "input_shapes": _shapes(cfg),
        "machine": machine,
        "provenance": {
            "repo": cfg.repo,
            "version": cfg.version,
            "kernel_commit_sha": sha,
            "custom_sha": _custom_sha(cfg),
        },
        "workloads": {w: asdict(t) for w, t in results.items()},
    }
    out = ANALYSIS / machine["host"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(record, indent=2))
    (out / f"{name}.log").write_text(log)
