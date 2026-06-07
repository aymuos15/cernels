"""Persist a profile run to analysis/<host>/profile/<model>/ (one dir per model).

analysis/<host>/profile/<model>/
    profile.json     # the structured record
    report.txt       # the human report
    inductor/        # generated Triton per graph region + INDEX.md (written by engine)
"""

import json
from datetime import datetime
from pathlib import Path

from benchmark.save import _machine

ANALYSIS = Path("analysis")


def model_dir(machine, name):
    return ANALYSIS / machine["host"] / "profile" / name


def build_record(name, prof, results, inductor, machine):
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": name,
        "model_id": prof.model_id,
        "dtype": str(prof.dtype).removeprefix("torch."),
        "decode_tokens": prof.decode_tokens,
        "machine": machine,
        "phases": {phase: {"modules": data["modules"], "ops": data["ops"]} for phase, data in results.items()},
        "inductor": inductor,  # [{region, file, kernels}]
    }


def save(record, report_text, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "profile.json").write_text(json.dumps(record, indent=2))
    (outdir / "report.txt").write_text(report_text)
    return outdir


def machine():
    return _machine()
