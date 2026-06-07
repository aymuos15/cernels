"""Profile a whole model end-to-end: prefill + decode, three lenses.

Usage: uv run --no-sync python -m profiling.main <model> [--no-inductor]
  <model> = a ModelProfile.name in src/profiling/registry/
Writes analysis/<host>/profile/<model>/{profile.json,report.txt,inductor/}. Spark-only (see AGENTS.md).
"""

import sys

import torch

from profiling import engine, report, save
from profiling.engine import Runner
from profiling.registry import MODELS

DEVICE = "cuda"


def run(prof, do_inductor, outdir):
    model, processor = prof.load(DEVICE)
    inputs = prof.inputs(processor, DEVICE)
    runner = Runner(model, inputs, DEVICE, prof.decode_tokens)

    # warmup so lazy init / autotuning doesn't land in the measured pass
    runner.prefill()
    torch.cuda.synchronize()

    results = {
        "prefill": {
            "modules": engine.profile_modules(model, runner.prefill),
            "ops": engine.profile_ops(runner.prefill),
        },
        "decode": {
            "modules": engine.profile_modules(model, runner.decode, setup=runner.setup_decode),
            "ops": engine.profile_ops(runner.decode, setup=runner.setup_decode),
        },
    }

    inductor = []
    if do_inductor:
        print("inductor: compiling the model and dumping generated Triton ...", file=sys.stderr)
        inductor = engine.dump_inductor(model, lambda c: _prefill(c, inputs), prof.name, str(outdir / "inductor"))
    return results, inductor


@torch.no_grad()
def _prefill(model, inputs):
    return model(**inputs, use_cache=True)


def main(name, do_inductor):
    prof = MODELS[name]()
    machine = save.machine()
    outdir = save.model_dir(machine, name)
    results, inductor = run(prof, do_inductor, outdir)
    record = save.build_record(name, prof, results, inductor, machine)
    text = report.build_report(record)
    save.save(record, text, outdir)
    print(text)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if len(args) != 1:
        sys.exit("usage: python -m profiling.main <model> [--no-inductor]")
    main(args[0], do_inductor="--no-inductor" not in sys.argv)
