"""Benchmark a whole model stock vs kernelized: prefill latency + decode throughput.

Usage: uv run --no-sync python -m modeling.main <model>   # <model> = a ModelProfile.name
Variants come from the model's entry in modeling.swaps (stock first). Before timing, every
kernelized variant passes a correctness gate vs stock: greedy-token prefix match over
GATE_TOKENS plus the max abs diff of the last prefill logits (reported, not gated — bf16
drift over many layers is expected). Writes analysis/<host>/model/<model>.json.
Spark-only (see AGENTS.md).
"""

import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.benchmark import Timer

from benchmark.save import _machine
from modeling.swaps import SWAPS
from profiling.registry import MODELS

DEVICE = "cuda"
GATE_TOKENS = 64
DECODE_TOKENS = 128


@torch.no_grad()
def greedy(model, inputs, n):
    out = model.generate(**inputs, max_new_tokens=n, min_new_tokens=n, do_sample=False)
    return out[0, inputs["input_ids"].size(1) :]


@torch.no_grad()
def last_logits(model, inputs):
    return model(**inputs).logits[0, -1].float()


@torch.no_grad()
def time_generate_s(model, inputs, n, runs=3):
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(**inputs, max_new_tokens=n, min_new_tokens=n, do_sample=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return min(times)


def run(model, inputs, name):
    apply, variants = SWAPS[name]
    results = {}
    ref_tokens = ref_logits = None
    for variant in variants:
        apply(model, variant)
        tokens, logits = greedy(model, inputs, GATE_TOKENS), last_logits(model, inputs)
        if variant == "stock":
            ref_tokens, ref_logits = tokens, logits
        match = int((tokens == ref_tokens).int().cumprod(0).sum())
        with torch.no_grad():
            m = Timer(stmt="model(**inputs)", globals={"model": model, "inputs": inputs}).blocked_autorange(
                min_run_time=5.0
            )
        prefill_ms = m.median * 1e3
        gen_s = time_generate_s(model, inputs, DECODE_TOKENS)
        decode_tok_s = DECODE_TOKENS / (gen_s - m.median)
        results[variant] = {
            "prefill_ms": round(prefill_ms, 2),
            "decode_tok_s": round(decode_tok_s, 2),
            "generate_s": round(gen_s, 3),
            "gate_token_match": f"{match}/{GATE_TOKENS}",
            "last_logits_max_diff": round((logits - ref_logits).abs().max().item(), 4),
        }
        print(f"{variant}: {results[variant]}", file=sys.stderr)
    return results


def main(name):
    prof = MODELS[name]()
    model, processor = prof.load(DEVICE)
    inputs = prof.inputs(processor, DEVICE)
    print(f"{name}: prefill {inputs['input_ids'].shape[1]} tokens, decode {DECODE_TOKENS}", file=sys.stderr)
    results = run(model, inputs, name)
    out = Path("analysis") / _machine()["host"] / "model"
    out.mkdir(parents=True, exist_ok=True)
    payload = {"model": name, "model_id": prof.model_id, "machine": _machine(), "variants": results}
    (out / f"{name}.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload["variants"], indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m modeling.main <model>  (a ModelProfile.name in src/profiling/)")
    main(sys.argv[1])
