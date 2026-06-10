"""Benchmark a whole model stock vs kernelized: prefill latency + decode throughput.

Usage: uv run --no-sync python -m modeling.main <model>   # <model> = a ModelProfile.name
Variants: stock (eager), custom_prefill (grouped-GEMM experts kernel at prefill shapes only),
custom (+ fused gather-GEMV at decode shapes). Before timing, every kernelized variant passes
a correctness gate vs stock: greedy-token prefix match over GATE_TOKENS plus the max abs diff
of the last prefill logits (reported, not gated — bf16 drift over 49 layers is expected).
Writes analysis/<host>/model/<model>.json. Spark-only (see AGENTS.md).
"""

import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.benchmark import Timer

from benchmark.save import _machine
from profiling.registry import MODELS

DEVICE = "cuda"
GATE_TOKENS = 64
DECODE_TOKENS = 128
DECODE_MAX_TOKENS = 4  # at or below this many tokens the decode entry point handles the op
VARIANTS = ("stock", "custom_prefill", "custom")


def kernelize(model, variant):
    """Explicitly swap our CUDA ops into every Cohere2MoeExperts instance (or restore stock)."""
    from kops.registry.cohere2_moe_experts import decode_kernel, kernel

    n = 0
    for mod in model.modules():
        if type(mod).__name__ != "Cohere2MoeExperts":
            continue
        n += 1
        if "forward" in vars(mod):
            del mod.forward  # drop a previous swap: back to the class forward
        if variant == "stock":
            continue

        def fwd(h, idx, w, _m=mod, _decode_too=variant == "custom"):
            if h.size(0) <= DECODE_MAX_TOKENS:
                return decode_kernel(h, idx, w, _m) if _decode_too else type(_m).forward(_m, h, idx, w)
            return kernel(h, idx, w, _m)

        mod.forward = fwd
    if not n:
        sys.exit("no Cohere2MoeExperts modules found — wrong model for this swap")
    return n


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


def run(model, inputs):
    results = {}
    ref_tokens = ref_logits = None
    for variant in VARIANTS:
        kernelize(model, variant)
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
    results = run(model, inputs)
    out = Path("analysis") / _machine()["host"] / "model"
    out.mkdir(parents=True, exist_ok=True)
    payload = {"model": name, "model_id": prof.model_id, "machine": _machine(), "variants": results}
    (out / f"{name}.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload["variants"], indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m modeling.main <model>  (a ModelProfile.name in src/profiling/)")
    main(sys.argv[1])
