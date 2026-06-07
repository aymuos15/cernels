# Correctness (verify) and use_compile

## verify()

Every non-eager workload (compile / lib / custom) is checked against the eager `baseline` output via `Config.verify(out, ref) -> bool`. The default is:

```python
def verify(self, out, ref) -> bool:
    return bool(torch.allclose(out, ref, atol=1e-2))
```

Override `verify` when:

- **The output isn't a single comparable tensor** — variable-length or index outputs. Compare sets, not values: NMS uses `set(out.tolist()) == set(ref.tolist())`.
- **Accumulation is reordered** (flash-style attention, quantized matmul) — tighten/loosen the bar honestly: `atol ~2e-2`, or a relative / cosine-similarity check for low precision. Document the chosen bar in a comment so a loose tolerance can't hide a wrong layout.
- **Multi-output ops** — the runner compares the *first* returned tensor against the baseline's first; structure the op so that comparison is meaningful (or override `verify` to check the tuple).

## Pin data-dependent selection

For top-k / routing / NMS, the *selection* must match or `verify` compares different choices. The runner builds `inputs` once and feeds the same tensors to every workload, so random inputs are already shared within a run — but pin any **weights** (router, indexer, compressor) so the baseline and the contender select the same indices.

## use_compile

Set `use_compile = False` when the op is **data-dependent** (NMS, `unique`/`nonzero`, ragged loops, top-k-with-threshold). `torch.compile` only graph-breaks there, so the compile workload measures nothing useful — and may raise, in which case the runner just skips it. Leave it `True` (the default) for everything compile can actually trace.

## Benchmark the real op — keep its distinctive work in the timed path

`inputs()` builds the op's *inputs*; it must not do the op's *work*. Every workload (eager / compile / lib / custom) is timed calling the op on those inputs, so anything you precompute in `inputs()` is excluded from the measurement — and if you precompute the part that makes this op hard, you benchmark a trivialized stand-in and report a hollow "win."

Example: for interleaved M-RoPE, building the interleaved `cos`/`sin` tables in `inputs()` reduces the kernel to plain RoPE — the interleave (the distinctive work) never runs in the timed path, so a ~1× result is meaningless. The interleave must happen inside the op (the baseline and the kernel), not in `inputs()`. Rule of thumb: `inputs()` produces the *same raw tensors the real caller would pass*; the op-specific transform belongs in `baseline`/`custom`.
