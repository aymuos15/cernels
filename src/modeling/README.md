# modeling

Whole-model integration: swap the [kops](../kops/README.md) custom kernels into a real transformers model and benchmark it end-to-end against stock — the model-level complement to the per-op tables.

`kernelize(model, variant)` does explicit per-instance swaps (no auto-discovery, no class patching): for every `Cohere2MoeExperts` module it assigns an instance `forward` that dispatches to the grouped-GEMM kernel at prefill shapes and (for the `custom` variant) the fused gather-GEMV at decode shapes (≤4 tokens); `stock` deletes the instance attribute, restoring the class forward. Models come from the [profiling registry](../profiling/README.md) (`ModelProfile.load`/`inputs`), so a model added for profiling is benchmarkable here for free.

```bash
ssh <spark> 'bash -lc "cd ~/kernels && uv run --no-sync python -m modeling.main north_mini_code"'
```

Per variant it reports prefill latency (`torch.utils.benchmark` median), decode tok/s (`generate` wall minus prefill), and a correctness gate vs stock: greedy-token prefix match over 64 tokens plus the max abs diff of the last prefill logits (reported, not gated — bf16 drift across 49 layers is expected). Results land in `analysis/<host>/model/<model>.json`; summarize locally with `uv run --no-sync python -m modeling.view`.
