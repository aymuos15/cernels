# modeling

Whole-model integration: swap the [kops](../kops/README.md) custom kernels into a real transformers model and benchmark it end-to-end against stock — the model-level complement to the per-op tables.

[`swaps.py`](swaps.py) holds one explicit swap function per model (no auto-discovery, no class patching): instance attributes dispatch to the kops kernels and `stock` deletes them, restoring the class methods. north_mini_code swaps the `Cohere2MoeExperts` forward (grouped GEMM at prefill shapes, fused gather-GEMV at ≤4 tokens); deepseek_ocr_2 swaps `DeepseekOcr2TextExperts` the same way and, in the `custom_full` variant, replaces the SAM attention's `get_decomposed_rel_pos` bias builder with the fused `sam_decomposed_rel_pos` op. Models come from the [profiling registry](../profiling/README.md) (`ModelProfile.load`/`inputs`), so a model added for profiling is benchmarkable here for free.

```bash
ssh <spark> 'bash -lc "cd ~/kernels && uv run --no-sync python -m modeling.main north_mini_code"'
```

Per variant it reports prefill latency (`torch.utils.benchmark` median), decode tok/s (`generate` wall minus prefill), and a correctness gate vs stock: greedy-token prefix match over 64 tokens plus the max abs diff of the last prefill logits (reported, not gated — bf16 drift across 49 layers is expected). Results land in `analysis/<host>/model/<model>.json`; summarize locally with `uv run --no-sync python -m modeling.view`.
