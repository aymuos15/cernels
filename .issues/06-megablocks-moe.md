# 06 · MoE grouped GEMM (MegaBlocks)

**Why compile loses.** A Mixture-of-Experts layer routes each token to a few experts and runs a different weight matrix per expert. Done naively this is a Python loop of variable-sized matmuls (or a masked dense matmul that wastes most of the FLOPs). MegaBlocks expresses it as a single block-sparse / grouped GEMM. `torch.compile` cannot fuse data-dependent routing into a grouped matmul — the dynamic shapes and gather/scatter defeat it — so this is one of the largest structural wins in the backlog.

**Source.** `kernels-community/megablocks`. Note alternative backends worth benchmarking under the same config: `kernels-community/scattermoe` and `kernels-community/sonic-moe`.

**Config sketch.** `HubConfig`, dtype bf16. `baseline` is a dense/loop reference MoE: compute router logits, top-k select, then for each expert gather its tokens and apply its MLP (gate/up/down with SiLU). `op` is the kernel's grouped-GEMM / MoE entry. Not `out_arg`. Inputs: hidden states (tokens, hidden), router weights, per-expert weight stacks (E, hidden, ffn), and top-k. `verify` over the routed output at atol ~2e-2; ensure the baseline uses the same routing decisions as the kernel (route once, feed both).

**Inputs to think about.** tokens=4096 (e.g. batch×seq), hidden=2048, ffn=5632, E=8 experts, top-k=2 — a small-MoE regime that fits one GPU.

**Difficulty.** High — routing/gather/scatter plumbing and making the baseline use identical routing are the hard parts. Start with a fixed (non-random) router so eager and kernel see the same assignment.

**Refs.** MegaBlocks paper (block-sparse MoE); ScatterMoE. Hub: https://huggingface.co/kernels-community/megablocks
