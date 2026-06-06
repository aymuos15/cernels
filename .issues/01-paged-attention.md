# 01 · Paged attention

**Why compile loses.** Attention is the canonical FlashAttention case: the softmax×matmul is tiled and kept in SRAM so the full S×S scores matrix is never materialized. `torch.compile` cannot synthesize that tiling — it fuses pointwise ops around the attention but the core remains an eager SDPA call. Paged attention adds a block-table KV-cache layout (vLLM-style) that eager has no equivalent for at all, so this is a clear structural win for the kernel.

**Source.** `kernels-community/paged-attention`.

**Config sketch.** `HubConfig`, dtype fp16. The reference op (`baseline`) is `torch.nn.functional.scaled_dot_product_attention` over a non-paged contiguous KV layout. Inputs: query plus a KV cache laid out in blocks, the block table mapping logical→physical blocks, sequence-length metadata, and a scale. `op` is whatever the kernel exposes (e.g. `paged_attention_v1` / a `paged_attention` entry); not `out_arg` unless the signature takes a leading out tensor. Override `verify` with a looser tolerance (atol ~2e-2) since flash-style accumulation reorders adds.

**Inputs to think about.** Decode-shaped: batch 8–32, heads 32, head_dim 128, context length 1k–4k, KV block size 16. This is the realistic serving regime where paged attention pays off; also worth a prefill-shaped variant.

**Difficulty.** Medium-high — the block-table/metadata plumbing is the bulk of the work, not the math. Get the input construction right against the kernel's expected layout before worrying about speed.

**Refs.** vLLM paged attention; FlashAttention paper (tiling/online-softmax). Hub: https://huggingface.co/kernels-community/paged-attention
