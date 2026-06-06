# 38 · Interleaved M-RoPE (3D multimodal rotary)

**Why compile loses.** Vision-language models (Qwen3-VL, Qwen 3.5, Qwen2.5-VL) position multimodal tokens with **multimodal RoPE**: the rotary dimensions are split across **temporal / height / width** axes so an image/video token carries 3D position. Qwen3-VL's **Interleaved-MRoPE** distributes t/h/w uniformly across the embedding dims (balanced frequency spectrum) rather than in contiguous blocks. It's a RoPE variant with a per-dimension axis assignment and per-axis position ids — a direct extension of our existing rotary win, but the interleaved layout and 3-axis position computation are bespoke. compile launches the rotary as separate elementwise ops; a fused kernel does the 3-axis interleaved rotation in one pass. Our VLM differentiation lane — most kernel work ignores multimodal positioning.

**Source.** Write our own, extending `src/kops/rope.cu`; reference Qwen3-VL Interleaved-MRoPE.

**Config sketch.** `Config` (non-Hub), dtype fp16/bf16. `baseline` = the reference M-RoPE: build per-axis position ids (t,h,w), map each rotary dim to an axis via the interleaved layout, rotate q/k. `custom` = fused interleaved-MRoPE kernel. Inputs: q/k (b, h, s, d), position ids per axis (s, 3), the dim→axis interleave map, cos/sin. `verify` allclose at atol ~1e-2.

**Inputs to think about.** A mixed text+image sequence: b=1, heads 32, head_dim 128, sequence 8k with an image block (e.g. 32×32 patch grid) plus text. Exercise all three axes.

**Difficulty.** Medium — the rotary kernel exists; the work is the 3-axis interleaved dim mapping and correct per-axis position ids. Validate against the transformers Qwen3-VL implementation.

**Refs.** Qwen3-VL Interleaved-MRoPE: https://arxiv.org/abs/2511.21631 · DeepStack/MRoPE writeup: https://thesalt.substack.com/p/qwen3-vl-deepstack-fusion-interleaved · see existing rotary config.
