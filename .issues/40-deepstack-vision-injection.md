# 40 · DeepStack vision-feature injection

**Why compile loses.** Qwen3-VL's **DeepStack** takes multi-level features from the vision ViT and injects them into *different* LLM layers (rather than only feeding the final ViT output at the input), strengthening vision-language alignment without lengthening the context. The op is an irregular scatter: route level-`l` visual tokens to the positions/layers they belong to and add them into the residual stream at that depth. The data-dependent placement (which tokens, which layer) is a #1/#2 graph break for `torch.compile`, and the cross-layer injection schedule isn't something it can fuse. A kernel does the gather/scatter-add of visual features into the hidden states in one pass.

**Source.** Write our own; reference Qwen3-VL DeepStack.

**Config sketch.** `Config` (non-Hub), dtype bf16. `baseline` = the eager DeepStack: for each injected level, index the visual-token positions in the hidden state and add the (projected) ViT features. `custom` = fused scatter-add of multi-level features. Inputs: hidden states (tokens, hidden), per-level ViT feature tensors, the visual-token position index, per-level projection weights. `verify` allclose at atol ~1e-2.

**Inputs to think about.** sequence 8k with an image block of ~1k visual tokens, 3–4 injected ViT levels, hidden 4096. Vary the number of visual tokens and levels.

**Difficulty.** Medium — mostly indexed scatter-add plumbing (related to issue 20 scatter/segment-reduce); correctness is straightforward, the win is fusing the per-level injection. VLM differentiation lane alongside issues 38, 25.

**Refs.** Qwen3-VL DeepStack: https://thesalt.substack.com/p/qwen3-vl-deepstack-fusion-interleaved · technical report: https://arxiv.org/abs/2511.21631 · see issue 20.
