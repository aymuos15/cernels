# 37 · Attention sinks

**Why compile loses.** gpt-oss gives each attention head a **learned bias in the softmax denominator** — an extra always-present "sink" logit so the head can choose to attend to (almost) nothing, stabilizing long-context and streaming. Mechanically it changes the flash-attention online-softmax normalization: the running denominator is seeded with `exp(sink_bias)` and never gets a matching numerator term. `torch.compile` calls an opaque SDPA that has no notion of this extra term, so the sink must be bolted on outside the (unfused) attention. A flash kernel folds the sink directly into the softmax accumulator at zero extra cost.

**Source.** Write our own (flash-attention with a per-head sink term); reference gpt-oss attention.

**Config sketch.** `Config` (non-Hub), dtype bf16. `baseline` = eager attention where the softmax denominator includes `exp(sink)` per head (concat a learned sink column to the logits before softmax, drop it from the value sum). `custom` = flash kernel seeding the denominator with the sink. Inputs: q/k/v (b, h, s, d), per-head `sink` (h,), causal flag. `verify` allclose at atol ~2e-2.

**Inputs to think about.** b=2, heads 32, head_dim 128, sequence 4k–32k, with the 128-token sliding window from gpt-oss (compose with issue 27). Sink matters most at long context / streaming.

**Difficulty.** Medium — a small, well-defined change to a flash kernel's softmax reduction; the main work is having a flash kernel to modify (build on issue 27). Composes cleanly with sliding-window (issue 27) and QK-norm prologue (issue 26).

**Refs.** gpt-oss attention sinks: https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the · StreamingLLM (original attention-sink observation).
