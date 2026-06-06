# 05 · Causal depthwise conv1d

**Why compile loses.** This is the cheap half of a Mamba block: a short-kernel (width 3–4) depthwise causal conv1d, optionally fused with a SiLU activation. It's closer to bandwidth-bound than the others here, so the win is smaller — but the fused kernel avoids the padding/slice copies and the separate activation pass that the PyTorch path incurs, and it matches the exact layout the Mamba scan expects. Useful as the companion to issue 03 and as a calibration point for "how much does fusion alone buy on a near-bandwidth op."

**Source.** `kernels-community/causal-conv1d`.

**Config sketch.** `HubConfig`, dtype fp16/bf16. `baseline` is `F.conv1d` with left padding `width-1`, grouped by channel (depthwise), sliced back to length L, followed by `F.silu` if the kernel fuses activation. `op` is the kernel's `causal_conv1d_fn` entry. Not `out_arg`. Inputs: `x` (B, C, L), `weight` (C, width), optional `bias` (C,), and an `activation` flag. Standard `verify` allclose at atol ~1e-2.

**Inputs to think about.** B=8, C=2048, L=2048, width=4 — the Mamba default.

**Difficulty.** Low-medium — simplest kernel in the backlog; mostly a layout/activation-flag match. Good warm-up before tackling the Mamba scan.

**Refs.** Mamba block (conv + scan); the `causal_conv1d` CUDA. Hub: https://huggingface.co/kernels-community/causal-conv1d
