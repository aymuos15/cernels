# 04 · RWKV WKV kernel

**Why compile loses.** RWKV's WKV operator is a numerically-stabilized linear-attention recurrence carried over the time dimension. The fused kernel keeps the running max/numerator/denominator in registers across timesteps; a PyTorch baseline must loop over time and write each state to HBM. `torch.compile` cannot reschedule the sequential recurrence into a fused single-pass kernel, so the kernel wins on both memory traffic and launch overhead.

**Source.** `kernels-community/rwkv`.

**Config sketch.** `HubConfig`, dtype fp16/fp32 (WKV often wants fp32 accumulation for the exp-stabilization). `baseline` is a reference time-loop implementing the stabilized WKV recurrence (track `aa`, `bb`, `pp` running state). `op` is the kernel's WKV entry. Not `out_arg`. Inputs: time-decay `w` (C,), bonus `u` (C,), keys `k` and values `v` (B, T, C). Standard `verify` allclose; if fp16, loosen to atol ~3e-2 because of the exp accumulation.

**Inputs to think about.** B=8, T=1024–4096, C=2048. Longer T widens the gap vs the loop.

**Difficulty.** Medium-high — the stabilized recurrence is easy to get subtly wrong in the baseline; match the kernel's exact `u`/`w` placement and accumulation order. Check which RWKV version (v4/v5/v6) the Hub kernel targets and mirror that math.

**Refs.** RWKV papers/repo (BlinkDL); the original `wkv` CUDA. Hub: https://huggingface.co/kernels-community/rwkv
