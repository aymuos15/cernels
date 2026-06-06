# 03 · Mamba selective scan

**Why compile loses.** The selective-scan (SSM) core is a sequential recurrence with input-dependent gates. The hardware-aware kernel runs it as a parallel scan in SRAM, fusing the discretization, scan, and gating into one pass. A naive PyTorch baseline is an explicit time-loop that materializes every intermediate state to HBM; `torch.compile` can fuse pointwise work inside the loop but cannot turn the sequential recurrence into a work-efficient parallel scan. Large structural win.

**Source.** `kernels-community/mamba-ssm` (the `selective_scan` kernel).

**Config sketch.** `HubConfig`, dtype fp16/bf16. `baseline` is a reference Python selective-scan: a `for t in range(L)` recurrence `state = state*dA + dB*x; y = (state*C).sum(...)`, matching the kernel's signature. `op` is the kernel's `selective_scan_fn` entry. Not `out_arg`. Inputs: `u` (B, D, L), discretization `delta` (B, D, L), state matrices `A` (D, N), `B`/`C` (B, N, L), and `D`/`z` gates. Standard `verify` allclose should work at atol ~2e-2.

**Inputs to think about.** B=8, D=2048 (model dim), N=16 (state size), L=2048 (sequence). The long-sequence regime is where the parallel scan crushes the loop.

**Difficulty.** High — getting the reference recurrence to match the kernel's exact discretization and gate conventions is fiddly; budget most of the time for the baseline, not the kernel call. Pairs naturally with issue 05 (causal-conv1d), the other half of a Mamba block.

**Refs.** Mamba paper (Gu & Dao); the original `selective_scan` CUDA. Hub: https://huggingface.co/kernels-community/mamba-ssm
