# 24 · Sequence decode (CTC / beam search / Viterbi)

**Why compile loses.** These decoders are inherently **sequential, branchy, and value-dependent**: beam search prunes hypotheses by score each step, CTC/Viterbi run dynamic-programming recurrences with data-dependent transitions. `torch.compile` graph-breaks on the per-step control flow and the `.item()`/argmax-driven branching, leaving the whole loop in eager Python. A fused kernel keeps the DP state on-device across steps and avoids the per-step CPU sync.

**Source.** Write our own; `lib` baseline = `torchaudio`'s CTC / a reference beam-search loop, or `torch` DP in Python.

**Config sketch.** `Config` (non-Hub), dtype fp32 (DP wants the dynamic range). Pick one decoder per trial. `baseline` = the eager Python DP/beam loop. `custom` = a kernel running the recurrence on-device (one block per batch item / per beam). Inputs: log-probs (T, batch, vocab), and decoder params (beam width, blank id). **Override `verify`** — output is a variable-length token sequence; compare decoded sequences (and scores) for equality, not `allclose`.

**Inputs to think about.** T=500–2000 timesteps, vocab=128–32k, batch=32, beam=4–16. The per-step CPU sync in the eager baseline is the cost you're removing — measure wall-clock including sync.

**Difficulty.** High — DP recurrence + variable-length output + on-device beam management; the most control-flow-heavy issue in the backlog. Start with greedy/CTC argmax before beam search.

**Refs.** PyTorch data-dependent control-flow graph breaks: https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html
