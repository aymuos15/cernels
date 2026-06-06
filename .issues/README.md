# Advanced kernel issues

Backlog of kernels worth benchmarking against `torch.compile`. Rationale: our elementwise work (relu, gelu, silu_and_mul) confirmed that `torch.compile` already saturates memory bandwidth, so bandwidth-bound kernels can't beat it (we saw ~1.0–1.04x, and silu_and_mul was deleted for this reason). The only win so far — RoPE at 1.21x — came from fusion. Everything below is a place where compile structurally cannot compete: it's either compute-bound (quantized matmul, attention tiling, MoE grouped GEMM) or it hits one of compile's known failure modes.

Each issue is self-contained: rationale, source repo (or "write our own"), a Config sketch against the `Config`/`HubConfig` contract in `src/configs/base.py`, difficulty, and references. Implementing one means adding a config under `src/configs/registry/` (custom CUDA goes in `src/kops/`), registering it in `CONFIGS`, then `uv run --no-sync python -m benchmark.main <name>`. The `lib`/`baseline` can be any callable, so non-Hub ops (torchvision, our own kernels) fit too.

## Where torch.compile structurally loses (the thesis)
1. **Data-dependent output shape** — output length depends on tensor *values* (NMS, `unique`, `nonzero`, top-k-with-threshold) → unbacked SymInt → graph break / `GuardOnDataDependentSymNode`.
2. **Data-dependent control flow** — `if tensor:` / `.item()` / tensor-bounded loops → graph splinters, falls back to eager.
3. **Data-dependent loop counts** — chunked MoE / varlen → Dynamo unrolls or pads, killing the benefit.
4. **Quantized/sparse cores** — compile fuses *around* a quant or sparse matmul but never *into* it.
5. **Small-batch / decode** — at batch 1–64 compile is often ~2× slower than eager; decode-time ops are fair game.
6. **Optimizer / training-only ops** — a regime compile barely touches.

## Hub kernels (01–09)
| # | issue | source | track |
|---|---|---|---|
| 01 | [paged-attention](01-paged-attention.md) | `kernels-community/paged-attention` | Hub |
| 02 | [sage-attention](02-sage-attention.md) | `kernels-community/sage-attention` | Hub |
| 03 | [mamba-ssm](03-mamba-ssm.md) | `kernels-community/mamba-ssm` | Hub |
| 04 | [rwkv](04-rwkv.md) | `kernels-community/rwkv` | Hub |
| 05 | [causal-conv1d](05-causal-conv1d.md) | `kernels-community/causal-conv1d` | Hub |
| 06 | [megablocks-moe](06-megablocks-moe.md) | `kernels-community/megablocks` | Hub |
| 07 | [fp8-quant-matmul](07-fp8-quant-matmul.md) | `kernels-community/finegrained-fp8` | Hub |
| 08 | [gptq-quant](08-gptq-quant.md) | `kernels-community/quantization-gptq` | Hub |
| 09 | [custom-fused-rmsnorm](09-custom-fused-rmsnorm.md) | write our own (`src/kops/`) | custom |

## Recent research ops (10–16) — state-of-the-art, mostly not on the Hub
| # | issue | failure mode | hardware |
|---|---|---|---|
| 10 | [dsa-lightning-indexer](10-dsa-lightning-indexer.md) | #1 select-then-compute | any |
| 11 | [native-sparse-attention](11-native-sparse-attention.md) | #1 block select + gating | any |
| 12 | [minimax-block-sparse](12-minimax-block-sparse.md) | #1 block select | any |
| 13 | [mxfp4-gemm](13-mxfp4-gemm.md) | #4 quant core | **Blackwell/FP4 only** |
| 14 | [sage-attention3-fp4](14-sage-attention3-fp4.md) | #4 quant attention | **Blackwell/FP4 only** |
| 15 | [muon-newton-schulz](15-muon-newton-schulz.md) | #6 optimizer step | any |
| 16 | [fused-linear-cross-entropy](16-fused-linear-cross-entropy.md) | memory fusion | any |

## Compile-hostile ops (17–25) — known graph-break / fallback patterns
| # | issue | failure mode |
|---|---|---|
| 17 | [nms](17-nms.md) | #1 data-dependent count (current trial) |
| 18 | [topk-gather-select](18-topk-gather-select.md) | #1 the reusable select-then-compute primitive |
| 19 | [unique-nonzero-compaction](19-unique-nonzero-compaction.md) | #1 unbacked SymInt |
| 20 | [scatter-segment-reduce](20-scatter-segment-reduce.md) | #2 ragged + atomics |
| 21 | [sort-argsort-cumsum](21-sort-argsort-cumsum.md) | weak lowering, shape-specialize |
| 22 | [sparse-spmm-blocksparse](22-sparse-spmm-blocksparse.md) | #4 sparse layout |
| 23 | [ragged-varlen-bmm](23-ragged-varlen-bmm.md) | #3 data-dependent loop count |
| 24 | [sequence-decode-ctc-beam](24-sequence-decode-ctc-beam.md) | #2 control flow |
| 25 | [deformable-attention-roialign](25-deformable-attention-roialign.md) | #1 irregular gather (vision) |

## 2026 frontier model ops (26–40) — highest priority, from the current SOTA models
Ops pulled from a deep dive into the leading 2026 open models: **Gemma 4** (Apr 2026), **DeepSeek V4** (Apr 2026), **Qwen 3.5**, plus Qwen3-Next, Qwen3-VL, Llama 4, gpt-oss. These are the highest-impact targets — the same ops recur across models, so one kernel pays off many times, and several (CSA/HCA/mHC/GDN) have no open kernel yet.

| # | issue | models | why it matters | priority |
|---|---|---|---|---|
| 26 | [fused-qknorm-rope-attention](26-fused-qknorm-rope-attention.md) | Gemma 4, Qwen 3.5, Qwen3 | near-universal attention prologue | ⭐ |
| 27 | [sliding-window-attention](27-sliding-window-attention.md) | Gemma 4, gpt-oss | majority of layers in leading models | ⭐ |
| 28 | [csa-compressed-sparse-attention](28-csa-compressed-sparse-attention.md) | DeepSeek V4 | 1M ctx @ ~2% KV; **no open kernel yet** | ⭐⭐ |
| 29 | [hca-heavily-compressed-attention](29-hca-heavily-compressed-attention.md) | DeepSeek V4 | long-range half of V4; **no open kernel yet** | ⭐⭐ |
| 30 | [mla-multi-latent-attention](30-mla-multi-latent-attention.md) | DeepSeek V4/V3.2 | 5–8× KV bandwidth cut (`flash-mla` on Hub) | high |
| 31 | [cross-layer-shared-kv-attention](31-cross-layer-shared-kv-attention.md) | Gemma 4 | KV-cache memory cut, novel | high |
| 32 | [dual-rope](32-dual-rope.md) | Gemma 4 | extends our existing RoPE win | med |
| 33 | [mhc-hyper-connections](33-mhc-hyper-connections.md) | DeepSeek V4 | replaces residual add, every layer | med |
| 34 | [gdn-global-dense-norm](34-gdn-global-dense-norm.md) | Qwen 3.5 | MoE router front-end | med |
| 35 | [finegrained-moe](35-finegrained-moe.md) | V4, Qwen 3.5, gpt-oss, Llama 4 | **every 2026 model is MoE** | ⭐⭐ |
| 36 | [gated-deltanet](36-gated-deltanet.md) | Qwen3-Next | SOTA linear-attention scan | high |
| 37 | [attention-sinks](37-attention-sinks.md) | gpt-oss | flash softmax denominator term | med |
| 38 | [interleaved-mrope](38-interleaved-mrope.md) | Qwen3-VL, Qwen 3.5 | VLM differentiation lane | high (VLM) |
| 39 | [mxfp4-moe-matmul](39-mxfp4-moe-matmul.md) | gpt-oss, V4-Flash | FP4 × MoE, **Blackwell-gated** | high (hw) |
| 40 | [deepstack-vision-injection](40-deepstack-vision-injection.md) | Qwen3-VL | multi-level ViT feature scatter | med (VLM) |

**Priority tier (⭐⭐):** CSA (28), HCA (29), fine-grained MoE (35) — biggest impact and least-kernelized; CSA/HCA put us genuinely state-of-the-art rather than chasing. Then the near-universal attention pair 26/27.

Deferred for now (not in this backlog): flash-attn2/3/4, flash-linear-attention, deep-gemm.
