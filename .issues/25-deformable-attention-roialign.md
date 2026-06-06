# 25 · Deformable attention / RoIAlign (vision gathers)

**Why compile loses.** Vision ops built on **irregular, value-dependent gathers**: deformable attention (DETR/Deformable-DETR) samples features at learned, data-dependent offset locations with bilinear interpolation; RoIAlign pools over data-dependent box regions. The sampling locations come from tensor values, so `torch.compile` can't fuse the gather + bilinear interp and falls back. A fused kernel does sample-offset → bilinear gather → weighted sum in one pass. Same "irregular gather" family as NMS, in the spatial domain.

**Source.** Write our own; `lib` baseline = `torchvision.ops.roi_align` / the reference deformable-attention `multi_scale_deformable_attn`.

**Config sketch.** `Config` (non-Hub), dtype fp16/fp32. Pick one per trial. For deformable attention: `baseline` = the reference PyTorch `grid_sample`-based implementation; `custom` = a fused MSDA kernel. Inputs: multi-scale value maps, sampling offsets (data-dependent), attention weights, spatial shapes. For RoIAlign: `baseline` = `torchvision.ops.roi_align`; inputs = feature map + boxes. `verify` allclose at atol ~1e-2.

**Inputs to think about.** Deformable: 4 feature levels, 8 heads, 4 sampling points, ~10k queries, channels 256. RoIAlign: 1k–10k boxes, output 7×7, channels 256.

**Difficulty.** Medium — bilinear-sample-and-accumulate kernel; the reference math (especially MSDA offset normalization across levels) is the fiddly part. Pairs thematically with issue 17 (NMS) as the vision/detection lane.

**Refs.** Deformable DETR (multi-scale deformable attention); torchvision RoIAlign CUDA.
