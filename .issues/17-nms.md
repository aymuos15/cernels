# 17 · Non-maximum suppression (NMS)

**Why compile loses.** The archetype of a compile-hostile op, and the one already being trialed. NMS keeps boxes greedily by IoU overlap — the **number of surviving boxes depends on the box values, not the input shape**. That makes the output an unbacked SymInt: `torch.compile` graph-breaks on it (or errors under `fullgraph=True` with `GuardOnDataDependentSymNode`) and falls back to eager. The greedy suppression is also inherently sequential/data-dependent control flow. A custom kernel does the pairwise-IoU + bitmask suppression in one pass with no graph break.

**Source.** Write our own; `lib` baseline = `torchvision.ops.nms` (a non-Hub production op — exactly the "lib can be anything" case the harness was designed for).

**Config sketch.** `Config` (non-Hub), dtype fp32. `baseline` = `torchvision.ops.nms(boxes, scores, iou_threshold)`. `custom` = our CUDA NMS (pairwise IoU into a per-block bitmask, then reduce). Inputs: `boxes` (N, 4) xyxy, `scores` (N,), `iou_threshold`. **Override `verify`** — output is a variable-length index tensor, so compare the *kept set* (as sorted index sets / a mask), not `allclose`.

**Inputs to think about.** N=2k, 8k, 30k boxes with realistic overlap (cluster boxes so suppression actually fires); iou_threshold 0.5/0.7. Also benchmark the all-overlap and no-overlap extremes.

**Difficulty.** Low-medium — the bitmask NMS kernel is well-known; the real work here is the `verify` for variable-length output and a fair baseline (compile the surrounding model, fence NMS out, show it's the break point).

**Refs.** torchvision NMS CUDA (bitmask algorithm) is the reference implementation to match and beat.
