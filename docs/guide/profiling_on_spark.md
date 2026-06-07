# Profiling on the Spark (GB10)

The GB10 is a UMA / iGPU Grace-Blackwell part, and the usual x86-desktop GPU-profiling instincts don't all transfer. These are the gotchas worth knowing before trusting any number.

## Treat wall-clock + tok/s as the main truth

On a UMA part a lot of the interesting perf questions are about host/device overlap, not raw kernel occupancy. Lead with timeline/throughput measurement — `nsys`, and for us `torch.profiler` and end-to-end `generate()` tok/s ([modelkernels](../../src/modelkernels/README.md)) — and only drop to kernel-counter analysis after you have a hot kernel and confirmed counters work. Our [profiler](../../src/profiling/README.md) is the `nsys`-first layer: a timeline of where time goes, with `torch.profiler` op self-times as the authoritative GPU view.

## Memory telemetry is unreliable (UMA)

Because GPU and CPU share memory, `nvidia-smi` can mislead and `cudaMemGetInfo()` / `torch.cuda.max_memory_allocated()` aren't the full story. Do **not** over-trust peak-VRAM numbers on the Spark — `modelkernels` reports `peak GB` only as a rough sanity figure, not a hard metric.

## `ncu` and hardware counters: `ERR_NVGPUCTRPERM`

`ncu` (and `nsys` GPU **hardware metrics**) can fail with `ERR_NVGPUCTRPERM` when hardware counters are permission-restricted — it looks like the whole profiler stack is broken when it's really a permissions gate. Plain `nsys` *tracing* still works without it. Fix: <https://developer.nvidia.com/ERR_NVGPUCTRPERM>. Confirm counters are actually enabled before relying on any `ncu` result.

## Arm SBSA Nsight workflow

The Spark uses the Arm SBSA Nsight workflow, which differs from the usual x86 desktop setup. NVIDIA's DGX Spark optimization guide is the reference: <https://docs.nvidia.com/dgx/dgx-spark-porting-guide/optimization.html>.

## Recommended order

1. `nsys` / `torch.profiler` timeline → find where wall-clock goes and which kernels are hot.
2. End-to-end `generate()` tok/s → the number that actually matters for a model.
3. Only then `ncu` on the specific hot kernels — after confirming the counter-permission gate is open.
