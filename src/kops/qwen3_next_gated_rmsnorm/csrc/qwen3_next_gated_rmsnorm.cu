#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <cuda_bf16.h>

// Fused Qwen3NextRMSNormGated. Matches transformers Qwen3NextRMSNormGated.forward EXACTLY
// (norm-BEFORE-gate, PLAIN weight (not 1+w), fp32 reduce, silu(gate) in fp32):
//   h32   = hidden.float(); var = mean(h32^2, -1)
//   hn    = h32 * rsqrt(var + eps)
//   h     = weight * hn.to(bf16)              (bf16 * bf16 -> bf16)
//   out   = h * silu(gate.float())            (promote to fp32, silu in fp32)
//   return out.to(bf16)
//
// One warp per row (the Qwen3-Next gated-RMSNorm dim is head_v_dim=128, small), with several
// warps per block. Warp-shuffle reduction over H in fp32 — no shared memory, no block barriers.
template <int WARPS>
__global__ void gated_rmsnorm_kernel(const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ g,
                                     const __nv_bfloat16 *__restrict__ w, __nv_bfloat16 *__restrict__ out, int H,
                                     float eps, long M) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    long row = (long)blockIdx.x * WARPS + warp;
    if (row >= M)
        return;

    const __nv_bfloat16 *xr = x + row * (long)H;
    const __nv_bfloat16 *gr = g + row * (long)H;
    __nv_bfloat16 *outr = out + row * (long)H;

    float partial = 0.f;
    for (int j = lane; j < H; j += 32) {
        float v = __bfloat162float(xr[j]);
        partial += v * v;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        partial += __shfl_xor_sync(0xffffffff, partial, o);

    float r = rsqrtf(partial / H + eps);

    for (int j = lane; j < H; j += 32) {
        float hn = __bfloat162float(xr[j]) * r;
        __nv_bfloat16 hb = __float2bfloat16(hn); // hidden_states.to(input_dtype)
        float wj = __bfloat162float(w[j]);
        __nv_bfloat16 wh = __float2bfloat16(wj * __bfloat162float(hb)); // weight * hn  (bf16)
        float a = __bfloat162float(gr[j]);                              // gate.to(float32)
        float silu = a / (1.f + __expf(-a));
        outr[j] = __float2bfloat16(__bfloat162float(wh) * silu); // * silu(gate), then to bf16
    }
}

at::Tensor qwen3_next_gated_rmsnorm(at::Tensor x, at::Tensor gate, at::Tensor weight, double eps) {
    auto xc = x.contiguous();
    auto gc = gate.contiguous();
    int H = weight.size(0);
    long M = xc.numel() / H;
    auto out = at::empty_like(xc);
    const int WARPS = 8; // 256 threads/block, 8 rows/block
    long blocks = (M + WARPS - 1) / WARPS;
    gated_rmsnorm_kernel<WARPS><<<blocks, WARPS * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(gc.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(out.data_ptr<at::BFloat16>()), H, (float)eps, M);
    return out.view_as(x);
}
