#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

// Fused SwiGLU activation: out = silu(gate) * up, elementwise (bf16 in/out, fp32 math).
// silu(x) = x * sigmoid(x). Replaces F.silu(gate) * up (two launches) with one.
__global__ void silu_mul_kernel(const __nv_bfloat16 *__restrict__ g, const __nv_bfloat16 *__restrict__ u,
                                __nv_bfloat16 *__restrict__ o, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float a = __bfloat162float(g[i]);
    float b = __bfloat162float(u[i]);
    float s = a / (1.f + __expf(-a));
    o[i] = __float2bfloat16(s * b);
}

at::Tensor silu_and_mul(at::Tensor gate, at::Tensor up) {
    auto g = gate.contiguous();
    auto u = up.contiguous();
    auto o = at::empty_like(g);
    long n = g.numel();
    int threads = 256;
    long blocks = (n + threads - 1) / threads;
    silu_mul_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16 *>(g.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(u.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(o.data_ptr<at::BFloat16>()), n);
    return o.view_as(gate);
}
