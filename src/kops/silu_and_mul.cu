#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

// Gated activation: out[..., j] = silu(x[..., j]) * x[..., j + d], input last dim 2*d.
// Memory-bound, so each thread processes 8 contiguous outputs with vectorized 16-byte
// (int4 = 8x half) loads/stores. Assumes d is a multiple of 8 (true for typical hidden sizes).
constexpr int V = 8;

__global__ void silu_and_mul_kernel(const __half *__restrict__ x, __half *__restrict__ out, long n_vec, int d) {
    long vid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (vid >= n_vec)
        return;
    long base = vid * V;          // first output element handled by this thread
    int j = (int)(base % d);      // column within the output row
    long row = base / d;          // output row
    long xb = row * (2L * d) + j; // a-half at xb, b-half at xb + d

    int4 av = *reinterpret_cast<const int4 *>(x + xb);
    int4 bv = *reinterpret_cast<const int4 *>(x + xb + d);
    int4 ov;
    const __half *a = reinterpret_cast<const __half *>(&av);
    const __half *b = reinterpret_cast<const __half *>(&bv);
    __half *o = reinterpret_cast<__half *>(&ov);
#pragma unroll
    for (int t = 0; t < V; t++) {
        float af = __half2float(a[t]);
        o[t] = __float2half(af / (1.f + __expf(-af)) * __half2float(b[t]));
    }
    *reinterpret_cast<int4 *>(out + base) = ov;
}

at::Tensor silu_and_mul(at::Tensor x) {
    auto sizes = x.sizes().vec();
    int d = sizes.back() / 2;
    sizes.back() = d;
    auto out = at::empty(sizes, x.options());
    long n_vec = out.numel() / V;
    int threads = 256;
    long blocks = (n_vec + threads - 1) / threads;
    silu_and_mul_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half *>(x.data_ptr<at::Half>()), reinterpret_cast<__half *>(out.data_ptr<at::Half>()),
        n_vec, d);
    return out;
}
