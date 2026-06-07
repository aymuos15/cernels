#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <cuda_bf16.h>

// One block per row; threads cooperatively reduce sum of squares over the H dim in fp32,
// then each thread writes its strided columns. Matches Lfm2RMSNorm exactly: normalize in
// fp32, cast to bf16, then multiply by the bf16 weight.
template <int BLOCK>
__global__ void rmsnorm_kernel(const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ w,
                               __nv_bfloat16 *__restrict__ out, int H, float eps) {
    long row = blockIdx.x;
    const __nv_bfloat16 *xr = x + row * (long)H;
    __nv_bfloat16 *outr = out + row * (long)H;

    float partial = 0.f;
    for (int j = threadIdx.x; j < H; j += BLOCK) {
        float v = __bfloat162float(xr[j]);
        partial += v * v;
    }
    __shared__ float sdata[BLOCK];
    sdata[threadIdx.x] = partial;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    __shared__ float rinv;
    if (threadIdx.x == 0)
        rinv = rsqrtf(sdata[0] / H + eps);
    __syncthreads();

    float r = rinv;
    for (int j = threadIdx.x; j < H; j += BLOCK) {
        float hn = __bfloat162float(xr[j]) * r;
        __nv_bfloat16 hb = __float2bfloat16(hn); // round to bf16 (== hidden_states.to(dtype))
        float wj = __bfloat162float(w[j]);
        outr[j] = __float2bfloat16(wj * __bfloat162float(hb));
    }
}

at::Tensor rmsnorm(at::Tensor x, at::Tensor weight, double eps) {
    auto xc = x.contiguous();
    int H = weight.size(0);
    long M = xc.numel() / H;
    auto out = at::empty_like(xc);
    const int BLOCK = 256;
    rmsnorm_kernel<BLOCK><<<M, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16 *>(xc.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(out.data_ptr<at::BFloat16>()), H, (float)eps);
    return out.view_as(x);
}
