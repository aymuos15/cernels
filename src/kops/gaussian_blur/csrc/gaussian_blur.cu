#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

// Separable Gaussian blur with reflect-101 borders (matches kornia's border_type="reflect").
// Two passes over (NC, H, W): horizontal (along W) into a temp, then vertical (along H).
// One thread per output pixel; the 1D taps are read from kx/ky.

__device__ __forceinline__ int reflect101(int i, int n) {
    if (n == 1)
        return 0;
    while (i < 0 || i >= n) {
        if (i < 0)
            i = -i;
        if (i >= n)
            i = 2 * (n - 1) - i;
    }
    return i;
}

__global__ void hblur(const float *__restrict__ x, const float *__restrict__ kx, float *__restrict__ out, int nc, int h,
                      int w, int kw) {
    long gid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)nc * h * w;
    if (gid >= total)
        return;
    int wi = (int)(gid % w);
    long row = gid / w; // index over nc*h
    long base = row * w;
    int rad = kw / 2;
    float acc = 0.f;
    for (int t = 0; t < kw; ++t) {
        int ww = reflect101(wi + t - rad, w);
        acc += x[base + ww] * kx[t];
    }
    out[gid] = acc;
}

__global__ void vblur(const float *__restrict__ x, const float *__restrict__ ky, float *__restrict__ out, int nc, int h,
                      int w, int kh) {
    long gid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)nc * h * w;
    if (gid >= total)
        return;
    int wi = (int)(gid % w);
    long tmp = gid / w;
    int hi = (int)(tmp % h);
    long n = tmp / h; // image index over nc
    long plane = n * (long)h * w;
    int rad = kh / 2;
    float acc = 0.f;
    for (int t = 0; t < kh; ++t) {
        int hh = reflect101(hi + t - rad, h);
        acc += x[plane + (long)hh * w + wi] * ky[t];
    }
    out[gid] = acc;
}

at::Tensor gblur(at::Tensor x, at::Tensor ky, at::Tensor kx) {
    int b = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3);
    int nc = b * c;
    int kh = ky.size(0), kw = kx.size(0);
    auto tmp = at::empty_like(x);
    auto out = at::empty_like(x);
    long total = (long)nc * h * w;
    int threads = 256;
    long blocks = (total + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();
    hblur<<<blocks, threads, 0, stream>>>(x.data_ptr<float>(), kx.data_ptr<float>(), tmp.data_ptr<float>(), nc, h, w,
                                          kw);
    vblur<<<blocks, threads, 0, stream>>>(tmp.data_ptr<float>(), ky.data_ptr<float>(), out.data_ptr<float>(), nc, h, w,
                                          kh);
    return out;
}
