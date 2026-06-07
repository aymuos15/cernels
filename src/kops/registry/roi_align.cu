#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

// RoI Align (torchvision-compatible). One thread per output element (k, c, ph, pw).
// For each output bin, sample sampling_ratio^2 points via bilinear interpolation and
// average. Matches torchvision semantics (aligned offset, sampling grid, zero outside).
// fp32 throughout (config runs fp32 so torchvision computes in fp32 too — verifies cleanly;
// fp16 boxes/coords would diverge because torchvision does the coord math in the input dtype).

__device__ __forceinline__ float bilinear(const float *__restrict__ in, int H, int W, float y, float x) {
    // Outside the [-1, H] x [-1, W] window contributes zero (torchvision convention).
    if (y < -1.0f || y > (float)H || x < -1.0f || x > (float)W)
        return 0.0f;
    if (y <= 0.0f)
        y = 0.0f;
    if (x <= 0.0f)
        x = 0.0f;
    int yl = (int)y, xl = (int)x;
    int yh, xh;
    if (yl >= H - 1) {
        yh = yl = H - 1;
        y = (float)yl;
    } else
        yh = yl + 1;
    if (xl >= W - 1) {
        xh = xl = W - 1;
        x = (float)xl;
    } else
        xh = xl + 1;
    float ly = y - yl, lx = x - xl, hy = 1.0f - ly, hx = 1.0f - lx;
    return hy * hx * in[yl * W + xl] + hy * lx * in[yl * W + xh] + ly * hx * in[yh * W + xl] +
           ly * lx * in[yh * W + xh];
}

__global__ void roi_align_kernel(const float *__restrict__ input, const float *__restrict__ boxes,
                                 float *__restrict__ out, int N, int C, int H, int W, int K, int oh, int ow,
                                 float spatial_scale, int sampling_ratio, bool aligned) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)K * C * oh * ow;
    if (idx >= total)
        return;
    int pw = (int)(idx % ow);
    int ph = (int)((idx / ow) % oh);
    int c = (int)((idx / ((long)ow * oh)) % C);
    int k = (int)(idx / ((long)ow * oh * C));

    const float *box = boxes + (long)k * 5;
    int n = (int)box[0];
    float off = aligned ? 0.5f : 0.0f;
    float rsw = box[1] * spatial_scale - off;
    float rsh = box[2] * spatial_scale - off;
    float rw = box[3] * spatial_scale - off - rsw;
    float rh = box[4] * spatial_scale - off - rsh;
    if (!aligned) {
        rw = fmaxf(rw, 1.0f);
        rh = fmaxf(rh, 1.0f);
    }
    float bin_w = rw / ow, bin_h = rh / oh;

    int gh = sampling_ratio > 0 ? sampling_ratio : (int)ceilf(rh / oh);
    int gw = sampling_ratio > 0 ? sampling_ratio : (int)ceilf(rw / ow);
    float count = (float)(gh * gw);

    const float *in = input + ((long)n * C + c) * H * W;
    float acc = 0.0f;
    for (int iy = 0; iy < gh; ++iy) {
        float yy = rsh + ph * bin_h + (iy + 0.5f) * bin_h / gh;
        for (int ix = 0; ix < gw; ++ix) {
            float xx = rsw + pw * bin_w + (ix + 0.5f) * bin_w / gw;
            acc += bilinear(in, H, W, yy, xx);
        }
    }
    out[idx] = acc / count;
}

at::Tensor roi_align(at::Tensor input, at::Tensor boxes, int64_t output_size, double spatial_scale,
                     int64_t sampling_ratio, bool aligned) {
    int N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    int K = boxes.size(0);
    int oh = (int)output_size, ow = (int)output_size;
    auto out = at::empty({K, C, oh, ow}, input.options());

    long total = (long)K * C * oh * ow;
    int threads = 256;
    long blocks = (total + threads - 1) / threads;
    roi_align_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), boxes.data_ptr<float>(), out.data_ptr<float>(), N, C, H, W, K, oh, ow,
        (float)spatial_scale, (int)sampling_ratio, aligned);
    return out;
}
