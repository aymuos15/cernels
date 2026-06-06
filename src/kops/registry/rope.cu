#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

// One thread per rotation pair (j, j+d/2) of one (batch, head, seq) row, for q and k.
// cos/sin are [b, s, d] (broadcast over heads) with duplicated halves, so we read the
// first half only: out[j] = x_j*c - x_{j+h}*s ; out[j+h] = x_{j+h}*c + x_j*s.
__global__ void rope_kernel(const __half *__restrict__ q, const __half *__restrict__ k, const __half *__restrict__ cs,
                            const __half *__restrict__ sn, __half *__restrict__ qo, __half *__restrict__ ko,
                            long n_pairs, int d, int s, int nh) {
    long gid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (gid >= n_pairs)
        return;
    int half = d >> 1;
    int j = (int)(gid % half);
    long row = gid / half; // index over b*nh*s
    int s_idx = (int)(row % s);
    long b_idx = row / ((long)nh * s);
    long xb = row * d + j;                       // x[b, head, s, j]
    long cb = (b_idx * (long)s + s_idx) * d + j; // cos/sin[b, s, j]

    float c = __half2float(cs[cb]), si = __half2float(sn[cb]);
    float q1 = __half2float(q[xb]), q2 = __half2float(q[xb + half]);
    float k1 = __half2float(k[xb]), k2 = __half2float(k[xb + half]);
    qo[xb] = __float2half(q1 * c - q2 * si);
    qo[xb + half] = __float2half(q2 * c + q1 * si);
    ko[xb] = __float2half(k1 * c - k2 * si);
    ko[xb + half] = __float2half(k2 * c + k1 * si);
}

std::tuple<at::Tensor, at::Tensor> rope(at::Tensor q, at::Tensor k, at::Tensor cos, at::Tensor sin) {
    auto qo = at::empty_like(q);
    auto ko = at::empty_like(k);
    int b = q.size(0), nh = q.size(1), s = q.size(2), d = q.size(3);
    long n_pairs = (long)b * nh * s * (d / 2);
    int threads = 256;
    long blocks = (n_pairs + threads - 1) / threads;
    rope_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half *>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(cos.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(sin.data_ptr<at::Half>()), reinterpret_cast<__half *>(qo.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(ko.data_ptr<at::Half>()), n_pairs, d, s, nh);
    return {qo, ko};
}
