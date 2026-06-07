#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <cuda_fp16.h>

// Fuses the two SAM decomposed-rel-pos einsums and the broadcast bias add:
//   rel_h[b,h,w,kh] = sum_c query[b,h,w,c] * Rh[h,kh,c]
//   rel_w[b,h,w,kw] = sum_c query[b,h,w,c] * Rw[w,kw,c]
//   out[b, h*W+w, kh*KW+kw] = attn[...] + rel_h[b,h,w,kh] + rel_w[b,h,w,kw]
// One thread per output element (B, q_h*q_w, k_h*k_w). Contractions accumulate in fp32.
// query: (B, q_h*q_w, C); Rh: (q_h, k_h, C); Rw: (q_w, k_w, C); attn: (B, q_h*q_w, k_h*k_w).
__global__ void sam_rel_pos_kernel(const __half *__restrict__ query, const __half *__restrict__ Rh,
                                   const __half *__restrict__ Rw, const __half *__restrict__ attn,
                                   __half *__restrict__ out, int B, int QH, int QW, int KH, int KW, int C) {
    long gid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long Q = (long)QH * QW; // query positions
    long K = (long)KH * KW; // key positions
    long total = (long)B * Q * K;
    if (gid >= total)
        return;

    int kw = (int)(gid % KW);
    int kh = (int)((gid / KW) % KH);
    long qpos = (gid / K) % Q; // h*QW + w
    int b = (int)(gid / (K * Q));
    int w = (int)(qpos % QW);
    int h = (int)(qpos / QW);

    const __half *q = query + ((long)b * Q + qpos) * C;
    const __half *rh = Rh + ((long)h * KH + kh) * C;
    const __half *rw = Rw + ((long)w * KW + kw) * C;

    float rel_h = 0.f, rel_w = 0.f;
    for (int c = 0; c < C; ++c) {
        float qc = __half2float(q[c]);
        rel_h += qc * __half2float(rh[c]);
        rel_w += qc * __half2float(rw[c]);
    }
    out[gid] = __float2half(__half2float(attn[gid]) + rel_h + rel_w);
}

at::Tensor sam_decomposed_rel_pos(at::Tensor query, at::Tensor Rh, at::Tensor Rw, at::Tensor attn) {
    auto qc = query.contiguous();
    auto rhc = Rh.contiguous();
    auto rwc = Rw.contiguous();
    auto ac = attn.contiguous();
    int B = qc.size(0);
    int C = qc.size(2);
    int QH = rhc.size(0), KH = rhc.size(1);
    int QW = rwc.size(0), KW = rwc.size(1);
    auto out = at::empty_like(ac);
    long total = (long)B * QH * QW * KH * KW;
    int threads = 256;
    long blocks = (total + threads - 1) / threads;
    sam_rel_pos_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half *>(qc.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(rhc.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(rwc.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(ac.data_ptr<at::Half>()), reinterpret_cast<__half *>(out.data_ptr<at::Half>()),
        B, QH, QW, KH, KW, C);
    return out;
}
