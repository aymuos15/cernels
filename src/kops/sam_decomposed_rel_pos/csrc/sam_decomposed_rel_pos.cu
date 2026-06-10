#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

// SAM decomposed-rel-pos bias, one block per (b, qpos) query row:
//   rel_h[kh] = sum_c query[b,qpos,c] * Rh[h,kh,c]   (qpos = h*QW + w)
//   rel_w[kw] = sum_c query[b,qpos,c] * Rw[w,kw,c]
//   out[b, qpos, kh*KW+kw] = (attn ? attn[...] : 0) + rel_h[kh] + rel_w[kw]
// The q row is staged in shared memory and each of the KH+KW dot products is computed
// ONCE per row (vs once per output element in the naive fusion), so read traffic no
// longer scales with k_h*k_w — the GEMM-equivalent work is B*Q*(KH+KW)*C MACs, exactly
// the einsums' FLOPs, and the broadcast add is fused into the same launch. With attn
// omitted the kernel writes the SDPA attn_mask bias directly (no zeros materialization).
// I/O fp16/bf16 (dispatched); dots and the add accumulate in fp32.
// query: (B, QH*QW, C); Rh: (QH, KH, C); Rw: (QW, KW, C); attn/out: (B, QH*QW, KH*KW).
template <typename scalar_t>
__global__ void sam_rel_pos_bias_kernel(const scalar_t *__restrict__ query, const scalar_t *__restrict__ Rh,
                                        const scalar_t *__restrict__ Rw, const scalar_t *__restrict__ attn,
                                        scalar_t *__restrict__ out, int QH, int QW, int KH, int KW, int C) {
    extern __shared__ float smem[]; // [C] q row | [KH] rel_h | [KW] rel_w
    float *qs = smem;
    float *rel = smem + C;

    long row = blockIdx.x; // b*QH*QW + h*QW + w
    int qpos = (int)(row % ((long)QH * QW));
    int w = qpos % QW, h = qpos / QW;

    const scalar_t *qrow = query + row * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        qs[c] = static_cast<float>(qrow[c]);
    __syncthreads();

    // 8-lane groups, one dot per group; the base loop trips uniformly across the block so
    // every warp lane reaches the full-mask shuffle together (no divergent __shfl_down_sync).
    constexpr int G = 8;
    int group = threadIdx.x / G, lane = threadIdx.x % G;
    int ngroups = blockDim.x / G;
    int ndots = KH + KW;
    for (int base = 0; base < ndots; base += ngroups) {
        int d = base + group;
        float acc = 0.f;
        if (d < ndots) {
            const scalar_t *r = d < KH ? Rh + ((long)h * KH + d) * C : Rw + ((long)w * KW + (d - KH)) * C;
            for (int c = lane; c < C; c += G)
                acc += qs[c] * static_cast<float>(r[c]);
        }
        for (int off = G / 2; off; off >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, off);
        if (lane == 0 && d < ndots)
            rel[d] = acc;
    }
    __syncthreads();

    long K = (long)KH * KW;
    const scalar_t *arow = attn ? attn + row * K : nullptr;
    scalar_t *orow = out + row * K;
    for (int k = threadIdx.x; k < (int)K; k += blockDim.x) {
        float v = rel[k / KW] + rel[KH + k % KW];
        if (arow)
            v += static_cast<float>(arow[k]);
        orow[k] = static_cast<scalar_t>(v);
    }
}

at::Tensor sam_decomposed_rel_pos(at::Tensor query, at::Tensor Rh, at::Tensor Rw, std::optional<at::Tensor> attn) {
    auto qc = query.contiguous();
    auto rhc = Rh.contiguous();
    auto rwc = Rw.contiguous();
    long B = qc.size(0);
    int C = (int)qc.size(2);
    int QH = rhc.size(0), KH = rhc.size(1);
    int QW = rwc.size(0), KW = rwc.size(1);
    at::Tensor ac;
    at::Tensor out;
    if (attn.has_value()) {
        ac = attn->contiguous();
        out = at::empty_like(ac);
    } else {
        out = at::empty({B, (long)QH * QW, (long)KH * KW}, qc.options());
    }
    long blocks = B * QH * QW;
    int threads = 256;
    int smem = (C + KH + KW) * (int)sizeof(float);
    AT_DISPATCH_REDUCED_FLOATING_TYPES(qc.scalar_type(), "sam_decomposed_rel_pos", [&] {
        sam_rel_pos_bias_kernel<scalar_t><<<blocks, threads, smem, at::cuda::getCurrentCUDAStream()>>>(
            qc.data_ptr<scalar_t>(), rhc.data_ptr<scalar_t>(), rwc.data_ptr<scalar_t>(),
            attn.has_value() ? ac.data_ptr<scalar_t>() : nullptr, out.data_ptr<scalar_t>(), QH, QW, KH, KW, C);
    });
    return out;
}
