#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <cuda_bf16.h>

// Qwen3-Next Gated DeltaNet — chunked delta-rule linear attention.
//
// One CUDA block per (batch, head). The block marches the sequence chunk by chunk,
// keeping the cross-chunk recurrent state S (DK x DV, fp32) resident in SHARED memory.
// GB10 caps opt-in dynamic shared memory at ~99 KB; S alone is 64 KB, so we tile with a
// small CHUNK=16 and alias the v-tile/v_new buffer to fit the chunk workspaces alongside S.
//
// Everything below transcribes torch_chunk_gated_delta_rule from transformers'
// modeling_qwen3_next.py exactly (use_qk_l2norm_in_kernel=True, scale=1/sqrt(dk)),
// all math in fp32. The delta-rule result is algebraically independent of the chunk
// size used to evaluate it, so we tile with CHUNK=16 internally (smaller shared
// footprint) even though the public op advertises chunk_size=64; results match the
// chunk=64 reference within the bf16 tolerance.
//
//   per step the q/k rows are l2-normalised; q is scaled by 1/sqrt(dk).
//   g is cumulative-summed within the chunk.
//   A[a,b]  = -(k_beta[a]·k[b]) * exp(g[a]-g[b])  for a>b else 0   (strictly lower)
//   forward-substitution makes T = (I - A)^{-1} (unit lower-tri), stored back in A (+I).
//   w        = T @ v_beta              ;  kcd = T @ (k_beta * exp(g))
//   then, carrying S (dk x dv):
//     attn2     = (q @ k^T) * decay     (lower-tri, diag incl)
//     v_prime   = kcd @ S
//     v_new     = w - v_prime
//     attn_inter= (q * exp(g)) @ S
//     out       = attn_inter + attn2 @ v_new
//     S         = S*exp(g[C-1]) + (k * exp(g[C-1]-g))^T @ v_new

#define CHUNK 16
#define DK 128
#define DV 128
#define NTHREADS 128

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v, float *smem) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0)
        smem[wid] = v;
    __syncthreads();
    float r = (threadIdx.x < 4) ? smem[threadIdx.x] : 0.f;
    if (wid == 0)
        r = warp_reduce_sum(r);
    if (threadIdx.x == 0)
        smem[0] = r;
    __syncthreads();
    float out = smem[0];
    __syncthreads();
    return out;
}

extern __shared__ float shmem[];

// Shared layout (fp32):
//   S_     : DK*DV      (state, persists across chunks)
//   qs,ks  : CHUNK*DK each
//   vws    : CHUNK*DV   (v tile -> w = T @ v_beta -> v_new, all in place)
//   kcd    : CHUNK*DK
//   gcum   : CHUNK
//   bes    : CHUNK
//   A      : CHUNK*CHUNK
//   red    : 4
__global__ void deltanet_kernel(const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
                                const __nv_bfloat16 *__restrict__ v, const __nv_bfloat16 *__restrict__ g,
                                const __nv_bfloat16 *__restrict__ beta, __nv_bfloat16 *__restrict__ out, int B, int H,
                                int S, int n_chunks) {
    int bh = blockIdx.x; // 0 .. B*H-1
    int b = bh / H, h = bh % H;
    int t = threadIdx.x;

    float *S_ = shmem;
    float *qs = S_ + DK * DV;
    float *ks = qs + CHUNK * DK;
    float *vws = ks + CHUNK * DK; // v tile, then w, then v_new (aliased)
    float *kcd = vws + CHUNK * DV;
    float *gcum = kcd + CHUNK * DK;
    float *bes = gcum + CHUNK;
    float *A = bes + CHUNK;
    float *red = A + CHUNK * CHUNK;

    const float scale = rsqrtf((float)DK);

    long base_qkv = ((long)b * S * H + (long)h) * DK; // (b,0,h,0); add s*H*DK
    long base_g = ((long)b * S * H + (long)h);        // (b,0,h);   add s*H

    // zero state
    for (int i = t; i < DK * DV; i += NTHREADS)
        S_[i] = 0.f;
    __syncthreads();

    for (int c = 0; c < n_chunks; ++c) {
        int s0 = c * CHUNK;
        // ---- load chunk: l2-normalise q,k rows; scale q; load v,beta ----
        for (int r = 0; r < CHUNK; ++r) {
            int s = s0 + r;
            bool valid = (s < S);
            const __nv_bfloat16 *qr = q + base_qkv + (long)s * H * DK;
            const __nv_bfloat16 *kr = k + base_qkv + (long)s * H * DK;
            const __nv_bfloat16 *vr = v + base_qkv + (long)s * H * DV;
            float qv = (valid && t < DK) ? __bfloat162float(qr[t]) : 0.f;
            float kv = (valid && t < DK) ? __bfloat162float(kr[t]) : 0.f;
            float qn = block_reduce_sum(qv * qv, red);
            float kn = block_reduce_sum(kv * kv, red);
            float qrs = rsqrtf(qn + 1e-6f);
            float krs = rsqrtf(kn + 1e-6f);
            if (t < DK) {
                qs[r * DK + t] = qv * qrs * scale;
                ks[r * DK + t] = kv * krs;
                vws[r * DV + t] = (valid) ? __bfloat162float(vr[t]) : 0.f;
            }
            if (t == 0)
                bes[r] = valid ? __bfloat162float(beta[base_g + (long)s * H]) : 0.f;
        }
        // gate cumsum within chunk (thread 0); padded steps contribute 0.
        if (t == 0) {
            float acc = 0.f;
            for (int r = 0; r < CHUNK; ++r) {
                int s = s0 + r;
                float gv = (s < S) ? __bfloat162float(g[base_g + (long)s * H]) : 0.f;
                acc += gv;
                gcum[r] = acc;
            }
        }
        __syncthreads();

        // ---- A[a,b] = -(k_beta[a]·k[b]) * exp(g[a]-g[b]) for a>b else 0 ----
        for (int idx = t; idx < CHUNK * CHUNK; idx += NTHREADS) {
            int a = idx / CHUNK, bb = idx % CHUNK;
            float val = 0.f;
            if (a > bb) {
                float dot = 0.f;
                for (int d = 0; d < DK; ++d)
                    dot += ks[a * DK + d] * bes[a] * ks[bb * DK + d];
                val = -(dot * __expf(gcum[a] - gcum[bb]));
            }
            A[idx] = val;
        }
        __syncthreads();

        // ---- forward substitution to T=(I-A)^{-1}; then +I ----
        for (int i = 1; i < CHUNK; ++i) {
            float newval = 0.f;
            bool active = (t < i);
            if (active) {
                float acc = A[i * CHUNK + t];
                for (int j = t + 1; j < i; ++j)
                    acc += A[i * CHUNK + j] * A[j * CHUNK + t];
                newval = acc;
            }
            __syncthreads();
            if (active)
                A[i * CHUNK + t] = newval;
            __syncthreads();
        }
        for (int idx = t; idx < CHUNK; idx += NTHREADS)
            A[idx * CHUNK + idx] += 1.f;
        __syncthreads();

        // ---- w = T @ v_beta (in vws) ; kcd = T @ (k_beta*exp(g)) ----
        if (t < DV) {
            // compute both, but w overwrites vws[r] which is also a source -> stage in regs.
            float wcol[CHUNK];
            float kcol[CHUNK];
            for (int a = 0; a < CHUNK; ++a) {
                float accw = 0.f, acck = 0.f;
                for (int r = 0; r < CHUNK; ++r) {
                    float aval = A[a * CHUNK + r];
                    accw += aval * vws[r * DV + t] * bes[r];
                    acck += aval * ks[r * DK + t] * bes[r] * __expf(gcum[r]);
                }
                wcol[a] = accw;
                kcol[a] = acck;
            }
            for (int a = 0; a < CHUNK; ++a) {
                vws[a * DV + t] = wcol[a];
                kcd[a * DK + t] = kcol[a];
            }
        }
        __syncthreads();

        // ---- v_prime = kcd @ S ; v_new = w - v_prime (overwrite vws in place) ----
        if (t < DV) {
            float vn[CHUNK];
            for (int a = 0; a < CHUNK; ++a) {
                float vp = 0.f;
                for (int d = 0; d < DK; ++d)
                    vp += kcd[a * DK + d] * S_[d * DV + t];
                vn[a] = vws[a * DV + t] - vp;
            }
            for (int a = 0; a < CHUNK; ++a)
                vws[a * DV + t] = vn[a];
        }
        __syncthreads();
        float *vnew = vws; // alias: vws now holds v_new

        // ---- precompute attn2[a,bb] = (q[a]·k[bb]) * exp(g[a]-g[bb]) for a>=bb (reuse A) ----
        for (int idx = t; idx < CHUNK * CHUNK; idx += NTHREADS) {
            int a = idx / CHUNK, bb = idx % CHUNK;
            float val = 0.f;
            if (a >= bb) {
                float qk = 0.f;
                for (int d = 0; d < DK; ++d)
                    qk += qs[a * DK + d] * ks[bb * DK + d];
                val = qk * __expf(gcum[a] - gcum[bb]);
            }
            A[idx] = val;
        }
        __syncthreads();

        // ---- out = (q*exp(g)) @ S + attn2 @ v_new ----
        if (t < DV) {
            for (int a = 0; a < CHUNK; ++a) {
                float ega = __expf(gcum[a]);
                float inter = 0.f;
                for (int d = 0; d < DK; ++d)
                    inter += qs[a * DK + d] * ega * S_[d * DV + t];
                float intra = 0.f;
                for (int bb = 0; bb <= a; ++bb) {
                    float qk = A[a * CHUNK + bb];
                    intra += qk * vnew[bb * DV + t];
                }
                float res = inter + intra;
                int s = s0 + a;
                if (s < S)
                    out[base_qkv + (long)s * H * DV + t] = __float2bfloat16(res);
            }
        }
        __syncthreads();

        // ---- state update: S = S*exp(g[C-1]) + (k*exp(g[C-1]-g))^T @ v_new ----
        float glast = gcum[CHUNK - 1];
        float decay_last = __expf(glast);
        if (t < DV) {
            for (int d = 0; d < DK; ++d) {
                float add = 0.f;
                for (int r = 0; r < CHUNK; ++r)
                    add += ks[r * DK + d] * __expf(glast - gcum[r]) * vnew[r * DV + t];
                S_[d * DV + t] = S_[d * DV + t] * decay_last + add;
            }
        }
        __syncthreads();
    }
}

at::Tensor qwen3_next_gated_deltanet(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor g, at::Tensor beta,
                                     int64_t chunk_size) {
    auto qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous();
    auto gc = g.contiguous(), bc = beta.contiguous();
    int B = qc.size(0), S = qc.size(1), H = qc.size(2), dk = qc.size(3);
    int dv = vc.size(3);
    TORCH_CHECK(dk == DK && dv == DV, "this kernel is built for head_dim=128");
    int n_chunks = (S + CHUNK - 1) / CHUNK;
    auto out = at::empty_like(vc);

    size_t shbytes = (DK * DV + CHUNK * DK + CHUNK * DK + CHUNK * DV + CHUNK * DK + CHUNK + CHUNK + CHUNK * CHUNK + 4) *
                     sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(deltanet_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shbytes);
    deltanet_kernel<<<B * H, NTHREADS, shbytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(qc.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(kc.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(vc.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(gc.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(bc.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(out.data_ptr<at::BFloat16>()), B, H, S, n_chunks);
    return out;
}
