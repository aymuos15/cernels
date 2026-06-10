// Fused grouped-GEMM MoE kernel for DeepSeek-OCR-2 routed experts (plain SwiGLU, no
// bias; the 2 shared experts live outside this op). Constant-swap port of
// cohere2-moe-experts retargeted at DeepSeek-OCR-2: E=64 experts, top-6, H=1280,
// F=896, softmax-greedy router weights x routed_scaling_factor (computed outside).
//
// Layout (routing precomputed on the Python side, matching transformers
// DeepseekOcr2TextExperts; F.linear(x, W) semantics so weights are stored [out, in]):
//   x              : [T, H]        bf16   input tokens
//   gate_up_proj   : [E, 2F, H]    bf16   gate/up weight (expert e: H -> 2F via x @ W^T)
//   down_proj      : [E, H, F]     bf16   down weight    (expert e: F -> H via g @ W^T)
//   indices        : [T, topk]     int32  expert index per (token, k)
//   weights        : [T, topk]     fp32   router score per (token, k)
//
// Per expert: gate_up = x @ gate_up_proj[e]^T            -> [nt, 2F]
//             gate = gate_up[:, :F], up = gate_up[:, F:]  (chunked split)
//             glu  = silu(gate) * up                      -> [nt, F]
//             out  = glu @ down_proj[e]^T                 -> [nt, H]
//             accumulate weighted by router score (fp32 accumulator).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <torch/all.h>

#include <cstdint>
#include <vector>

// Count tokens per expert.
__global__ void count_tokens_kernel(const int *__restrict__ indices, int *__restrict__ expert_counts, int Ttopk,
                                    int E) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Ttopk)
        return;
    int e = indices[i];
    if (e >= 0 && e < E)
        atomicAdd(&expert_counts[e], 1);
}

// Build permutation arrays sorted by expert.
__global__ void build_perm_kernel(const int *__restrict__ indices, const float *__restrict__ weights,
                                  int *__restrict__ write_cursors, int *__restrict__ perm_token_ids,
                                  float *__restrict__ perm_weights, int T, int topk, int E) {
    int tk = blockIdx.x * blockDim.x + threadIdx.x;
    if (tk >= T * topk)
        return;
    int t = tk / topk;
    int k = tk % topk;
    int e = indices[t * topk + k];
    if (e < 0 || e >= E)
        return;
    float w = weights[t * topk + k];
    int pos = atomicAdd(&write_cursors[e], 1);
    perm_token_ids[pos] = t;
    perm_weights[pos] = w;
}

// gathered[i, :] = x[perm_token_ids[i], :]
__global__ void gather_kernel(const __nv_bfloat16 *__restrict__ x, const int *__restrict__ perm_ids,
                              __nv_bfloat16 *__restrict__ gathered, int Ttk, int H) {
    int i = blockIdx.x;
    int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= Ttk || h >= H)
        return;
    gathered[i * H + h] = x[perm_ids[i] * H + h];
}

// Plain SwiGLU with chunked split.
// gate_up: [nt, 2F]  ->  glu: [nt, F]
//   gate = gate_up[:, :F], up = gate_up[:, F:];  glu = silu(gate) * up
__global__ void swiglu_kernel(const __nv_bfloat16 *__restrict__ gate_up, __nv_bfloat16 *__restrict__ glu, int nt,
                              int F) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nt * F)
        return;
    int row = idx / F;
    int col = idx % F;
    const __nv_bfloat16 *gu = gate_up + (long)row * 2 * F;
    float gate = __bfloat162float(gu[col]);
    float up = __bfloat162float(gu[F + col]);
    float silu = gate / (1.f + expf(-gate));
    glu[idx] = __float2bfloat16(silu * up);
}

// Weighted scatter-add: contrib[i,:] * perm_weights[i], scattered to its token.
__global__ void scatter_add_f32_kernel(const __nv_bfloat16 *__restrict__ contrib,
                                       const int *__restrict__ perm_token_ids, const float *__restrict__ perm_weights,
                                       float *__restrict__ out_f32, int nt, int H) {
    int i = blockIdx.x;
    int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= nt || h >= H)
        return;
    float v = __bfloat162float(contrib[i * H + h]) * perm_weights[i];
    atomicAdd(&out_f32[perm_token_ids[i] * H + h], v);
}

// ---------------------------------------------------------------------------
// Decode path (T ~ 1..4): fused gather-GEMV. The grouped-GEMM path above costs a
// D2H sync (host reads per-expert offsets) plus ~3 launches per hit expert — at one
// token that overhead dominates the ~41 MB of expert-weight traffic (top-6 x 3 x
// 896 x 1280 bf16). Here the whole
// op is two persistent-shape kernels (+ zeros + bf16 cast), no sync: one block per
// (token, k) pair × output tile, weights streamed once, fp32 accumulation.

// Warp-cooperative dot of two bf16 rows of length n (n % 8 == 0, rows 16B-aligned:
// both H*2 and F*2 are multiples of 16 for any n % 8 == 0). float4 = 8 bf16 per lane
// per step -> 512 B per warp iteration, fully coalesced. Result valid on lane 0.
__device__ float warp_dot_bf16(const __nv_bfloat16 *__restrict__ a, const __nv_bfloat16 *__restrict__ b, int n,
                               int lane) {
    float acc = 0.f;
    const float4 *a4 = reinterpret_cast<const float4 *>(a);
    const float4 *b4 = reinterpret_cast<const float4 *>(b);
    for (int i = lane; i < n / 8; i += 32) {
        float4 av = a4[i], bv = b4[i];
        const __nv_bfloat162 *ah = reinterpret_cast<const __nv_bfloat162 *>(&av);
        const __nv_bfloat162 *bh = reinterpret_cast<const __nv_bfloat162 *>(&bv);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 af = __bfloat1622float2(ah[j]);
            float2 bf = __bfloat1622float2(bh[j]);
            acc = fmaf(af.x, bf.x, acc);
            acc = fmaf(af.y, bf.y, acc);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    return acc;
}

constexpr int kDecodeThreads = 256; // 8 warps
constexpr int kDecodeTileF = 32;    // glu rows per block (kernel 1)
constexpr int kDecodeTileH = 32;    // out rows per block (kernel 2)

// glu[pair, f] = silu(x[t] . gate_up[e][f]) * (x[t] . gate_up[e][F+f])
// grid (T*topk, F/kDecodeTileF); x[t] staged in smem (H*2 bytes) once per block.
__global__ void decode_gateup_swiglu_kernel(const __nv_bfloat16 *__restrict__ x,
                                            const __nv_bfloat16 *__restrict__ gate_up, const int *__restrict__ indices,
                                            float *__restrict__ glu, int H, int F, int topk) {
    extern __shared__ __nv_bfloat16 sx[];
    int pair = blockIdx.x;
    int t = pair / topk;
    int e = indices[pair];
    const float4 *xsrc = reinterpret_cast<const float4 *>(x + (long)t * H);
    float4 *xdst = reinterpret_cast<float4 *>(sx);
    for (int i = threadIdx.x; i < H / 8; i += blockDim.x)
        xdst[i] = xsrc[i];
    __syncthreads();
    if (e < 0)
        return;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const __nv_bfloat16 *w1 = gate_up + (long)e * 2 * F * H;
    int f_end = min(F, (int)(blockIdx.y + 1) * kDecodeTileF);
    for (int f = blockIdx.y * kDecodeTileF + warp; f < f_end; f += kDecodeThreads / 32) {
        float g = warp_dot_bf16(sx, w1 + (long)f * H, H, lane);
        float u = warp_dot_bf16(sx, w1 + (long)(F + f) * H, H, lane);
        if (lane == 0)
            glu[(long)pair * F + f] = (g / (1.f + expf(-g))) * u;
    }
}

// out[t, h] += weights[pair] * (glu[pair] . down[e][h]);  fp32 atomics (topk-way).
// grid (T*topk, H/kDecodeTileH); glu[pair] staged in smem (F*4 bytes) once per block.
__global__ void decode_down_accum_kernel(const float *__restrict__ glu, const __nv_bfloat16 *__restrict__ down,
                                         const int *__restrict__ indices, const float *__restrict__ weights,
                                         float *__restrict__ out, int H, int F, int topk) {
    extern __shared__ float sg[];
    int pair = blockIdx.x;
    int t = pair / topk;
    int e = indices[pair];
    for (int i = threadIdx.x; i < F; i += blockDim.x)
        sg[i] = glu[(long)pair * F + i];
    __syncthreads();
    if (e < 0)
        return;
    float w = weights[pair];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const __nv_bfloat16 *w2 = down + (long)e * H * F;
    int h_end = min(H, (int)(blockIdx.y + 1) * kDecodeTileH);
    for (int h = blockIdx.y * kDecodeTileH + warp; h < h_end; h += kDecodeThreads / 32) {
        float acc = 0.f;
        const float4 *b4 = reinterpret_cast<const float4 *>(w2 + (long)h * F);
        for (int i = lane; i < F / 8; i += 32) {
            float4 bv = b4[i];
            const __nv_bfloat162 *bh = reinterpret_cast<const __nv_bfloat162 *>(&bv);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 bf = __bfloat1622float2(bh[j]);
                acc = fmaf(sg[i * 8 + 2 * j], bf.x, acc);
                acc = fmaf(sg[i * 8 + 2 * j + 1], bf.y, acc);
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, o);
        if (lane == 0)
            atomicAdd(&out[(long)t * H + h], w * acc);
    }
}

at::Tensor deepseek_ocr2_moe_experts_decode(at::Tensor x,            // [T, H]      bf16
                                            at::Tensor gate_up_proj, // [E, 2F, H]  bf16
                                            at::Tensor down_proj,    // [E, H, F]   bf16
                                            at::Tensor indices,      // [T, topk]   int32
                                            at::Tensor weights,      // [T, topk]   fp32
                                            int64_t topk) {
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int T = (int)x.size(0), H = (int)x.size(1);
    int twoF = (int)gate_up_proj.size(1);
    int F = twoF / 2;
    int pairs = T * (int)topk;

    TORCH_CHECK(x.is_contiguous() && gate_up_proj.is_contiguous() && down_proj.is_contiguous() &&
                    indices.is_contiguous() && weights.is_contiguous(),
                "all inputs must be contiguous");
    TORCH_CHECK(H % 8 == 0 && F % 8 == 0, "H and F must be multiples of 8 (float4 bf16 loads)");

    auto glu = at::empty({pairs, F}, x.options().dtype(at::kFloat));
    auto out_f32 = at::zeros({T, H}, x.options().dtype(at::kFloat));

    {
        dim3 grid(pairs, (F + kDecodeTileF - 1) / kDecodeTileF);
        decode_gateup_swiglu_kernel<<<grid, kDecodeThreads, H * sizeof(__nv_bfloat16), stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16 *>(gate_up_proj.data_ptr<at::BFloat16>()), indices.data_ptr<int>(),
            glu.data_ptr<float>(), H, F, (int)topk);
    }
    {
        dim3 grid(pairs, (H + kDecodeTileH - 1) / kDecodeTileH);
        decode_down_accum_kernel<<<grid, kDecodeThreads, F * sizeof(float), stream>>>(
            glu.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16 *>(down_proj.data_ptr<at::BFloat16>()),
            indices.data_ptr<int>(), weights.data_ptr<float>(), out_f32.data_ptr<float>(), H, F, (int)topk);
    }

    return out_f32.to(at::kBFloat16);
}

at::Tensor deepseek_ocr2_moe_experts(at::Tensor x,            // [T, H]      bf16
                                     at::Tensor gate_up_proj, // [E, 2F, H]  bf16
                                     at::Tensor down_proj,    // [E, H, F]   bf16
                                     at::Tensor indices,      // [T, topk]   int32
                                     at::Tensor weights,      // [T, topk]   fp32
                                     int64_t topk) {
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int T = (int)x.size(0), H = (int)x.size(1);
    int E = (int)gate_up_proj.size(0);
    int twoF = (int)gate_up_proj.size(1);
    int F = twoF / 2;
    int Ttk = T * topk;

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(gate_up_proj.is_contiguous(), "gate_up_proj must be contiguous");
    TORCH_CHECK(down_proj.is_contiguous(), "down_proj must be contiguous");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");

    // 1. Count tokens per expert.
    auto expert_counts = at::zeros({E}, indices.options());
    {
        int threads = 256, blocks = (Ttk + threads - 1) / threads;
        count_tokens_kernel<<<blocks, threads, 0, stream>>>(indices.data_ptr<int>(), expert_counts.data_ptr<int>(), Ttk,
                                                            E);
    }

    // 2. Prefix sum -> offsets [E+1].
    auto expert_offsets = at::zeros({E + 1}, indices.options());
    expert_offsets.narrow(0, 1, E).copy_(at::cumsum(expert_counts, 0));
    auto write_cursors = expert_offsets.narrow(0, 0, E).clone();

    // 3. Build permutation arrays.
    auto perm_token_ids = at::empty({Ttk}, indices.options());
    auto perm_weights = at::empty({Ttk}, weights.options());
    {
        int threads = 256, blocks = (Ttk + threads - 1) / threads;
        build_perm_kernel<<<blocks, threads, 0, stream>>>(indices.data_ptr<int>(), weights.data_ptr<float>(),
                                                          write_cursors.data_ptr<int>(), perm_token_ids.data_ptr<int>(),
                                                          perm_weights.data_ptr<float>(), T, topk, E);
    }

    // 4. Gather tokens into permuted layout.
    auto gathered_all = at::empty({Ttk, H}, x.options());
    {
        int threads = 128, bh = (H + threads - 1) / threads;
        dim3 grid(Ttk, bh);
        gather_kernel<<<grid, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr<at::BFloat16>()), perm_token_ids.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16 *>(gathered_all.data_ptr<at::BFloat16>()), Ttk, H);
    }

    cublasHandle_t cublas = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas, stream);

    auto gateup_all = at::empty({Ttk, twoF}, x.options()); // bf16  [Ttk, 2F]
    auto glu_all = at::empty({Ttk, F}, x.options());       // bf16  [Ttk, F]
    auto contrib_all = at::empty({Ttk, H}, x.options());   // bf16  [Ttk, H]
    auto out_f32 = at::zeros({T, H}, x.options().dtype(at::kFloat));

    float alpha = 1.0f, beta = 0.0f;

    // Read per-expert offsets to host (tiny D2H sync).
    std::vector<int> h_offsets(E + 1);
    cudaMemcpyAsync(h_offsets.data(), expert_offsets.data_ptr<int>(), (E + 1) * sizeof(int), cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);

    const __nv_bfloat16 *x_all = reinterpret_cast<const __nv_bfloat16 *>(gathered_all.data_ptr<at::BFloat16>());
    __nv_bfloat16 *gu_all = reinterpret_cast<__nv_bfloat16 *>(gateup_all.data_ptr<at::BFloat16>());
    __nv_bfloat16 *gl_all = reinterpret_cast<__nv_bfloat16 *>(glu_all.data_ptr<at::BFloat16>());
    __nv_bfloat16 *c_all = reinterpret_cast<__nv_bfloat16 *>(contrib_all.data_ptr<at::BFloat16>());
    const __nv_bfloat16 *gup_ptr = reinterpret_cast<const __nv_bfloat16 *>(gate_up_proj.data_ptr<at::BFloat16>());
    const __nv_bfloat16 *down_ptr = reinterpret_cast<const __nv_bfloat16 *>(down_proj.data_ptr<at::BFloat16>());

    for (int e = 0; e < E; e++) {
        int nt = h_offsets[e + 1] - h_offsets[e];
        if (nt == 0)
            continue;
        int off = h_offsets[e];

        const __nv_bfloat16 *Ae = x_all + (long)off * H;         // [nt, H]   row-major
        __nv_bfloat16 *GUe = gu_all + (long)off * twoF;          // [nt, 2F]
        __nv_bfloat16 *GLe = gl_all + (long)off * F;             // [nt, F]
        __nv_bfloat16 *De = c_all + (long)off * H;               // [nt, H]
        const __nv_bfloat16 *w1e = gup_ptr + (long)e * twoF * H; // [2F, H]   row-major
        const __nv_bfloat16 *w2e = down_ptr + (long)e * H * F;   // [H, F]    row-major

        // GEMM1: GUe[nt, 2F] = Ae[nt, H] @ w1e[2F, H]^T
        //   col-major: C[2F, nt] = op(A=w1e)[2F,H] @ op(B=Ae)[H,nt]
        //   w1e stored row-major [2F,H] == col-major [H,2F] -> OP_T, lda=H
        //   Ae  stored row-major [nt,H] == col-major [H,nt] -> OP_N, ldb=H
        //   C   col-major [2F,nt] == row-major [nt,2F],          ldc=2F
        cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, twoF, nt, H, &alpha, w1e, CUDA_R_16BF, H, Ae, CUDA_R_16BF, H,
                     &beta, GUe, CUDA_R_16BF, twoF, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        // Plain SwiGLU (chunked) -> GLe[nt, F].
        {
            int n = nt * F, threads = 256;
            swiglu_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(GUe, GLe, nt, F);
        }

        // GEMM2: De[nt, H] = GLe[nt, F] @ w2e[H, F]^T
        //   col-major: C[H, nt] = op(A=w2e)[H,F] @ op(B=GLe)[F,nt]
        //   w2e stored row-major [H,F] == col-major [F,H] -> OP_T, lda=F
        //   GLe stored row-major [nt,F] == col-major [F,nt] -> OP_N, ldb=F
        //   C   col-major [H,nt] == row-major [nt,H],           ldc=H
        cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, H, nt, F, &alpha, w2e, CUDA_R_16BF, F, GLe, CUDA_R_16BF, F,
                     &beta, De, CUDA_R_16BF, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        // Weight by router score + scatter-add into fp32 accumulator.
        {
            int threads = 128, bh = (H + threads - 1) / threads;
            dim3 grid(nt, bh);
            scatter_add_f32_kernel<<<grid, threads, 0, stream>>>(De, perm_token_ids.data_ptr<int>() + off,
                                                                 perm_weights.data_ptr<float>() + off,
                                                                 out_f32.data_ptr<float>(), nt, H);
        }
    }

    return out_f32.to(at::kBFloat16);
}
