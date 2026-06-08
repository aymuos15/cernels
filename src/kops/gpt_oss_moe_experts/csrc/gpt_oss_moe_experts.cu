// Fused grouped-GEMM MoE kernel for gpt-oss experts.
//
// Layout (routing precomputed on the Python side, matching the transformers reference):
//   x              : [T, H]        bf16   input tokens
//   gate_up_proj   : [E, H, 2F]    bf16   gate/up weight (expert e: H -> 2F, interleaved)
//   gate_up_bias   : [E, 2F]       fp32   gate/up bias
//   down_proj      : [E, F, H]     bf16   down weight    (expert e: F -> H)
//   down_bias      : [E, H]        fp32   down bias
//   indices        : [T, topk]     int32  expert index per (token, k)
//   weights        : [T, topk]     fp32   router score per (token, k)
//
// Per expert: gate_up = x @ gate_up_proj[e] + bias  -> clamped/limited SwiGLU -> [nt, F]
//             out     = glu @ down_proj[e] + down_bias[e]
//             accumulate weighted by router score.
//
// gpt-oss SwiGLU (NOT canonical silu_mul):
//   gate = gate_up[..., ::2], up = gate_up[..., 1::2]   (interleaved even/odd split)
//   gate = clamp(gate, max=7.0);  up = clamp(up, -7.0, 7.0)
//   glu  = gate * sigmoid(gate * 1.702)
//   out  = (up + 1) * glu

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <torch/all.h>

#include <cstdint>
#include <vector>

#define GATE_LIMIT 7.0f
#define GATE_ALPHA 1.702f

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

// Add gate_up bias (per-expert, broadcast over the nt rows of this expert slot).
// gate_up: [nt, 2F]  bias: [2F]
__global__ void add_gateup_bias_kernel(__nv_bfloat16 *__restrict__ gate_up, const float *__restrict__ bias, int nt,
                                       int twoF) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nt * twoF)
        return;
    int c = idx % twoF;
    float v = __bfloat162float(gate_up[idx]) + bias[c];
    gate_up[idx] = __float2bfloat16(v);
}

// Clamped/limited SwiGLU with interleaved split.
// gate_up: [nt, 2F]  (already has bias added)  ->  glu: [nt, F]
//   gate = gate_up[:, ::2], up = gate_up[:, 1::2]
__global__ void swiglu_kernel(const __nv_bfloat16 *__restrict__ gate_up, __nv_bfloat16 *__restrict__ glu, int nt,
                              int F) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nt * F)
        return;
    int row = idx / F;
    int col = idx % F;
    const __nv_bfloat16 *gu = gate_up + (long)row * 2 * F;
    float gate = __bfloat162float(gu[2 * col]);
    float up = __bfloat162float(gu[2 * col + 1]);
    gate = fminf(gate, GATE_LIMIT);
    up = fmaxf(-GATE_LIMIT, fminf(up, GATE_LIMIT));
    float glu_v = gate / (1.f + expf(-gate * GATE_ALPHA));
    glu[idx] = __float2bfloat16((up + 1.f) * glu_v);
}

// Weighted scatter-add (down bias added then scaled by router weight, matching the reference
// where the bias is part of `out` before the routing-weight multiply).
// contrib[i, :] += (down_proj output[i,:] + down_bias[e]) * perm_weights[i], scattered to token.
__global__ void scatter_add_f32_kernel(const __nv_bfloat16 *__restrict__ contrib,
                                       const int *__restrict__ perm_token_ids, const float *__restrict__ perm_weights,
                                       const float *__restrict__ down_bias, float *__restrict__ out_f32, int nt,
                                       int H) {
    int i = blockIdx.x;
    int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= nt || h >= H)
        return;
    float v = (__bfloat162float(contrib[i * H + h]) + down_bias[h]) * perm_weights[i];
    atomicAdd(&out_f32[perm_token_ids[i] * H + h], v);
}

at::Tensor gpt_oss_moe_experts(at::Tensor x,            // [T, H]      bf16
                               at::Tensor gate_up_proj, // [E, H, 2F]  bf16
                               at::Tensor gate_up_bias, // [E, 2F]     fp32
                               at::Tensor down_proj,    // [E, F, H]   bf16
                               at::Tensor down_bias,    // [E, H]      fp32
                               at::Tensor indices,      // [T, topk]   int32
                               at::Tensor weights,      // [T, topk]   fp32
                               int64_t topk) {
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int T = (int)x.size(0), H = (int)x.size(1);
    int E = (int)gate_up_proj.size(0);
    int twoF = (int)gate_up_proj.size(2);
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
    const float *gub_ptr = gate_up_bias.data_ptr<float>();
    const float *db_ptr = down_bias.data_ptr<float>();

    for (int e = 0; e < E; e++) {
        int nt = h_offsets[e + 1] - h_offsets[e];
        if (nt == 0)
            continue;
        int off = h_offsets[e];

        const __nv_bfloat16 *Ae = x_all + (long)off * H;         // [nt, H]
        __nv_bfloat16 *GUe = gu_all + (long)off * twoF;          // [nt, 2F]
        __nv_bfloat16 *GLe = gl_all + (long)off * F;             // [nt, F]
        __nv_bfloat16 *De = c_all + (long)off * H;               // [nt, H]
        const __nv_bfloat16 *w1e = gup_ptr + (long)e * H * twoF; // [H, 2F]
        const __nv_bfloat16 *w2e = down_ptr + (long)e * F * H;   // [F, H]

        // GEMM1: GUe[nt, 2F] = Ae[nt, H] @ w1e[H, 2F]   (col-major: m=2F, n=nt, k=H)
        cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, twoF, nt, H, &alpha, w1e, CUDA_R_16BF, twoF, Ae, CUDA_R_16BF, H,
                     &beta, GUe, CUDA_R_16BF, twoF, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        // Add gate_up bias.
        {
            int n = nt * twoF, threads = 256;
            add_gateup_bias_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(GUe, gub_ptr + (long)e * twoF,
                                                                                        nt, twoF);
        }

        // Clamped/limited SwiGLU with interleaved split -> GLe[nt, F].
        {
            int n = nt * F, threads = 256;
            swiglu_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(GUe, GLe, nt, F);
        }

        // GEMM2: De[nt, H] = GLe[nt, F] @ w2e[F, H]   (col-major: m=H, n=nt, k=F)
        cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, H, nt, F, &alpha, w2e, CUDA_R_16BF, H, GLe, CUDA_R_16BF, F,
                     &beta, De, CUDA_R_16BF, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        // Add down bias + weight by router score + scatter-add into fp32 accumulator.
        {
            int threads = 128, bh = (H + threads - 1) / threads;
            dim3 grid(nt, bh);
            scatter_add_f32_kernel<<<grid, threads, 0, stream>>>(
                De, perm_token_ids.data_ptr<int>() + off, perm_weights.data_ptr<float>() + off, db_ptr + (long)e * H,
                out_f32.data_ptr<float>(), nt, H);
        }
    }

    return out_f32.to(at::kBFloat16);
}
