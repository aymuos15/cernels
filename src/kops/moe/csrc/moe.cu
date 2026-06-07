// Fused grouped-GEMM MoE kernel — optimized version.
//
// Layout:
//   x          : [T, H]         bf16   input tokens
//   w1         : [E, H, F]      bf16   gate/up weight (expert i: H->F)
//   w2         : [E, F, H]      bf16   down weight    (expert i: F->H)
//   indices    : [T, topk]      int32  expert index per (token, k)
//   weights    : [T, topk]      fp32   expert weight per (token, k)
//
// Algorithm (all on-device, no D2H sync):
//   1. Count tokens per expert + exclusive prefix sum -> expert offsets (on device)
//   2. Scatter token positions into a permutation array sorted by expert
//   3. For each expert: cuBLAS bf16 GEMM w1, activation, cuBLAS bf16 GEMM w2, scatter-add
//   4. Convert fp32 accumulator to bf16

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <torch/all.h>

#include <cfloat>
#include <cstdint>

// ──────────────────────────────────────────────────────────────────────────────
// Step 1: count tokens assigned to each expert.
// indices: [T, topk], expert_counts: [E]  (zero-initialized by caller)
// ──────────────────────────────────────────────────────────────────────────────
__global__ void count_tokens_kernel(const int *__restrict__ indices, // [T*topk]
                                    int *__restrict__ expert_counts, // [E]
                                    int Ttopk, int E) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Ttopk)
        return;
    int e = indices[i];
    if (e >= 0 && e < E)
        atomicAdd(&expert_counts[e], 1);
}

// ──────────────────────────────────────────────────────────────────────────────
// Step 2: build permutation arrays.
// For each (token, k) pair, place the token index at the right position within
// its expert's slot (using atomicAdd on a running write cursor).
//
// perm_token_ids[expert_write_cursor[e]++] = t
// perm_weights[same pos]                   = weights[t,k]
//
// expert_offsets[E+1]: prefix sums, expert_write_cursors[E]: starts as copy of
// offsets[0..E-1] and grows.
// ──────────────────────────────────────────────────────────────────────────────
__global__ void build_perm_kernel(const int *__restrict__ indices,        // [T, topk]
                                  const float *__restrict__ weights,      // [T, topk]
                                  const int *__restrict__ expert_offsets, // [E+1]
                                  int *__restrict__ write_cursors,        // [E] (mutable)
                                  int *__restrict__ perm_token_ids,       // [T*topk]
                                  float *__restrict__ perm_weights,       // [T*topk]
                                  int T, int topk, int E) {
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

// ──────────────────────────────────────────────────────────────────────────────
// Gather: gathered[i, :] = x[perm_token_ids[i], :]
// ──────────────────────────────────────────────────────────────────────────────
__global__ void gather_kernel(const __nv_bfloat16 *__restrict__ x,  // [T, H]
                              const int *__restrict__ perm_ids,     // [Ttk]
                              __nv_bfloat16 *__restrict__ gathered, // [Ttk, H]
                              int Ttk, int H) {
    int i = blockIdx.x;
    int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= Ttk || h >= H)
        return;
    gathered[i * H + h] = x[perm_ids[i] * H + h];
}

// ──────────────────────────────────────────────────────────────────────────────
// Activation in-place (gelu tanh approx / silu / relu)
// act_id: 0=gelu_tanh, 1=silu, 2=relu
// ──────────────────────────────────────────────────────────────────────────────
__device__ inline float gelu_f(float x) {
    float c = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.f + tanhf(c));
}

__global__ void activation_kernel(__nv_bfloat16 *__restrict__ x, int n, int act_id) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float v = __bfloat162float(x[i]);
    if (act_id == 0)
        v = gelu_f(v);
    else if (act_id == 1)
        v = v / (1.f + expf(-v));
    else
        v = fmaxf(0.f, v);
    x[i] = __float2bfloat16(v);
}

// ──────────────────────────────────────────────────────────────────────────────
// Weighted scatter-add into fp32 accumulator
// out_f32[perm_token_ids[off+i], h] += perm_weights[off+i] * contrib[off+i, h]
// ──────────────────────────────────────────────────────────────────────────────
__global__ void scatter_add_f32_kernel(const __nv_bfloat16 *__restrict__ contrib, // [nt, H]
                                       const int *__restrict__ perm_token_ids,    // [nt]
                                       const float *__restrict__ perm_weights,    // [nt]
                                       float *__restrict__ out_f32,               // [T, H]
                                       int nt, int H) {
    int i = blockIdx.x;
    int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= nt || h >= H)
        return;
    float v = __bfloat162float(contrib[i * H + h]) * perm_weights[i];
    atomicAdd(&out_f32[perm_token_ids[i] * H + h], v);
}

// ──────────────────────────────────────────────────────────────────────────────
// Entry point.  Routing is precomputed on the Python side (matching megablocks dtype).
// ──────────────────────────────────────────────────────────────────────────────
at::Tensor moe_grouped_gemm(at::Tensor x,       // [T, H]      bf16
                            at::Tensor w1,      // [E, H, F]   bf16
                            at::Tensor w2,      // [E, F, H]   bf16
                            at::Tensor indices, // [T, topk]   int32
                            at::Tensor weights, // [T, topk]   fp32
                            int64_t topk, int64_t act_id) {
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int T = (int)x.size(0), H = (int)x.size(1);
    int E = (int)w1.size(0), F = (int)w1.size(2);
    int Ttk = T * topk;

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(w1.is_contiguous(), "w1 must be contiguous");
    TORCH_CHECK(w2.is_contiguous(), "w2 must be contiguous");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");

    // ── 1. Count tokens per expert (on device) ────────────────────────────
    auto expert_counts = at::zeros({E}, indices.options());
    {
        int threads = 256;
        int blocks = (Ttk + threads - 1) / threads;
        count_tokens_kernel<<<blocks, threads, 0, stream>>>(indices.data_ptr<int>(), expert_counts.data_ptr<int>(), Ttk,
                                                            E);
    }

    // ── 2. Prefix sum -> expert offsets (E+1 elements) on device ─────────
    // offsets[0]=0, offsets[e+1]=offsets[e]+counts[e]
    // Use at::cumsum (runs on device, no D2H sync needed here).
    auto expert_offsets = at::zeros({E + 1}, indices.options()); // [E+1]
    expert_offsets.narrow(0, 1, E).copy_(at::cumsum(expert_counts, 0));
    // write_cursors[e] = expert_offsets[e] (starting write position for each expert)
    auto write_cursors = expert_offsets.narrow(0, 0, E).clone();

    // ── 3. Build permutation arrays (on device) ───────────────────────────
    auto perm_token_ids = at::empty({Ttk}, indices.options());
    auto perm_weights = at::empty({Ttk}, weights.options());
    {
        int threads = 256;
        int blocks = (Ttk + threads - 1) / threads;
        build_perm_kernel<<<blocks, threads, 0, stream>>>(
            indices.data_ptr<int>(), weights.data_ptr<float>(), expert_offsets.data_ptr<int>(),
            write_cursors.data_ptr<int>(), perm_token_ids.data_ptr<int>(), perm_weights.data_ptr<float>(), T, topk, E);
    }

    // ── 4. Gather all tokens (permuted layout) [Ttk, H] ──────────────────
    auto gathered_all = at::empty({Ttk, H}, x.options()); // [Ttk, H] bf16
    {
        int threads = 128;
        int bh = (H + threads - 1) / threads;
        dim3 grid(Ttk, bh);
        gather_kernel<<<grid, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr<at::BFloat16>()), perm_token_ids.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16 *>(gathered_all.data_ptr<at::BFloat16>()), Ttk, H);
    }

    // ── 5. cuBLAS handle on current stream ───────────────────────────────
    cublasHandle_t cublas = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas, stream);

    // ── 6. Allocate output buffers for all experts ────────────────────────
    // hidden_all [Ttk, F] — after w1 + activation
    // contrib_all [Ttk, H] — after w2
    auto hidden_all = at::empty({Ttk, F}, x.options());  // bf16
    auto contrib_all = at::empty({Ttk, H}, x.options()); // bf16

    // fp32 output accumulator
    auto out_f32 = at::zeros({T, H}, x.options().dtype(at::kFloat));

    // cuBLAS bf16->fp32 GEMM scalars
    float alpha = 1.0f, beta = 0.0f;

    // ── 7. Read expert offsets to host (E+1 ints = 36 bytes, tiny sync) ──
    // We need per-expert sizes to launch the correct sub-matrix GEMMs.
    // This is a single small D2H copy, not per-token data.
    std::vector<int> h_offsets(E + 1);
    cudaMemcpyAsync(h_offsets.data(), expert_offsets.data_ptr<int>(), (E + 1) * sizeof(int), cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream); // sync ONLY for the 36-byte offset array

    const __nv_bfloat16 *x_all = reinterpret_cast<const __nv_bfloat16 *>(gathered_all.data_ptr<at::BFloat16>());
    __nv_bfloat16 *h_all = reinterpret_cast<__nv_bfloat16 *>(hidden_all.data_ptr<at::BFloat16>());
    __nv_bfloat16 *c_all = reinterpret_cast<__nv_bfloat16 *>(contrib_all.data_ptr<at::BFloat16>());
    const __nv_bfloat16 *w1_ptr = reinterpret_cast<const __nv_bfloat16 *>(w1.data_ptr<at::BFloat16>());
    const __nv_bfloat16 *w2_ptr = reinterpret_cast<const __nv_bfloat16 *>(w2.data_ptr<at::BFloat16>());

    // ── 8. Per-expert cuBLAS GEMMs (no per-expert allocations) ───────────
    for (int e = 0; e < E; e++) {
        int nt = h_offsets[e + 1] - h_offsets[e];
        if (nt == 0)
            continue;

        int off = h_offsets[e]; // offset in permuted arrays

        const __nv_bfloat16 *Ae = x_all + (long)off * H;     // gathered tokens for expert e: [nt, H]
        __nv_bfloat16 *Ce = h_all + (long)off * F;           // hidden output for expert e: [nt, F]
        const __nv_bfloat16 *w1e = w1_ptr + (long)e * H * F; // [H, F]
        const __nv_bfloat16 *w2e = w2_ptr + (long)e * F * H; // [F, H]
        __nv_bfloat16 *De = c_all + (long)off * H;           // down output for expert e: [nt, H]

        // GEMM1: Ce[nt, F] = Ae[nt, H] @ w1e[H, F]
        // cuBLAS is column-major. Use C^T = W1^T * A^T equivalence:
        //   result col-major [F, nt] = op(W1) * op(A)
        //   W1 row-major [H,F] = col-major [F,H] with lda=F; op=N gives F×H
        //   A  row-major [nt,H]= col-major [H,nt] with lda=H; op=N gives H×nt
        //   m=F, n=nt, k=H
        cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, // op(W1)=N: F×H, op(A)=N: H×nt
                     F, nt, H,                         // m=F, n=nt, k=H
                     &alpha, w1e, CUDA_R_16BF, F,      // A=W1 col-major [F,H], lda=F
                     Ae, CUDA_R_16BF, H,               // B=A  col-major [H,nt], ldb=H
                     &beta, Ce, CUDA_R_16BF, F,        // C col-major [F,nt], ldc=F  => row-major [nt,F]
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        // Activation in-place on Ce [nt, F]
        {
            int n = nt * F;
            int threads = 256;
            activation_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(Ce, n, act_id);
        }

        // GEMM2: De[nt, H] = Ce[nt, F] @ w2e[F, H]
        // W2 row-major [F,H] = col-major [H,F] with lda=H; op=N gives H×F
        // Ce row-major [nt,F]= col-major [F,nt] with lda=F; op=N gives F×nt
        // m=H, n=nt, k=F
        cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, // op(W2)=N: H×F, op(Ce)=N: F×nt
                     H, nt, F,                         // m=H, n=nt, k=F
                     &alpha, w2e, CUDA_R_16BF, H,      // A=W2 col-major [H,F], lda=H
                     Ce, CUDA_R_16BF, F,               // B=Ce col-major [F,nt], ldb=F
                     &beta, De, CUDA_R_16BF, H,        // C col-major [H,nt], ldc=H  => row-major [nt,H]
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        // Scatter-add weighted into fp32 accumulator
        {
            int threads = 128;
            int bh = (H + threads - 1) / threads;
            dim3 grid(nt, bh);
            scatter_add_f32_kernel<<<grid, threads, 0, stream>>>(De, perm_token_ids.data_ptr<int>() + off,
                                                                 perm_weights.data_ptr<float>() + off,
                                                                 out_f32.data_ptr<float>(), nt, H);
        }
    }

    return out_f32.to(at::kBFloat16);
}
