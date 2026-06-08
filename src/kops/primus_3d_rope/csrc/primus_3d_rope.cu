#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

// 3D axial RoPE using timm's cat-style interleaved-pair layout.
//
// apply_rot_embed_cat layout (from timm's rot() function):
//   out[2i]   = x[2i]   * cos[i] - x[2i+1] * sin[i]
//   out[2i+1] = x[2i+1] * cos[i] + x[2i]   * sin[i]
//
// Embed shape: (N, rope_channels * 2) — first half sin, second half cos.
// cos/sin are repeat_interleave'd: cos[2i] == cos[2i+1] at index i.
// We read the un-interleaved form directly from the first half of emb.
//
// q,k shape: (B, NH, N, HD) in bfloat16. rope_channels <= HD.
// Threads cover all (B*NH*N, rope_channels/2) pairs.

__global__ void rope3d_kernel(const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
                              const __nv_bfloat16 *__restrict__ sin_emb, // (N, rope_channels)
                              const __nv_bfloat16 *__restrict__ cos_emb, // (N, rope_channels)
                              __nv_bfloat16 *__restrict__ qo, __nv_bfloat16 *__restrict__ ko,
                              long n_rows, // B * NH * N
                              int HD,      // head_dim (total channels including passthrough)
                              int N,       // sequence length
                              int NH,      // num heads
                              int rope_ch) // rope_channels (channels to rotate)
{
    // Each thread handles one pair (i, i+1) in the rope region of one row.
    int n_pairs = rope_ch / 2;
    long gid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (gid >= n_rows * n_pairs)
        return;

    int pair = (int)(gid % n_pairs); // which pair in the rope region
    long row = gid / n_pairs;        // which (b, nh, tok) row

    // Compute sequence position for embed lookup (row = b*NH*N + nh*N + tok)
    int tok = (int)(row % N);

    // Indices into flat q/k
    long x_base = row * HD + pair * 2;
    // emb layout: (N, rope_ch*2) in memory = row tok has sin[0..rope_ch-1] then cos[0..rope_ch-1]
    // sin_emb and cos_emb are pointers into the emb tensor, both using the same row stride rope_ch*2
    // sin_emb[tok*(rope_ch*2) + pair*2] and cos_emb[tok*(rope_ch*2) + pair*2]
    long e_base = (long)tok * (rope_ch * 2) + pair * 2;

    // Read q/k pair
    float q0 = __bfloat162float(q[x_base]);
    float q1 = __bfloat162float(q[x_base + 1]);
    float k0 = __bfloat162float(k[x_base]);
    float k1 = __bfloat162float(k[x_base + 1]);

    // cos/sin are repeat_interleave'd: [2i] == [2i+1], read either element of the pair
    float s = __bfloat162float(sin_emb[e_base]);
    float c = __bfloat162float(cos_emb[e_base]);

    // Rotate: out[2i] = q[2i]*c - q[2i+1]*s ; out[2i+1] = q[2i+1]*c + q[2i]*s
    qo[x_base] = __float2bfloat16(q0 * c - q1 * s);
    qo[x_base + 1] = __float2bfloat16(q1 * c + q0 * s);
    ko[x_base] = __float2bfloat16(k0 * c - k1 * s);
    ko[x_base + 1] = __float2bfloat16(k1 * c + k0 * s);
}

// Copy passthrough channels (beyond rope_channels) unchanged.
__global__ void copy_passthrough_kernel(const __nv_bfloat16 *__restrict__ src, __nv_bfloat16 *__restrict__ dst,
                                        long n_rows, int HD, int rope_ch) {
    long gid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    int pass_ch = HD - rope_ch;
    if (gid >= n_rows * pass_ch)
        return;
    int ch = (int)(gid % pass_ch);
    long row = gid / pass_ch;
    long idx = row * HD + rope_ch + ch;
    dst[idx] = src[idx];
}

std::tuple<at::Tensor, at::Tensor> primus_3d_rope(at::Tensor q, at::Tensor k, at::Tensor emb) {
    // q, k: (B, NH, N, HD) bfloat16
    // emb: (N, rope_channels*2) bfloat16 — first half sin, second half cos
    auto qo = at::empty_like(q);
    auto ko = at::empty_like(k);

    int B = q.size(0);
    int NH = q.size(1);
    int N = q.size(2);
    int HD = q.size(3);
    int rope_ch = (int)(emb.size(1) / 2); // rope_channels
    long n_rows = (long)B * NH * N;
    int n_pairs = rope_ch / 2;

    int threads = 256;
    long blocks_rot = (n_rows * n_pairs + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();

    const __nv_bfloat16 *qp = reinterpret_cast<const __nv_bfloat16 *>(q.data_ptr<at::BFloat16>());
    const __nv_bfloat16 *kp = reinterpret_cast<const __nv_bfloat16 *>(k.data_ptr<at::BFloat16>());
    __nv_bfloat16 *qop = reinterpret_cast<__nv_bfloat16 *>(qo.data_ptr<at::BFloat16>());
    __nv_bfloat16 *kop = reinterpret_cast<__nv_bfloat16 *>(ko.data_ptr<at::BFloat16>());

    // emb is (N, rope_ch*2): each row = [sin[0..rope_ch-1], cos[0..rope_ch-1]]
    // sin_p points at sin[0] of row 0; cos_p points at cos[0] of row 0 (offset rope_ch within row)
    const __nv_bfloat16 *sin_p = reinterpret_cast<const __nv_bfloat16 *>(emb.data_ptr<at::BFloat16>());
    const __nv_bfloat16 *cos_p = sin_p + rope_ch; // offset within the first row; kernel uses row stride rope_ch*2

    rope3d_kernel<<<blocks_rot, threads, 0, stream>>>(qp, kp, sin_p, cos_p, qop, kop, n_rows, HD, N, NH, rope_ch);

    if (rope_ch < HD) {
        long blocks_pass = (n_rows * (HD - rope_ch) + threads - 1) / threads;
        copy_passthrough_kernel<<<blocks_pass, threads, 0, stream>>>(qp, qop, n_rows, HD, rope_ch);
        copy_passthrough_kernel<<<blocks_pass, threads, 0, stream>>>(kp, kop, n_rows, HD, rope_ch);
    }

    return {qo, ko};
}
