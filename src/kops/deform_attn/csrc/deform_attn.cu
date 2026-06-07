#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

// Fused multi-scale deformable attention forward — performance-tuned.
//
// Parallelization: one thread-GROUP of (hidden_dim/4) threads per (b,q,h).
// Each thread owns 4 contiguous channels and loads them with a single float4
// (the hidden_dim axis is innermost/contiguous in `value`), so corner gathers
// are vectorized AND coalesced across the group.
//
// Many groups are packed per block (blockDim 256) for high occupancy.
//
// Bilinear geometry (corner offsets/weights, attention weight) is recomputed in
// registers per (level,point) — cheap scalar work, reused across the 4 channels.
//
// grid_sample semantics: align_corners=False, padding_mode=zeros
//   px = loc_x * W - 0.5,  py = loc_y * H - 0.5

template <int VEC>
__global__ void ms_deform_attn_fwd_kernel(const float *__restrict__ value, const int64_t *__restrict__ spatial_shapes,
                                          const int64_t *__restrict__ level_start_index,
                                          const float *__restrict__ sampling_loc, const float *__restrict__ attn_weight,
                                          float *__restrict__ out, int B, int L_total, int n_heads, int hidden_dim,
                                          int num_queries, int n_levels, int n_points) {
    const int lanes = hidden_dim / VEC; // threads per (b,q,h) group
    int gid = threadIdx.x / lanes;      // group within block
    int lane = threadIdx.x % lanes;     // which VEC-chunk of channels
    int groups_per_block = blockDim.x / lanes;
    int bqh = blockIdx.x * groups_per_block + gid; // index over B*num_queries*n_heads

    int total = B * num_queries * n_heads;
    if (bqh >= total)
        return;

    int h = bqh % n_heads;
    int tmp = bqh / n_heads;
    int q = tmp % num_queries;
    int b = tmp / num_queries;

    int lp = n_levels * n_points;
    const float *loc_base = sampling_loc + (size_t)bqh * lp * 2;
    const float *aw_base = attn_weight + (size_t)bqh * lp;

    const long stride_row = (long)n_heads * hidden_dim; // in floats
    const long stride_row_v = stride_row / VEC;         // in float4 units
    const float4 *val_bh = reinterpret_cast<const float4 *>(value + ((size_t)b * L_total) * n_heads * hidden_dim +
                                                            (size_t)h * hidden_dim) +
                           lane;

    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

    int pidx = 0;
    for (int lv = 0; lv < n_levels; lv++) {
        int H = (int)spatial_shapes[lv * 2];
        int W = (int)spatial_shapes[lv * 2 + 1];
        const float4 *val_lv = val_bh + (size_t)level_start_index[lv] * stride_row_v;

        for (int pt = 0; pt < n_points; pt++, pidx++) {
            float loc_x = loc_base[pidx * 2];
            float loc_y = loc_base[pidx * 2 + 1];
            float aw = aw_base[pidx];

            float px = loc_x * (float)W - 0.5f;
            float py = loc_y * (float)H - 0.5f;

            int x0 = (int)floorf(px);
            int y0 = (int)floorf(py);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            float wx1 = px - (float)x0;
            float wy1 = py - (float)y0;
            float wx0 = 1.f - wx1;
            float wy0 = 1.f - wy1;

            bool xv0 = (x0 >= 0 && x0 < W);
            bool xv1 = (x1 >= 0 && x1 < W);
            bool yv0 = (y0 >= 0 && y0 < H);
            bool yv1 = (y1 >= 0 && y1 < H);

#define ADD(cond, ww, xc, row)                                                                                         \
    if (cond) {                                                                                                        \
        float4 v = __ldg((row) + (long)(xc) * stride_row_v);                                                           \
        float w = (ww) * aw;                                                                                           \
        acc.x += w * v.x;                                                                                              \
        acc.y += w * v.y;                                                                                              \
        acc.z += w * v.z;                                                                                              \
        acc.w += w * v.w;                                                                                              \
    }
            if (yv0) {
                const float4 *row = val_lv + (long)y0 * W * stride_row_v;
                ADD(xv0, wy0 * wx0, x0, row)
                ADD(xv1, wy0 * wx1, x1, row)
            }
            if (yv1) {
                const float4 *row = val_lv + (long)y1 * W * stride_row_v;
                ADD(xv0, wy1 * wx0, x0, row)
                ADD(xv1, wy1 * wx1, x1, row)
            }
#undef ADD
        }
    }

    float4 *out_v = reinterpret_cast<float4 *>(out + (size_t)bqh * hidden_dim) + lane;
    *out_v = acc;
}

at::Tensor ms_deform_attn_forward(at::Tensor value, at::Tensor spatial_shapes, at::Tensor level_start_index,
                                  at::Tensor sampling_locations, at::Tensor attention_weights, int64_t im2col_step) {
    int B = value.size(0);
    int L_total = value.size(1);
    int n_heads = value.size(2);
    int hidden_dim = value.size(3);
    int num_queries = sampling_locations.size(1);
    int n_levels = spatial_shapes.size(0);
    int n_points = sampling_locations.size(4);

    auto out = at::empty({B, num_queries, n_heads * hidden_dim}, value.options());

    int total = B * num_queries * n_heads;
    int threads = 256;

    auto stream = at::cuda::getCurrentCUDAStream();
    auto launch = [&](auto vec_tag) {
        constexpr int VEC = decltype(vec_tag)::value;
        int lanes = hidden_dim / VEC;
        int groups_per_block = threads / lanes;
        int blocks = (total + groups_per_block - 1) / groups_per_block;
        ms_deform_attn_fwd_kernel<VEC><<<blocks, threads, 0, stream>>>(
            value.data_ptr<float>(), spatial_shapes.data_ptr<int64_t>(), level_start_index.data_ptr<int64_t>(),
            sampling_locations.data_ptr<float>(), attention_weights.data_ptr<float>(), out.data_ptr<float>(), B,
            L_total, n_heads, hidden_dim, num_queries, n_levels, n_points);
    };

    if (hidden_dim % 4 == 0)
        launch(std::integral_constant<int, 4>{});
    else if (hidden_dim % 2 == 0)
        launch(std::integral_constant<int, 2>{});
    else
        launch(std::integral_constant<int, 1>{});

    return out;
}
