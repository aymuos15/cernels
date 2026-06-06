#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstring>
#include <vector>

// Block-bitmask NMS (the classic torchvision/Detectron approach). Boxes must be
// pre-sorted by score (descending). Each block compares a 64-box row tile against a
// 64-box col tile and sets a bit per suppressed pair; the host then reduces the mask.
#define TPB 64

__device__ inline float iou_(const float *a, const float *b) {
    float l = fmaxf(a[0], b[0]), t = fmaxf(a[1], b[1]);
    float r = fminf(a[2], b[2]), d = fminf(a[3], b[3]);
    float inter = fmaxf(0.f, r - l) * fmaxf(0.f, d - t);
    float aa = (a[2] - a[0]) * (a[3] - a[1]), ab = (b[2] - b[0]) * (b[3] - b[1]);
    return inter / (aa + ab - inter);
}

__global__ void nms_kernel(const float *__restrict__ boxes, int n, float thresh, unsigned long long *mask) {
    int row = blockIdx.y, col = blockIdx.x;
    int row_size = min(n - row * TPB, TPB), col_size = min(n - col * TPB, TPB);
    __shared__ float blk[TPB * 4];
    if (threadIdx.x < col_size)
        for (int k = 0; k < 4; k++)
            blk[threadIdx.x * 4 + k] = boxes[(col * TPB + threadIdx.x) * 4 + k];
    __syncthreads();
    if (threadIdx.x < row_size) {
        int cur = row * TPB + threadIdx.x;
        const float *cb = boxes + cur * 4;
        unsigned long long t = 0;
        int start = (row == col) ? threadIdx.x + 1 : 0;
        for (int i = start; i < col_size; i++)
            if (iou_(cb, blk + i * 4) > thresh)
                t |= (1ULL << i);
        int col_blocks = (n + TPB - 1) / TPB;
        mask[(long)cur * col_blocks + col] = t;
    }
}

at::Tensor nms(at::Tensor boxes, double thresh) {
    int n = boxes.size(0);
    int col_blocks = (n + TPB - 1) / TPB;
    auto opts = boxes.options().dtype(at::kLong);
    auto mask = at::zeros({(long)n * col_blocks}, opts); // int64 storage used as uint64
    dim3 blocks(col_blocks, col_blocks);
    nms_kernel<<<blocks, TPB, 0, at::cuda::getCurrentCUDAStream()>>>(
        boxes.data_ptr<float>(), n, (float)thresh, reinterpret_cast<unsigned long long *>(mask.data_ptr<int64_t>()));

    auto mask_cpu = mask.to(at::kCPU);
    auto mh = reinterpret_cast<unsigned long long *>(mask_cpu.data_ptr<int64_t>());
    std::vector<unsigned long long> remv(col_blocks, 0ULL);
    std::vector<int64_t> keep;
    for (int i = 0; i < n; i++) {
        int nb = i / TPB, ib = i % TPB;
        if (!(remv[nb] & (1ULL << ib))) {
            keep.push_back(i);
            for (int j = nb; j < col_blocks; j++)
                remv[j] |= mh[(long)i * col_blocks + j];
        }
    }
    auto out = at::empty({(long)keep.size()}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    std::memcpy(out.data_ptr<int64_t>(), keep.data(), keep.size() * sizeof(int64_t));
    return out; // CPU int64: indices into the (score-sorted) boxes
}
