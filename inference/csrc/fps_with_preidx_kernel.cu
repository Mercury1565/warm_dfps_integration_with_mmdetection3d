#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ void update_max(float* __restrict__ dists,
                                            int* __restrict__ dists_i,
                                            int a, int b) {
  if (dists[b] > dists[a]) {
    dists[a] = dists[b];
    dists_i[a] = dists_i[b];
  }
}

template <unsigned int block_size>
__global__ void fps_with_preidx_kernel(int n, int m, int s,
                                       const float* __restrict__ dataset,
                                       const int64_t* __restrict__ preidx,
                                       float* __restrict__ temp,
                                       int* __restrict__ idxs) {
  // dataset: (n, 3)   preidx: (s,)   temp: (n,) workspace   idxs: (m,) output
  __shared__ float dists[block_size];
  __shared__ int dists_i[block_size];

  const int tid = threadIdx.x;
  const int stride = block_size;

  // 1. seeds are already chosen -- just place them.
  for (int i = tid; i < s; i += stride) {
    idxs[i] = static_cast<int>(preidx[i]);
  }
  __syncthreads();

  // 2. fold every seed's distance into temp before any picking happens.
  for (int si = 0; si < s; si++) {
    const int seed = idxs[si];
    const float x1 = dataset[seed * 3 + 0];
    const float y1 = dataset[seed * 3 + 1];
    const float z1 = dataset[seed * 3 + 2];
    for (int k = tid; k < n; k += stride) {
      const float dx = dataset[k * 3 + 0] - x1;
      const float dy = dataset[k * 3 + 1] - y1;
      const float dz = dataset[k * 3 + 2] - z1;
      const float d = dx * dx + dy * dy + dz * dz;
      if (d < temp[k]) temp[k] = d;
    }
  }
  __syncthreads();

  // 3. standard greedy loop, continuing from the seeded temp/idxs.
  for (int j = s; j < m; j++) {
    int besti = 0;
    float best = -1.0f;
    for (int k = tid; k < n; k += stride) {
      const float d2 = temp[k];
      if (d2 > best) {
        best = d2;
        besti = k;
      }
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

#pragma unroll
    for (int t = 1024; t >= 2; t >>= 1) {
      const int half = t / 2;
      if (block_size >= t && tid < half) {
        update_max(dists, dists_i, tid, tid + half);
      }
      __syncthreads();
    }

    const int old = dists_i[0];
    if (tid == 0) idxs[j] = old;
    __syncthreads();

    const float x1 = dataset[old * 3 + 0];
    const float y1 = dataset[old * 3 + 1];
    const float z1 = dataset[old * 3 + 2];
    for (int k = tid; k < n; k += stride) {
      const float dx = dataset[k * 3 + 0] - x1;
      const float dy = dataset[k * 3 + 1] - y1;
      const float dz = dataset[k * 3 + 2] - z1;
      const float d = dx * dx + dy * dy + dz * dz;
      if (d < temp[k]) temp[k] = d;
    }
    __syncthreads();
  }
}

inline unsigned int opt_n_threads(int work_size) {
  const int pow_2 = static_cast<int>(
      std::log(static_cast<double>(work_size)) / std::log(2.0));
  int threads = 1 << pow_2;
  if (threads > 1024) threads = 1024;
  if (threads < 1) threads = 1;
  return static_cast<unsigned int>(threads);
}

}  // namespace

void FPSWithPreidxLauncher(int n, int m, int s, const float* dataset,
                           const int64_t* preidx, float* temp, int* idxs,
                           cudaStream_t stream) {
  const unsigned int n_threads = opt_n_threads(n);

#define LAUNCH(NT)                                                          \
  fps_with_preidx_kernel<NT><<<1, NT, 0, stream>>>(n, m, s, dataset, preidx, \
                                                    temp, idxs)

  switch (n_threads) {
    case 1024: LAUNCH(1024); break;
    case 512:  LAUNCH(512);  break;
    case 256:  LAUNCH(256);  break;
    case 128:  LAUNCH(128);  break;
    case 64:   LAUNCH(64);   break;
    case 32:   LAUNCH(32);   break;
    case 16:   LAUNCH(16);   break;
    case 8:    LAUNCH(8);    break;
    case 4:    LAUNCH(4);    break;
    case 2:    LAUNCH(2);    break;
    case 1:    LAUNCH(1);    break;
    default:   LAUNCH(512);
  }
#undef LAUNCH
}
