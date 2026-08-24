// CUDA kernel for warm_dfps_manager._thin_redundant's greedy declumping
// pass: among carried samples, drop the lower-occupancy member of any
// too-close pair.
//
// Structurally a cousin of fps_with_preidx_kernel's greedy loop, but
// simpler in one respect and different in another:
//   - The processing order is entirely fixed up front (argsort(occupancy)
//     doesn't change as the loop runs), unlike FPS's argmax, which is
//     recomputed fresh every iteration. So no reduction is needed to pick
//     "which sample is next" -- that's precomputed on the host, once.
//   - Each iteration instead needs a reduction to answer "does any
//     still-kept neighbor fall within this sample's separation
//     threshold" -- an OR-reduction instead of FPS's max-reduction, but
//     the same block-wide shared-memory tree pattern.

#include <cstdint>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

template <unsigned int block_size>
__global__ void thin_redundant_kernel(int m,
                                      const float* __restrict__ d2,      // (m, m), diagonal already inf
                                      const int64_t* __restrict__ order, // (m,) argsort(occupancy)
                                      const float* __restrict__ sep_sq,  // (m,)
                                      int* __restrict__ keep) {          // (m,) mutated in place
  __shared__ int found_shared[block_size];

  const int tid = threadIdx.x;
  const int stride = block_size;

  for (int idx = 0; idx < m; idx++) {
    const int i = static_cast<int>(order[idx]);
    if (!keep[i]) continue;  // uniform across all threads -- same value, same branch, no divergence

    const float thresh = sep_sq[i];
    const float* row = d2 + i * m;

    int found = 0;
    for (int j = tid; j < m; j += stride) {
      if (keep[j] && row[j] < thresh) {
        found = 1;
        break;
      }
    }
    found_shared[tid] = found;
    __syncthreads();

#pragma unroll
    for (int t = 1024; t >= 2; t >>= 1) {
      const int half = t / 2;
      if (block_size >= t && tid < half) {
        found_shared[tid] = found_shared[tid] | found_shared[tid + half];
      }
      __syncthreads();
    }

    if (tid == 0 && found_shared[0]) {
      keep[i] = 0;
    }
    __syncthreads();  // must complete before the next idx reads keep[]
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

void ThinRedundantLauncher(int m, const float* d2, const int64_t* order,
                           const float* sep_sq, int* keep,
                           cudaStream_t stream) {
  const unsigned int n_threads = opt_n_threads(m);

#define LAUNCH(NT) \
  thin_redundant_kernel<NT><<<1, NT, 0, stream>>>(m, d2, order, sep_sq, keep)

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
