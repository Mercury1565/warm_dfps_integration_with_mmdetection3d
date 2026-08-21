#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

void FPSWithPreidxLauncher(int n, int m, int s, const float* dataset,
                           const int64_t* preidx, float* temp, int* idxs,
                           cudaStream_t stream);

void farthest_point_sampling_with_preidx_forward(torch::Tensor points_tensor,
                                                  torch::Tensor preidx_tensor,
                                                  torch::Tensor temp_tensor,
                                                  torch::Tensor idx_tensor,
                                                  int64_t n, int64_t m,
                                                  int64_t s) {
  TORCH_CHECK(points_tensor.is_cuda(), "points must be a CUDA tensor");
  TORCH_CHECK(preidx_tensor.is_cuda(), "preidx must be a CUDA tensor");
  TORCH_CHECK(points_tensor.is_contiguous(), "points must be contiguous");
  TORCH_CHECK(preidx_tensor.is_contiguous(), "preidx must be contiguous");
  TORCH_CHECK(points_tensor.scalar_type() == torch::kFloat32,
              "points must be float32");
  TORCH_CHECK(preidx_tensor.scalar_type() == torch::kInt64,
              "preidx must be int64");
  TORCH_CHECK(temp_tensor.scalar_type() == torch::kFloat32,
              "temp must be float32");
  TORCH_CHECK(idx_tensor.scalar_type() == torch::kInt32,
              "idx must be int32");
  TORCH_CHECK(s >= 1, "need at least one seed index");
  TORCH_CHECK(s <= m, "seed count must be <= num_samples");

  const float* dataset = points_tensor.data_ptr<float>();
  const int64_t* preidx = preidx_tensor.data_ptr<int64_t>();
  float* temp = temp_tensor.data_ptr<float>();
  int* idxs = idx_tensor.data_ptr<int>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  FPSWithPreidxLauncher(static_cast<int>(n), static_cast<int>(m),
                        static_cast<int>(s), dataset, preidx, temp, idxs,
                        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("farthest_point_sampling_with_preidx_forward",
        &farthest_point_sampling_with_preidx_forward,
        "Farthest point sampling continued from a pre-supplied seed set "
        "(CUDA)");
}
