// Host-side wrapper + pybind registration for the declumping kernel
// (thin_redundant_kernel.cu). CUDA-only, no multi-backend dispatch.

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

void ThinRedundantLauncher(int m, const float* d2, const int64_t* order,
                           const float* sep_sq, int* keep,
                           cudaStream_t stream);

void thin_redundant_forward(torch::Tensor d2_tensor, torch::Tensor order_tensor,
                            torch::Tensor sep_sq_tensor, torch::Tensor keep_tensor,
                            int64_t m) {
  TORCH_CHECK(d2_tensor.is_cuda(), "d2 must be a CUDA tensor");
  TORCH_CHECK(order_tensor.is_cuda(), "order must be a CUDA tensor");
  TORCH_CHECK(sep_sq_tensor.is_cuda(), "sep_sq must be a CUDA tensor");
  TORCH_CHECK(keep_tensor.is_cuda(), "keep must be a CUDA tensor");
  TORCH_CHECK(d2_tensor.is_contiguous(), "d2 must be contiguous");
  TORCH_CHECK(order_tensor.is_contiguous(), "order must be contiguous");
  TORCH_CHECK(sep_sq_tensor.is_contiguous(), "sep_sq must be contiguous");
  TORCH_CHECK(keep_tensor.is_contiguous(), "keep must be contiguous");
  TORCH_CHECK(d2_tensor.scalar_type() == torch::kFloat32, "d2 must be float32");
  TORCH_CHECK(order_tensor.scalar_type() == torch::kInt64, "order must be int64");
  TORCH_CHECK(sep_sq_tensor.scalar_type() == torch::kFloat32, "sep_sq must be float32");
  TORCH_CHECK(keep_tensor.scalar_type() == torch::kInt32, "keep must be int32");

  const float* d2 = d2_tensor.data_ptr<float>();
  const int64_t* order = order_tensor.data_ptr<int64_t>();
  const float* sep_sq = sep_sq_tensor.data_ptr<float>();
  int* keep = keep_tensor.data_ptr<int>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ThinRedundantLauncher(static_cast<int>(m), d2, order, sep_sq, keep, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("thin_redundant_forward", &thin_redundant_forward,
        "Greedy declumping of carried samples (CUDA)");
}
