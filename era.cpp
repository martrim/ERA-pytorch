#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> era_cuda_forward(
    torch::Tensor input,
    torch::Tensor num_weights,
    torch::Tensor denom_weights);

std::vector<torch::Tensor> era_cuda_backward(
    torch::Tensor input,
    torch::Tensor num_weights,
    torch::Tensor new_cell,
    torch::Tensor denom_weights);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> era_forward(
    torch::Tensor input,
    torch::Tensor num_weights,
    torch::Tensor denom_weights) {
  CHECK_INPUT(input);
  CHECK_INPUT(num_weights);
  CHECK_INPUT(denom_weights);

  return era_cuda_forward(input, num_weights, denom_weights);
}

std::vector<torch::Tensor> era_backward(
    torch::Tensor input,
    torch::Tensor num_weights,
    torch::Tensor denom_weights,) {
  CHECK_INPUT(input);
  CHECK_INPUT(num_weights);
  CHECK_INPUT(denom_weights);

  return era_cuda_backward(
      input,
      num_weights,
      denom_weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &era_forward, "ERA forward (CUDA)");
  m.def("backward", &era_backward, "ERA backward (CUDA)");
}