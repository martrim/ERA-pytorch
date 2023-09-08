#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

std::vector<torch::Tensor> era_cuda_forward(
    torch::Tensor input,
    torch::Tensor num_weights,
    torch::Tensor denom_weights) {
  const auto batch_size = input.size(0);
  const auto input_size = input.size(1);

  /*f: numerator, g: denominator*/
  auto f = torch::zeros_like(input);
  auto g = torch::zeros_like(input);

  const int threads = 1024;
  const dim3 blocks((input_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "era_forward_cuda", ([&] {
    era_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        f.data<scalar_t>(),
        g.data<scalar_t>(),
        input_size);
  }));

  return {f, g};
}
