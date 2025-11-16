// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/matmul.h"
#include "mlx/backend/cuda/cublas_utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/quantized/cublas_qqmm.h"
#include "mlx/backend/cuda/quantized/matmul.h"
#include "mlx/backend/cuda/quantized/qqmm_utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>
#include <numeric>

namespace mlx::core {

namespace {

void qqmm_impl(
    cu::CommandEncoder& encoder,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    bool b_transposed,
    int64_t ldb,
    array& out,
    const array& a,
    const array& b,
    const array& a_scale,
    const array& b_scale,
    float alpha = 1.0f) {
  // Invoke CublasQQMM for quantized matrix multiplication
  CublasQQMM qqmm(
      encoder.device(), a_transposed, M, K, lda, b_transposed, K, N, ldb);

  qqmm.run(encoder, out, a, b, a_scale, b_scale, alpha);
}

std::pair<array, array> repack_for_tensor_cores(
    const array& scale_a,
    const array& scale_b,
    int M,
    int N,
    int K,
    int group_size,
    cu::CommandEncoder& encoder,
    const Stream& s) {
  array scale_a_tiled(scale_a.shape(), uint8);
  array scale_b_tiled(scale_b.shape(), uint8);

  scale_a_tiled.set_data(
      cu::malloc_async(scale_a_tiled.nbytes(), encoder.stream()));
  scale_b_tiled.set_data(
      cu::malloc_async(scale_b_tiled.nbytes(), encoder.stream()));

  // Repack scales from linear to tiled layout
  repack_scales(scale_a, scale_a_tiled, M, K, group_size, encoder, s);
  repack_scales(scale_b, scale_b_tiled, N, K, group_size, encoder, s);
  encoder.add_temporary(scale_a_tiled);
  encoder.add_temporary(scale_b_tiled);

  return {scale_a_tiled, scale_b_tiled};
}

} // namespace

void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("QQMatmul::eval_gpu");

  // TODO: for now minimalistic implementation without batching support
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  assert(inputs.size() == 4);
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto& scale_a_pre = inputs[2];
  auto& scale_b_pre = inputs[3];
  // Return 0s if either input is empty.
  if (a.size() == 0 || b.size() == 0) {
    array zero(0, a.dtype());
    encoder.add_temporary(zero);
    fill_gpu(zero, out, s);
    return;
  }
  out.set_data(cu::malloc_async(out.nbytes(), encoder.stream()));

  int M = a.shape(-2);
  int N = b.shape(-2); // b always transposed
  int K = a.shape(-1);

  // Repack scales from linear to tiled layout for tensor cores
  auto [scale_a_tiled, scale_b_tiled] = repack_for_tensor_cores(
      scale_a_pre, scale_b_pre, M, N, K, group_size_, encoder, s);

  // Set transpose flags and leading dimensions for TN layout
  bool a_transposed = false; // a is normal (M x K)
  bool b_transposed = true; // b is transposed (N x K -> K x N)
  int64_t lda = K; // Leading dimension of a
  int64_t ldb = K; // Leading dimension of b

  qqmm_impl(
      encoder,
      M,
      N,
      K,
      a_transposed,
      lda,
      b_transposed,
      ldb,
      out,
      a,
      b,
      scale_a_tiled,
      scale_b_tiled);
}

} // namespace mlx::core
