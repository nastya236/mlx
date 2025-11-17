// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/quantized/cublas_qqmm.h"
#include "mlx/backend/cuda/quantized/qqmm_utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/fast_primitives.h"

#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace {

inline array ensure_row_contiguous(
    const array& x,
    cu::CommandEncoder& enc,
    const Stream& s) {
  if (!x.flags().row_contiguous) {
    array x_copy = contiguous_copy_gpu(x, s);
    enc.add_temporary(x_copy);
    return x_copy;
  } else {
    return x;
  }
}

inline array ensure_row_contiguous_matrix(
    const array& x,
    cu::CommandEncoder& enc,
    const Stream& s) {
  if (x.ndim() < 2) {
    if (x.strides()[0] == 1) {
      return x;
    }
  } else {
    auto stride_0 = x.strides()[x.ndim() - 2];
    auto stride_1 = x.strides()[x.ndim() - 1];
    if (stride_0 == x.shape(-1) && stride_1 == 1) {
      return x;
    }
  }
  array x_copy = contiguous_copy_gpu(x, s);
  enc.add_temporary(x_copy);
  return x_copy;
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
  // Compute padded dimensions for full tiles (128 rows × 4 cols)
  auto [padded_M, padded_cols_a] = get_padded_scale_dims(M, K, group_size);
  auto [padded_N, padded_cols_b] = get_padded_scale_dims(N, K, group_size);

  // When tensor dimensions are not multiples of the tile size above,
  // it is necessary to still allocate full tile for storage and fill
  // out of bounds values with zeroes.
  // https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

  array scale_a_tiled(
      {static_cast<int>(padded_M), static_cast<int>(padded_cols_a)}, uint8);
  array scale_b_tiled(
      {static_cast<int>(padded_N), static_cast<int>(padded_cols_b)}, uint8);

  scale_a_tiled.set_data(
      cu::malloc_async(scale_a_tiled.nbytes(), encoder.stream()));
  scale_b_tiled.set_data(
      cu::malloc_async(scale_b_tiled.nbytes(), encoder.stream()));

  // Repack scales from linear to tiled layout
  // Kernel will zero-fill out-of-bounds regions
  repack_scales(scale_a, scale_a_tiled, M, K, group_size, encoder, s);
  repack_scales(scale_b, scale_b_tiled, N, K, group_size, encoder, s);
  encoder.add_temporary(scale_a_tiled);
  encoder.add_temporary(scale_b_tiled);

  return {scale_a_tiled, scale_b_tiled};
}

} // namespace

void fast::Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("Quantize::eval_gpu");
  auto& s = stream();
  auto& d = cu::device(s.device);
  auto& enc = d.get_command_encoder(s);

  if (dequantize_) {
    auto wq = ensure_row_contiguous(inputs[0], enc, s);
    auto scales = ensure_row_contiguous(inputs[1], enc, s);
    auto& w = outputs[0];

    w.set_data(cu::malloc_async(w.nbytes(), enc.stream()));

    if (mode_ == QuantizationMode::Affine) {
      auto biases = ensure_row_contiguous(inputs[2], enc, s);
      affine_dequantize(wq, scales, biases, w, group_size_, bits_, enc, s);
    } else {
      fp_dequantize(wq, scales, w, group_size_, bits_, enc, s);
    }
  } else {
    auto w = ensure_row_contiguous(inputs[0], enc, s);
    auto& wq = outputs[0];
    auto& scales = outputs[1];

    wq.set_data(cu::malloc_async(wq.nbytes(), enc.stream()));
    scales.set_data(cu::malloc_async(scales.nbytes(), enc.stream()));
    if (mode_ == QuantizationMode::Affine) {
      auto& biases = outputs[2];
      biases.set_data(cu::malloc_async(biases.nbytes(), enc.stream()));
      affine_quantize(w, wq, scales, biases, group_size_, bits_, enc, s);
    } else {
      fp_quantize(w, wq, scales, group_size_, bits_, enc, s);
    }
  }
}

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
      encoder.device(), a_transposed, M, K, lda, b_transposed, N, K, ldb);

  qqmm.run(encoder, out, a, b, a_scale, b_scale, alpha);
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
