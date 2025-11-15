// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/quantized/cublas_qqmm.h"
#include <cuda_fp8.h>
#include <fmt/format.h>
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/gemms/cublas_utils.h"
#include "mlx/dtype_utils.h"
#include "mlx/utils.h"

using fp8e4m3 = __nv_fp8_e4m3;

namespace mlx::core {

CublasQuantizedGemm::CublasQuantizedGemm(
    cu::Device& device,
    bool a_transposed,
    uint64_t a_rows,
    uint64_t a_cols,
    int64_t lda,
    bool b_transposed,
    uint64_t b_rows,
    uint64_t b_cols,
    int64_t ldb)
    // int32_t batch_count,
    // int64_t a_batch_stride,
    // int64_t b_batch_stride)
    : handle_(device.lt_handle()),
      pref_(cublas_utils::get_preference(device)),
      M_(a_rows),
      N_(b_transposed ? b_rows : b_cols) {
  heuristic_.state = CUBLAS_STATUS_NOT_INITIALIZED;

  cublasComputeType_t gemm_compute_type = CUBLAS_COMPUTE_32F;
  CHECK_CUBLAS_ERROR(
      cublasLtMatmulDescCreate(&matmul_desc_, gemm_compute_type, CUDA_R_32F));

  cublasOperation_t a_op = b_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_TRANSA,
      &a_op,
      sizeof(cublasOperation_t)));
  cublasOperation_t b_op = a_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_TRANSB,
      &b_op,
      sizeof(cublasOperation_t)));

  // alpha, beta pointer mode set to host ? (TODO)
  int32_t pointer_mode = CUBLASLT_POINTER_MODE_HOST;
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_POINTER_MODE,
      &pointer_mode,
      sizeof(int32_t)));

  // scales:
  //  TODO this is just for NVFP4 for now, need to generalize later
  a_scale_mode_ = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  b_scale_mode_ = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;

  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
      &a_scale_mode_,
      sizeof(a_scale_mode_)));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
      &b_scale_mode_,
      sizeof(b_scale_mode_)));

  // here i need to set CUBLASLT_MATMUL_DESC_A_SCALE_POINTERs

  // creating layouts for A, B, and output matrices
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutCreate(
      &a_desc_, CUDA_R_4F_E2M1, b_cols, b_rows, ldb));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutCreate(
      &b_desc_, CUDA_R_4F_E2M1, a_cols, a_rows, lda));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutCreate(
      &out_desc_,
      CUDA_R_16BF, // output in bf16
      b_transposed ? b_rows : b_cols, // m
      a_rows, // asume that never transposed (supported only TN layout)
      b_transposed ? b_rows : b_cols));
}

CublasQuantizedGemm::~CublasQuantizedGemm() {
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(a_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(b_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(c_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(out_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescDestroy(matmul_desc_));
}

void CublasQuantizedGemm::run(
    cu::CommandEncoder& encoder,
    array& out,
    const array& a,
    const array& b,
    const array& a_scale,
    const array& b_scale,
    // const Shape& batch_shape,
    // const Strides& a_batch_strides,
    // const Strides& b_batch_strides,
    float alpha) {
  // int batch_count = out.size() / (M_ * N_);
  // if (batch_count / batch_shape.back() > 1) {
  //   run_batched(
  //       encoder,
  //       out,
  //       a,
  //       b,
  //       batch_shape,
  //       a_batch_strides,
  //       b_batch_strides,
  //       alpha);
  //   return;
  // }
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(a_scale);
  encoder.set_input_array(b_scale);
  encoder.set_output_array(out);

  execute(
      encoder,
      gpu_ptr<void>(out),
      gpu_ptr<void>(a),
      gpu_ptr<void>(b),
      gpu_ptr<void>(a_scale),
      gpu_ptr<void>(b_scale),
      nullptr,
      alpha);
}

void CublasQuantizedGemm::execute(
    cu::CommandEncoder& encoder,
    void* out,
    const void* a,
    const void* b,
    const void* a_scale,
    const void* b_scale,
    const void* c,
    float alpha /* = 1 */,
    float beta /* = 0 */) {
  // Set scale pointers (quantized-specific)
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
      &b_scale,
      sizeof(b_scale)));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
      &a_scale,
      sizeof(a_scale)));

  const void* alpha_ptr = &alpha;
  const void* beta_ptr = &beta;

  // Use shared matmul execution
  cublas_utils::execute_matmul(
      encoder,
      handle_,
      matmul_desc_,
      a_desc_,
      b_desc_,
      c_desc_,
      out_desc_,
      heuristic_,
      pref_,
      out,
      a,
      b,
      c,
      alpha_ptr,
      beta_ptr);
}

} // namespace mlx::core
