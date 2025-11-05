// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/gemms/cublas_gemm.h"
#include "mlx/dtype_utils.h"
#include "mlx/utils.h"

#include <fmt/format.h>

namespace mlx::core {

namespace {

struct CublasPreference {
  CublasPreference(cu::Device& device) {
    // The recommended cublas workspace size is 4 MiB for pre-Hopper and 32 MiB
    // for Hopper+:
    // https://docs.nvidia.com/cuda/cublas/#cublassetworkspace

    uint64_t MiB = 1024 * 1024;
    uint64_t workspace_size =
        device.compute_capability_major() >= 9 ? 32 * MiB : 4 * MiB;

    // creates a matrix multiply heuristic search preferences descriptor
    // by allocating the memory needed to hold its opaque structure:

    CHECK_CUBLAS_ERROR(cublasLtMatmulPreferenceCreate(&pref_));
    CHECK_CUBLAS_ERROR(cublasLtMatmulPreferenceSetAttribute(
        pref_,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(uint64_t)));
  }

  ~CublasPreference() {
    CHECK_CUBLAS_ERROR(cublasLtMatmulPreferenceDestroy(pref_));
  }

  cublasLtMatmulPreference_t pref_{nullptr};
};

cublasLtMatmulPreference_t cublas_preference(cu::Device& device) {
  static CublasPreference pref(device);
  return pref.pref_;
}

} // namespace

CublasGemm::CublasQuantizedGemm(
    cu::Device& device,
    Dtype dtype,
    bool a_transposed,
    uint64_t a_rows,
    uint64_t a_cols,
    int64_t lda,
    bool b_transposed,
    uint64_t b_rows,
    uint64_t b_cols,
    int64_t ldb,
    int32_t batch_count,
    int64_t a_batch_stride,
    int64_t b_batch_stride)
    : handle_(device.lt_handle()),
      pref_(cublas_preference(device)),
      M_(a_rows),
      N_(b_cols) {
  heuristic_.state = CUBLAS_STATUS_NOT_INITIALIZED;

  // here CUBLAS_COMPUTE_32F is for operation descriptor
  // then operation precision is defined with matrix layout
  // [http://sanqian.synology.me:8418/zhangyiss/mlx/commit/a6d780154f2fe79e893045659d17fbace243802a?style=split&whitespace=show-all&show-outdated=]
  cublasComputeType_t gemm_compute_type = CUBLAS_COMPUTE_32F;
  CHECK_CUBLAS_ERROR(
      cublasLtMatmulDescCreate(&operationDesc, gemm_compute_type, CUDA_R_32F));

  // setting opA and opB for C = A @ B
  // note the swap due to column-major layout in cuBLAS
  // C^T = (A @ B)^T = B^T @ A^T

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

  // alpha, beta pointer mode set to device ? (TODO)
  cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_POINTER_MODE,
      &pointer_mode,
      sizeof(pointer_mode)));

  // scales:
  //  TODO this is just for NVFP4 for now, need to generalize later
  a_scale_mode_ = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  b_scale_mode_ = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;

  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_A_MATRIX_SCALE_TYPE,
      &a_scale_mode_,
      sizeof(a_scale_mode_)));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_B_MATRIX_SCALE_TYPE,
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
      b_cols, // m
      a_rows, // n
      b_cols));
}

CublasGemm::~CublasGemm() {
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(a_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(b_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(c_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(out_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescDestroy(matmul_desc_));
}

// void CublasGemm::set_out(
//     Dtype dtype,
//     bool transposed,
//     uint64_t rows,
//     uint64_t cols,
//     int64_t ld,
//     int32_t batch_count,
//     int64_t batch_stride) {
//   CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(out_desc_));
//   out_desc_ = create_matrix_layout(
//       dtype_to_cublas_type(dtype),
//       cols,
//       rows,
//       transposed,
//       ld,
//       batch_count,
//       batch_stride);
// }

// void CublasGemm::set_bias(cu::CommandEncoder& encoder, const array& bias) {
//   encoder.set_input_array(bias);
//   cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
//   CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
//       matmul_desc_,
//       CUBLASLT_MATMUL_DESC_EPILOGUE,
//       &epilogue,
//       sizeof(epilogue)));
//   auto* bias_ptr = bias.data<void>();
//   CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
//       matmul_desc_,
//       CUBLASLT_MATMUL_DESC_BIAS_POINTER,
//       &bias_ptr,
//       sizeof(bias_ptr)));
// }

void CublasGemm::run(
    cu::CommandEncoder& encoder,
    array& out,
    const array& a,
    const array& b,
    const Shape& batch_shape,
    const Strides& a_batch_strides,
    const Strides& b_batch_strides,
    float alpha) {
  int batch_count = out.size() / (M_ * N_);
  if (batch_count / batch_shape.back() > 1) {
    run_batched(
        encoder,
        out,
        a,
        b,
        batch_shape,
        a_batch_strides,
        b_batch_strides,
        alpha);
    return;
  }

  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);

  execute(
      encoder,
      out.data<void>(),
      a.data<void>(),
      b.data<void>(),
      nullptr,
      alpha);
}

void CublasGemm::run(
    cu::CommandEncoder& encoder,
    array& out,
    const array& a,
    const array& b,
    const array& c,
    const Shape& batch_shape,
    const Strides& a_batch_strides,
    const Strides& b_batch_strides,
    const Strides& c_batch_strides,
    float alpha,
    float beta) {
  int batch_count = out.size() / (M_ * N_);
  if (batch_count / batch_shape.back() > 1) {
    run_batched(
        encoder,
        out,
        a,
        b,
        c,
        batch_shape,
        a_batch_strides,
        b_batch_strides,
        c_batch_strides,
        alpha,
        beta);
    return;
  }

  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(c);
  encoder.set_output_array(out);

  execute(
      encoder,
      out.data<void>(),
      a.data<void>(),
      b.data<void>(),
      c.data<void>(),
      alpha,
      beta);
}

void CublasGemm::execute(
    cu::CommandEncoder& encoder,
    void* out,
    const void* a,
    const void* b,
    const void* c,
    float alpha /* = 1 */,
    float beta /* = 0 */) {
  if (heuristic_.state != CUBLAS_STATUS_SUCCESS) {
    int ret = 0;
    // using cached result from the heuristic search

    CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoGetHeuristic(
        handle_,
        matmul_desc_,
        a_desc_,
        b_desc_,
        c ? c_desc_ : out_desc_,
        out_desc_,
        pref_,
        1,
        &heuristic_,
        &ret));
    if (ret == 0) {
      throw std::runtime_error("Can not find algorithm for matmul.");
    }
  }

  const void* alpha_ptr = &alpha;
  const void* beta_ptr = &beta;
  complex64_t alpha_c, beta_c;
  if (scale_type_ == CUDA_C_32F) {
    alpha_c = complex64_t{alpha, 0.0f};
    beta_c = complex64_t{beta, 0.0f};
    alpha_ptr = &alpha_c;
    beta_ptr = &beta_c;
  }

  void* workspace_ptr = nullptr;
  if (heuristic_.workspaceSize > 0) {
    // Ensure workspace is 256-byte aligned
    int nbytes = cuda::ceil_div(heuristic_.workspaceSize, 256) * 256;
    array workspace(
        allocator::malloc(nbytes),
        {static_cast<int>(heuristic_.workspaceSize)},
        int8);
    encoder.add_temporary(workspace);
    workspace_ptr = workspace.data<void>();
  }

  auto capture = encoder.capture_context();
  CHECK_CUBLAS_ERROR(cublasLtMatmul(
      handle_,
      matmul_desc_,
      alpha_ptr,
      b, // a and b are swapped
      a_desc_,
      a,
      b_desc_,
      beta_ptr,
      c ? c : out,
      c ? c_desc_ : out_desc_,
      out,
      out_desc_,
      &heuristic_.algo,
      workspace_ptr,
      heuristic_.workspaceSize,
      encoder.stream()));
}

} // namespace mlx::core
