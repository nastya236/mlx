// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/quantized/cublas_qqmm.h"
#include "mlx/dtype_utils.h"
#include "mlx/utils.h"
#include <cuda_fp8.h>
#include <fmt/format.h>

#define CHECK_LT(expr) do {                                \
  cublasStatus_t _s = (expr);                              \
  if (_s != CUBLAS_STATUS_SUCCESS) {                       \
    fprintf(stderr, "[cuBLASLt] %s failed: %d at %s:%d\n", \
            #expr, int(_s), __FILE__, __LINE__);           \
    throw std::runtime_error("cuBLASLt error");            \
  }                                                        \
} while(0)

namespace {

inline void print_desc(const char* tag, cublasLtMatrixLayout_t d) {
  int64_t rows, cols, ld; cublasLtOrder_t order; size_t sz; int dtype;
  CHECK_LT(cublasLtMatrixLayoutGetAttribute(d, CUBLASLT_MATRIX_LAYOUT_ROWS,  &rows, sizeof(rows), &sz));
  CHECK_LT(cublasLtMatrixLayoutGetAttribute(d, CUBLASLT_MATRIX_LAYOUT_COLS,  &cols, sizeof(cols), &sz));
  CHECK_LT(cublasLtMatrixLayoutGetAttribute(d, CUBLASLT_MATRIX_LAYOUT_LD,    &ld,   sizeof(ld),   &sz));
  CHECK_LT(cublasLtMatrixLayoutGetAttribute(d, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,sizeof(order),&sz));
  CHECK_LT(cublasLtMatrixLayoutGetAttribute(d, CUBLASLT_MATRIX_LAYOUT_TYPE,  &dtype,sizeof(dtype),&sz));
  fprintf(stderr, "%s: rows=%ld cols=%ld ld=%ld order=%d type=%d\n",
          tag, (long)rows, (long)cols, (long)ld, (int)order, dtype);
}

inline void print_op(cublasLtMatmulDesc_t op) {
  cublasOperation_t ta, tb; cublasLtPointerMode_t pm; size_t sz;
  CHECK_LT(cublasLtMatmulDescGetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta), &sz));
  CHECK_LT(cublasLtMatmulDescGetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb), &sz));
  CHECK_LT(cublasLtMatmulDescGetAttribute(op, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm), &sz));
  fprintf(stderr, "op: transA=%d transB=%d pointer_mode=%d\n", ta, tb, pm);
  int a_mode, b_mode;
  if (cublasLtMatmulDescGetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &a_mode, sizeof(a_mode), &sz) == CUBLAS_STATUS_SUCCESS)
    fprintf(stderr, "A_SCALE_MODE=%d\n", a_mode);
  if (cublasLtMatmulDescGetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &b_mode, sizeof(b_mode), &sz) == CUBLAS_STATUS_SUCCESS)
    fprintf(stderr, "B_SCALE_MODE=%d\n", b_mode);
}

} // anonymous namespace


using fp8e4m3 = __nv_fp8_e4m3;

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
      pref_(cublas_preference(device)),
      M_(a_rows),
      N_(b_cols) {
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
      b_cols, // m
      a_rows, // n
      b_cols));

  print_desc("A'", a_desc_);
  print_desc("B'", b_desc_);
  print_desc("D (C^T)", out_desc_);
  print_op(matmul_desc_);
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

  // set scale pointers
  const fp8e4m3* a_scale_ptr = reinterpret_cast<const fp8e4m3*>(a_scale);
  const fp8e4m3* b_scale_ptr = reinterpret_cast<const fp8e4m3*>(b_scale);

  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
      &a_scale,
      sizeof(a_scale)));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
      &b_scale,
      sizeof(b_scale)));

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
  // complex64_t alpha_c, beta_c;
  // if (scale_type_ == CUDA_C_32F) {
  //   alpha_c = complex64_t{alpha, 0.0f};
  //   beta_c = complex64_t{beta, 0.0f};
  //   alpha_ptr = &alpha_c;
  //   beta_ptr = &beta_c;
  // }

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
