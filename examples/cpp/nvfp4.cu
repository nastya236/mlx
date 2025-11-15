#include "mlx/mlx.h"
#include "mlx/backend/cuda/quantized/cublas_qqmm.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/stream.h"
#include <iostream>


namespace mx = mlx::core;

int main() {

  // pick GPU 0 (or read MLX_RANK, etc.)
  mx::Device device(mx::Device::gpu, 0);
  auto s = mx::default_stream(device);
  auto& encoder = mx::cu::get_command_encoder(s);

  mx::array a = mx::random::uniform({128, 256}, mx::bfloat16);  // (M, K)
  mx::array b = mx::random::uniform({128, 256}, mx::bfloat16);  // (N, K)


  mx::array c = mx::matmul(a, mx::transpose(b));

  std::cout << "Reference: " << c << std::endl;
  auto scaled_a = mx::quantize(a, 16, 4, "nvfp4");
  auto scaled_b = mx::quantize(b, 16, 4, "nvfp4");

  mx::array a_quantized = scaled_a[0];
  mx::array a_scale = scaled_a[1];
  mx::array b_quantized = scaled_b[0];
  mx::array b_scale = scaled_b[1];

  bool a_transposed = false, b_transposed = true;
  uint64_t M = a.shape(0), K = a.shape(1), N = b.shape(0);

  int64_t lda = K;
  int64_t ldb = K;

  mx::eval(a_quantized, b_quantized, a_scale, b_scale);
  mx::array out = mx::zeros({128, 128}, mx::bfloat16);
  mx::eval(out);
  
  mx::CublasQuantizedGemm gemm(
      encoder.device(),
      a_transposed,
      M, // rows of A
      K, // cols of A
      lda,
      b_transposed,
      N, // rows of B
      K, // cols of B
      ldb);
  
  gemm.run(
      encoder,
      out,
      a_quantized,
      b_quantized,
      a_scale,
      b_scale,
      1.0f);

  std::cout << "Output: " << out << std::endl;

  mx::array a_dequantized = mx::dequantize(a_quantized, a_scale, {}, 16, 4, "nvfp4");
  mx::array b_dequantized = mx::dequantize(b_quantized, b_scale, {}, 16, 4, "nvfp4");

  mx::array reference_deq = mx::matmul(a_dequantized, mx::transpose(b_dequantized));
  std::cout << "Reference dequantized: " << reference_deq << std::endl;

  return 0;
}