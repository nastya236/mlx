#include "mlx/mlx.h"
#include "mlx/backend/cuda/quantized/cublas_qqmm.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/stream.h"
#include <cuda_fp8.h>

namespace mx = mlx::core;

int main() {
  // pick GPU 0 (or read MLX_RANK, etc.)
  mx::Device device(mx::Device::gpu, 0);
  auto s = mx::default_stream(device);
  auto& encoder = mx::cu::get_command_encoder(s);

  mx::array a = mx::random::uniform({32, 32});
  mx::array b = mx::random::uniform({32, 32});

  //along the reduction dimension

  auto scaled_a = mx::quantize(a, 16, 4, "nvfp4");
  auto scaled_b = mx::quantize(b, 16, 4, "nvfp4");

  mx::array a_quantized = scaled_a[0];
  mx::array a_scale = scaled_a[1];
  mx::array b_quantized = scaled_b[0];
  mx::array b_scale = scaled_b[1];

  bool a_transposed = false, b_transposed = true;
  uint64_t M = a.shape(0), K = a.shape(1), N = b.shape(1);

  int64_t lda = a_transposed ? M : K;
  int64_t ldb = b_transposed ? K : N;

  mx::eval(a_quantized, a_scale, b_quantized, b_scale);
  mx::array out = mx::zeros({32, 32});
  mx::eval(out);
  
  mx::CublasQuantizedGemm gemm(
      encoder.device(),
      a_transposed,
      M,
      K,
      lda,
      b_transposed,
      K,
      N,
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
  return 0;
}