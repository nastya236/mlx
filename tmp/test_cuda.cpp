#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>
#include <string>

#include <iostream>
#include "mlx/backend/cuda/cuda.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/mlx.h"

namespace mx = mlx::core;

int main() {
  std::cout << "Default device" << mx::default_device() << std::endl;
  mx::array x = mx::ones({10}, mx::float32, mx::default_device());
  mx::eval(x);
  std::cout << "is contiguous: " << x.flags().row_contiguous << std::endl;
  std::cout << "Stream: " << s.id() << std::endl;
}