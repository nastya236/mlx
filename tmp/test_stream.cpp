
#include <iostream>

#include "mlx/backend/metal/metal.h"
#include "mlx/distributed/ops.h"
#include "mlx/distributed/primitives.h"

inline mlx::core::Device get_device() {
    return mlx::core::metal::is_available()
           ? mlx::core::Device::cpu
           : mlx::core::Device::gpu;
  }
  

int main() {
    auto device = get_device();
    std::cout << "Default device: " << device << std::endl;
    return 0;
}