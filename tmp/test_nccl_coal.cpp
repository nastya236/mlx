#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>
#include "mlx/backend/cuda/device.h"
#include "mlx/distributed/distributed.h"
#include "mlx/mlx.h"

namespace mx = mlx::core;

int main() {
  // Set the default device to GPU
  const char* rank_str = std::getenv("MLX_RANK");
  int rank = std::atoi(rank_str);
  mx::Device device(mx::Device::gpu, rank);
  mx::set_default_device(device);

  auto group = mx::distributed::init(/*strict=*/true, /*bk=*/"nccl");

  std::vector<mx::array> arrays;
  for (int i = 0; i < 3; ++i) {
    arrays.push_back(1e-2 * mx::ones({10}) * rank);
  }
  mx::eval(arrays);
  std::vector<mx::array> results = mx::distributed::all_sum_coalesced(arrays, group);
  mx::eval(results);
  for (const auto& a : results) {
    std::cout << "Rank: " << rank << " Output: " << a << std::endl;
  }
}
