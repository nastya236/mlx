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

  mx::array a = 1e-2 * mx::ones({1400000}) * rank;
  mx::eval(a);
  std::cout << "Rank: " << rank << " A: " << a << std::endl;
  // std::cout << "If contigous: " << a.flags().row_contiguous << std::endl;
  mx::array b = mx::distributed::all_sum(a, group);
  // std::cout << b.size() << std::endl;
  mx::eval(b);
  std::cout << "Rank: " << rank << " Result: " << b << std::endl;
  return 0;
}
