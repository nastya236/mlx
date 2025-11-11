#include <iostream>
#include "mlx/mlx.h"

namespace mx = mlx::core;

int main() {
  const char* rank_str = std::getenv("MLX_RANK");
  int rank = std::atoi(rank_str);
  mx::Device device(mx::Device::gpu, rank);
  mx::set_default_device(device);

  auto group = mx::distributed::init(/*strict=*/true, /*bk=*/"nccl");

  mx::array a = mx::ones({2}) * rank;
  std::cout << "Rank: " << rank << " A: " << a << std::endl;
  mx::array b = mx::distributed::all_gather(a, group);
  mx::eval(b);
  std::cout << "Rank: " << rank << " Result: " << b << std::endl;
  return 0;
}
