import os
import sys

sys.path.append("/mnt/task_runtime/python")
import mlx.core as mx

from mlx import nn

print()


def main():
    rank = int(os.environ["MLX_RANK"])
    print(f"Hello from rank {rank}!")
    mx.set_default_device(mx.Device(mx.gpu, rank))
    world = mx.distributed.init(strict=True, backend="nccl")
    x = mx.ones((10, 10), dtype=mx.float32) * (rank + 1)
    y = mx.distributed.all_max(x)
    print(f"Rank {rank} max value: {y}")


if __name__ == "__main__":
    main()
