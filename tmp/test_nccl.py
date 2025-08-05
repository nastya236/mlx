import os 
import sys
sys.path.append('/mnt/task_runtime/python')
import mlx.core as mx
from mlx import nn

print()
def main():
    rank = int(os.environ['MLX_RANK'])
    print(f"Hello from rank {rank}!")
    mx.set_default_device(mx.Device(mx.gpu, rank))
    world = mx.distributed.init(strict=True, backend="nccl")
    for i in range(100):
        A = mx.random.uniform(shape=(1024, 1024))
        B = mx.random.uniform(shape=(1024, 1024))
        activation = nn.ReLU()
        x = mx.random.uniform(shape=(1024, ))
        y = B @ (activation(A @ x))
        y = mx.distributed.all_sum(y, stream=mx.gpu)
        mx.eval(y)
        print(f"Iteration {i}: Rank {rank} has data: {y}")

    # print(f"Rank {rank} has data: {y}")

if __name__ == "__main__":
    main()


