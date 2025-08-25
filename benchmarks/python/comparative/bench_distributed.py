import argparse
import os
import time
from html import parser

import mlx.core as mx
import torch
import torch.distributed as dist

N_ITER = 1000


def time_fn(fn, x, is_mx=True, *args, **kwargs):

    start = time.perf_counter()
    for _ in range(N_ITER):
        if is_mx:
            out = fn(x)
        else:
            out = fn(x, *args, **kwargs)
    if is_mx:
        eval(out)
    end = time.perf_counter()
    return 1e3 * (end - start) / N_ITER


def mlx_all_sum():
    group = mx.distributed.init(backend="nccl")
    dev = mx.Device(mx.gpu, group.rank())
    mx.set_default_device(dev)
    x = mx.random.uniform(shape=SHAPE)

    return time_fn(mx.distributed.all_sum, x, is_mx=True)


def torch_all_sum():
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    addr = os.environ["MASTER_ADDR"]
    port = os.environ["MASTER_PORT"]
    init_method = f"tcp://{addr}:{port}"

    dist.init_process_group(
        # backend='nccl',
        backend="gloo",
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    # torch.cuda.set_device(rank)

    x = torch.rand(SHAPE, device="cuda" if torch.cuda.is_available() else "cpu")

    return time_fn(dist.all_reduce, x, is_mx=False, op=dist.ReduceOp.SUM)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark MXNet and PyTorch all_sum performance."
    )
    parser.add_argument(
        "--backend",
        choices=["mlx", "torch"],
        default="mlx",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=(1000,),
        help="Shape of the tensor to use in the benchmark (default: 5000,)",
    )
    args = parser.parse_args()
    SHAPE = tuple(args.shape)

    if args.backend == "mlx":
        time_all_sum = mlx_all_sum()
    elif args.backend == "torch":
        time_all_sum = torch_all_sum()
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    print(f"{args.backend} all_sum average time:   {time_all_sum:.3f} ms/iter")
