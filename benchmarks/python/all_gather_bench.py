# Copyright © 2026 Apple Inc.

import time

import mlx.core as mx

INPUT_BYTES = [500_000_000, 1_000_000_000, 2_000_000_000, 4_000_000_000]
DTYPE = mx.float32
WARMUP = 5
ITERS = 10


def sync():
    mx.eval(mx.distributed.all_sum(mx.array(0, mx.int32)))


def bench(x):
    for _ in range(WARMUP):
        mx.eval(mx.distributed.all_gather(x))
    sync()

    ts = []
    for _ in range(ITERS):
        tic = time.perf_counter()
        mx.eval(mx.distributed.all_gather(x))
        ts.append(time.perf_counter() - tic)

    slowest = mx.distributed.all_max(mx.array([sum(ts) / ITERS, min(ts)]))
    return slowest.tolist()


if __name__ == "__main__":
    world = mx.distributed.init()
    nranks = world.size()

    if world.rank() == 0:
        print(f"# ranks {nranks}, warmup {WARMUP}, iters {ITERS}")
        print("# GB/s = (ranks - 1) * input / time")
        head = ("GB", "ms", "best ms", "GB/s")
        print("{:>7} {:>9} {:>9} {:>8}".format(*head))

    for nbytes in INPUT_BYTES:
        rows = nbytes // (DTYPE.size * 1_000_000)
        x = mx.zeros((rows, 1_000_000), dtype=DTYPE)
        mx.eval(x)
        t, best = bench(x)
        del x
        mx.clear_cache()

        if world.rank() == 0:
            print(
                f"{nbytes / 1e9:7.2f} {t * 1e3:9.2f} {best * 1e3:9.2f} "
                f"{(nranks - 1) * nbytes / t / 1e9:8.2f}"
            )
