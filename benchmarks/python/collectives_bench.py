# Copyright © 2026 Apple Inc.

import time

import mlx.core as mx

CHUNK_BYTES = [1_000_000_000, 2_000_000_000, 4_000_000_000]
DTYPES = [mx.float32, mx.bfloat16]
WARMUP = 5
ITERS = 10


def sync():
    mx.eval(mx.distributed.all_sum(mx.array(0, mx.int32)))


def bench(fn, x):
    for _ in range(WARMUP):
        mx.eval(fn(x))
    sync()

    ts = []
    for _ in range(ITERS):
        tic = time.perf_counter()
        mx.eval(fn(x))
        ts.append(time.perf_counter() - tic)

    return mx.distributed.all_max(mx.array([sum(ts) / ITERS, min(ts)])).tolist()


if __name__ == "__main__":
    world = mx.distributed.init()
    n = world.size()

    ops = [
        ("all_sum", mx.distributed.all_sum, n, 2 * (n - 1)),
        ("all_gather", mx.distributed.all_gather, 1, n - 1),
        ("sum_scatter", mx.distributed.sum_scatter, n, n - 1),
    ]

    if world.rank() == 0:
        print(f"# ranks {n}, warmup {WARMUP}, iters {ITERS}")
        print("# GB/s = wire bytes / time; wire bytes = factor * chunk")
        head = ("op", "dtype", "chunk GB", "ms", "best ms", "GB/s")
        print("{:>12} {:>9} {:>9} {:>9} {:>9} {:>8}".format(*head))

    for chunk in CHUNK_BYTES:
        for name, fn, in_mult, factor in ops:
            for dtype in DTYPES:
                rows = in_mult * chunk // (dtype.size * 1_000_000)
                x = mx.zeros((rows, 1_000_000), dtype=dtype)
                mx.eval(x)
                t, best = bench(fn, x)
                del x
                mx.clear_cache()

                if world.rank() == 0:
                    print(
                        f"{name:>12} {str(dtype)[9:]:>9} {chunk / 1e9:9.2f} "
                        f"{t * 1e3:9.2f} {best * 1e3:9.2f} "
                        f"{factor * chunk / t / 1e9:8.2f}"
                    )
