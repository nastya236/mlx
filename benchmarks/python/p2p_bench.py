# Copyright © 2026 Apple Inc.

"""
Point to point benchmark between two ring neighbors.

    mlx.launch --backend jaccl-ring --hostfile hosts.json p2p_bench.py

Rank 0 sends to rank 1 over every wire in the hostfile. No reduction and no ring
hops, so this isolates the staging and DMA path from the collective. Rank 1
times the receive and prints.
"""

import time

import mlx.core as mx

BYTES = [1_000_000_000, 2_000_000_000, 4_000_000_000, 8_000_000_000]
DTYPE = mx.float32
WARMUP = 5
ITERS = 10


def sync():
    mx.eval(mx.distributed.all_sum(mx.array(0, mx.int32)))


def bench(nbytes, rank):
    shape = (nbytes // (DTYPE.size * 1_000_000), 1_000_000)
    x = mx.zeros(shape, dtype=DTYPE) if rank == 0 else None
    if x is not None:
        mx.eval(x)

    def step():
        if rank == 0:
            mx.eval(mx.distributed.send(x, 1))
        elif rank == 1:
            mx.eval(mx.distributed.recv(shape, DTYPE, 0))

    for _ in range(WARMUP):
        step()
    sync()

    ts = []
    for _ in range(ITERS):
        tic = time.perf_counter()
        step()
        ts.append(time.perf_counter() - tic)

    del x
    mx.clear_cache()
    sync()
    return sum(ts) / ITERS, min(ts)


if __name__ == "__main__":
    world = mx.distributed.init()
    rank = world.rank()
    if world.size() < 2:
        raise SystemExit("needs at least 2 ranks")

    if rank == 1:
        print(f"# rank 0 -> rank 1, warmup {WARMUP}, iters {ITERS}")
        head = ("GB", "ms", "best ms", "GB/s", "best GB/s")
        print("{:>7} {:>9} {:>9} {:>8} {:>10}".format(*head))

    for nbytes in BYTES:
        t, best = bench(nbytes, rank)
        if rank == 1:
            print(
                f"{nbytes / 1e9:7.1f} {t * 1e3:9.2f} {best * 1e3:9.2f} "
                f"{nbytes / t / 1e9:8.2f} {nbytes / best / 1e9:10.2f}"
            )
