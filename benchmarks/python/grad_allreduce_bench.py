# Copyright © 2026 Apple Inc.

"""
Gradient all-reduce benchmark.

Multi node with a JACCL ring:
    mlx.launch --backend jaccl-ring --hostfile hosts.json grad_allreduce_bench.py

The ring model defaults to 3 cables per neighbor, each with 10 GB/s per direction.
These options describe the hardware; the hostfile sets the actual connections.

Single node:
    mlx.launch -n 2 grad_allreduce_bench.py
"""

import argparse
import math
import os
import time

import mlx.core as mx

PARAMS = [1_000_000_000, 2_000_000_000, 4_000_000_000]
DTYPES = [mx.float32, mx.bfloat16]
WARMUP = 5
ITERS = 10


def sync():
    mx.eval(mx.distributed.all_sum(mx.array(0, mx.int32)))


def bench(x, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        mx.eval(mx.distributed.all_sum(x))
    sync()

    ts = []
    for _ in range(iters):
        tic = time.perf_counter()
        mx.eval(mx.distributed.all_sum(x))
        ts.append(time.perf_counter() - tic)

    slowest = mx.distributed.all_max(mx.array([sum(ts) / iters, min(ts)]))
    return slowest.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=int, nargs="+", default=PARAMS)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"))
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Log JACCL ring phase timings; requires MLX rebuilt with profiling.",
    )
    parser.add_argument(
        "--cables-per-neighbor",
        type=int,
        default=3,
        help="Parallel cables to each ring neighbor (default: 3).",
    )
    parser.add_argument(
        "--link-gb-s",
        type=float,
        default=10.0,
        help="Bandwidth of one cable in one direction, in GB/s (default: 10).",
    )
    args = parser.parse_args()
    if min(args.params) < 1 or args.warmup < 0 or args.iters < 1:
        parser.error("params and iters must be positive; warmup must be nonnegative")
    if args.cables_per_neighbor < 1:
        parser.error("--cables-per-neighbor must be positive")
    if not math.isfinite(args.link_gb_s) or args.link_gb_s <= 0:
        parser.error("--link-gb-s must be finite and positive")
    if args.profile:
        os.environ["JACCL_PROFILE"] = "1"

    world = mx.distributed.init()
    nranks = world.size()
    bus_factor = 2 * (nranks - 1) / nranks
    cables = min(2, nranks - 1) * args.cables_per_neighbor

    if world.rank() == 0:
        print(f"# ranks {nranks}, warmup {args.warmup}, iters {args.iters}")
        if args.profile:
            print("# Profiling enabled; benchmark times include profile output")
        if nranks == 1:
            print("# single rank: all_sum is a no-op, bandwidth is reported as nan")
        else:
            ideal_bus = cables * args.link_gb_s
            print(
                f"# Ring model: {args.cables_per_neighbor} cables per neighbor, "
                f"{cables} per node, {args.link_gb_s:g} GB/s per cable per direction"
            )
            print(
                f"# Ideal: alg {ideal_bus / bus_factor:.2f} GB/s, "
                f"bus {ideal_bus:.2f} GB/s"
            )
        print("# alg = S/t; bus = alg * 2*(ranks-1)/ranks")
        print("# cable = bus / cables per node (send only); link % = cable / limit")
        head = (
            "params",
            "dtype",
            "GB",
            "ms",
            "best ms",
            "alg GB/s",
            "bus GB/s",
            "cable GB/s",
            "link %",
        )
        print("{:>7} {:>9} {:>7} {:>9} {:>9} {:>9} {:>9} {:>10} {:>8}".format(*head))

    dtypes = [getattr(mx, args.dtype)] if args.dtype else DTYPES
    for p in args.params:
        for dtype in dtypes:
            x = mx.zeros((p,), dtype=dtype)
            mx.eval(x)
            t, best = bench(x, args.warmup, args.iters)
            nbytes = x.nbytes
            del x
            mx.clear_cache()

            if world.rank() == 0:
                alg_bw = nbytes / t / 1e9 if nranks > 1 else float("nan")
                bus_bw = alg_bw * bus_factor
                cable_bw = bus_bw / cables if cables else float("nan")
                print(
                    f"{f'{p / 1e9:g}B':>7} {str(dtype)[9:]:>9} "
                    f"{nbytes / 1e9:7.1f} {t * 1e3:9.2f} {best * 1e3:9.2f} "
                    f"{alg_bw:9.2f} {bus_bw:9.2f} {cable_bw:10.2f} "
                    f"{100 * cable_bw / args.link_gb_s:8.2f}"
                )
