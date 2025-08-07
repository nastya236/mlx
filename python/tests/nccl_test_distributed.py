# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx
import mlx_distributed_tests
import mlx_tests


class TestNCCLDistributed(mlx_tests.MLXTestCase):
    @classmethod
    def setUpClass(cls):
        world = mx.distributed.init(strict=True, backend="nccl")
        rank = world.rank()
        mx.set_default_device(mx.Device(mx.gpu, rank % 8))
        print(f"Rank {rank}")

    def test_all_reduce(self):
        world = mx.distributed.init()
        dtypes = [
            (mx.int8, 0),
            (mx.uint8, 0),
            (mx.int16, 0),
            (mx.uint16, 0),
            (mx.int32, 0),
            (mx.uint32, 0),
            (mx.float32, 1e-6),
            (mx.float16, 5e-3),
            (mx.bfloat16, 1e-1),
            (mx.complex64, 1e-6),
        ]
        sizes = [
            (7,),
            (10,),
            (1024,),
            (1024, 1024),
        ]
        key = mx.random.key(0)

        for dt, rtol in dtypes:
            for sh in sizes:
                x = (
                    mx.random.uniform(shape=(world.size(),) + sh, key=key) * 10
                ).astype(dt)

                # All sum
                print(f"Testing all_sum with dtype {dt} and shape {sh}")
                y = mx.distributed.all_sum(x[world.rank()])
                z = x.sum(0)
                maxrelerror = (y - z).abs()
                if rtol > 0:
                    maxrelerror /= z.abs()
                maxrelerror = maxrelerror.max()
                self.assertLessEqual(maxrelerror, rtol)

                # All max
                # y = mx.distributed.all_max(x[world.rank()])
                # z = x.max(0)
                # self.assertTrue(mx.all(y == z))

                # All min
                # y = mx.distributed.all_min(x[world.rank()])
                # z = x.min(0)
                # self.assertTrue(mx.all(y == z))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
