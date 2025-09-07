#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description="Launch an NCCL-backed Python script across N ranks."
    )
    p.add_argument(
        "--nproc-per-node",
        "-n",
        type=int,
        required=True,
        help="Number of processes (ranks) to launch on this node.",
    )
    p.add_argument(
        "--master-host",
        type=str,
        default="127.0.0.1",
        help="The NCCL bootstrap host (default: 127.0.0.1).",
    )
    p.add_argument(
        "--master-port",
        type=int,
        default=12345,
        help="The NCCL bootstrap port (default: 12345).",
    )
    p.add_argument(
        "script", metavar="SCRIPT", help="The Python script to run (e.g. train.py)."
    )
    return p.parse_args()


def main():
    args = parse_args()
    world_size = args.nproc_per_node

    base_env = os.environ.copy()
    base_env.update(
        {
            "NCCL_DEBUG": "INFO",
            "NCCL_SOCKET_IFNAME": "lo",
            "NCCL_HOST_IP": args.master_host,
            "NCCL_PORT": str(args.master_port),
            "MLX_WORLD_SIZE": str(world_size),
        }
    )

    procs = []
    for rank in range(world_size):
        env = base_env.copy()
        env["MLX_RANK"] = str(rank)

        cmd = [sys.executable, args.script]
        p = subprocess.Popen(cmd, env=env)
        procs.append(p)

    # wait for all ranks
    exit_codes = [p.wait() for p in procs]
    if any(code != 0 for code in exit_codes):
        print("One or more ranks failed:", exit_codes, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
