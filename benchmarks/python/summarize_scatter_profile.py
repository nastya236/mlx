# Copyright © 2026 Apple Inc.

"""Summarize JACCL_PROFILE_SCATTER logs by message size, rank, and wire."""

import argparse
import json
from collections import defaultdict
from statistics import mean


def summarize(path, skip):
    groups = defaultdict(list)
    with open(path) as stream:
        for line in stream:
            marker = "JACCL_SCATTER_PROFILE "
            if marker not in line:
                continue
            record, _ = json.JSONDecoder().raw_decode(line.split(marker, 1)[1])
            key = tuple(record[k] for k in (
                "output_bytes", "element_bytes", "ranks", "wires", "rank", "wire"
            ))
            groups[key].append(record)
    if not groups:
        raise ValueError(f"No scatter profiles found in {path}")
    print(f"\n{path} (skip first {skip} calls per size/rank/wire)")
    print("out_GB elem ranks wires rank wire calls dispatch worker cpu% "
          "copy reduce poll post other delay block0 block1 empty%")
    for key, records in sorted(groups.items()):
        records.sort(key=lambda r: r["call"])
        rows = records[skip:]
        if not rows:
            continue
        def avg(field):
            return mean(r[field] for r in rows)
        worker = avg("worker_ms")
        polls = sum(r["polls"] for r in rows)
        cpu = 100 * avg("cpu_ms") / worker if worker else 0
        empty = 100 * sum(r["empty_polls"] for r in rows) / polls if polls else 0
        output, element, ranks, wires, rank, wire = key
        values = [avg("dispatch_ms"), worker, cpu, avg("copy_ms"),
                  avg("reduce_ms"), avg("poll_ms"),
                  avg("post_send_ms") + avg("post_recv_ms"), avg("other_ms"),
                  avg("start_delay_ms"), mean(r["blocked_ms"][0] for r in rows),
                  mean(r["blocked_ms"][1] for r in rows), empty]
        print(f"{output / 1e9:.3f} {element} {ranks} {wires} {rank} {wire} "
              f"{len(rows)} " + " ".join(f"{v:.3f}" for v in values))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--skip", type=int, default=5,
                        help="Warmup calls per size/rank/wire (default: 5)")
    args = parser.parse_args()
    if args.skip < 0:
        parser.error("--skip must be nonnegative")
    for path in args.logs:
        summarize(path, args.skip)
