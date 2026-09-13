# Copyright © 2026 Apple Inc.

"""Summarize phase timings from grad_allreduce_bench.py --profile."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def summarize(path, skip):
    groups = defaultdict(lambda: defaultdict(dict))
    for line in path.read_text().splitlines():
        _, marker, record = line.partition("[jaccl-profile] ")
        if not marker:
            continue
        fields = dict(part.split("=", 1) for part in record.split())
        key = tuple(
            int(fields[name])
            for name in ("ranks", "wires", "dirs", "bytes", "element_bytes", "inplace")
        )
        call = int(fields["rank"]), int(fields["call"])
        phase = int(fields["wire"]), fields["phase"]
        groups[key][call][phase] = fields

    if not groups:
        raise ValueError(f"{path}: no JACCL profile records found")

    for key, calls in sorted(groups.items()):
        nranks, wires, directions, nbytes, element_bytes, inplace = key
        print(
            f"# {path}: ranks={nranks} wires={wires} dirs={directions} "
            f"bytes={nbytes} element_bytes={element_bytes} inplace={inplace}"
        )
        print("# Phase columns are medians; cpu% uses summed CPU / summed wall time")
        print(
            f"{'rank':>4} {'wire':>4} {'phase':>14} {'calls':>5} "
            f"{'start ms':>10} {'wall ms':>10} {'cpu ms':>10} {'cpu%':>7}"
        )
        phase_keys = [
            (wire, phase)
            for wire in range(wires)
            for phase in ("reduce_scatter", "all_gather")
        ]
        expected = set(phase_keys)
        ranks = sorted({rank for rank, _ in calls})
        for rank in ranks:
            selected = [
                phases for (r, _), phases in sorted(calls.items()) if r == rank
            ][skip:]
            if any(set(phases) != expected for phases in selected):
                raise ValueError(f"{path}: incomplete profile call for rank {rank}")
            if not selected:
                print(f"# rank {rank}: no calls remain after skipping {skip}")
                continue
            active = [
                sum(float(p["cpu_us"]) for p in phases.values())
                / float(next(iter(phases.values()))["total_us"])
                for phases in selected
            ]
            print(f"# rank {rank}: mean active CPU threads = {mean(active):.2f}")
            for wire, phase in phase_keys:
                records = [phases[wire, phase] for phases in selected]
                starts = [float(p["start_us"]) for p in records]
                walls = [float(p["wall_us"]) for p in records]
                cpus = [float(p["cpu_us"]) for p in records]
                print(
                    f"{rank:4d} {wire:4d} {phase:>14} {len(records):5d} "
                    f"{median(starts) / 1e3:10.3f} {median(walls) / 1e3:10.3f} "
                    f"{median(cpus) / 1e3:10.3f} {100 * sum(cpus) / sum(walls):7.1f}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument(
        "--skip",
        type=int,
        default=5,
        help="Warmup calls to skip per rank and message configuration (default: 5).",
    )
    args = parser.parse_args()
    if args.skip < 0:
        parser.error("--skip must be nonnegative")
    for path in args.logs:
        try:
            summarize(path, args.skip)
        except (OSError, ValueError, KeyError) as exc:
            parser.exit(1, f"{exc}\n")
