# Standalone JACCL ring reduce-scatter profiling

Rebuild MLX with these changes on every node, using your normal release build.
Use the same Python environment and build on all nodes. This instruments the
standalone `sum_scatter` ring path, not the reduce-scatter phase of `all_sum`.

Run your normal two-cable configuration with profiling off for the bandwidth
reference. Then add `--env JACCL_PROFILE_SCATTER=1` to your launcher command:

```sh
mlx.launch --backend jaccl-ring --hostfile hosts-2.json --env JACCL_PROFILE_SCATTER=1 -- python benchmarks/python/sum_scatter_bench.py > scatter-2.log 2>&1
```

Repeat with your three-cable hostfile, writing `scatter-3.log`. The hostfile names
above are placeholders for your existing configurations. Profiling is off unless
the variable is exactly `1`. There are no timer calls in the disabled worker
specialization. Each rank prints one JSON record per worker per collective after
all its workers finish. There is no logging inside the transfer loop.

```sh
python benchmarks/python/summarize_scatter_profile.py scatter-2.log scatter-3.log --skip 5
```

Use one benchmark run per log. The summary skips five calls per message size,
rank and wire, matching `sum_scatter_bench.py`. Match `--skip` to your warmup count.
`out_GB` is output bytes per rank; full input bytes are `ranks * output_bytes`.
`elem` is element width in bytes, not a unique dtype identifier. All time columns
are mean milliseconds per call; `cpu%` and `empty%` are percentages.

| Field | Meaning |
| --- | --- |
| dispatch | Time from dispatch through completion of all local workers; excludes printing. Repeated on each wire row. |
| delay | Time from dispatch to this worker starting; includes dispatch overhead and scheduling. |
| worker | Elapsed time inside this worker. Workers run concurrently: do not add their times to get latency. |
| cpu% | Thread CPU time / worker elapsed time. Busy polling counts as CPU usage. |
| copy | Time copying tensor slices into send buffers, including initial prefill. |
| reduce | Time adding received slices and local input into the output. |
| poll | Time inside completion-queue polling, including empty polls. It is not a direct measurement of network transfer time. |
| post | Time inside posting send and receive requests; transfers continue asynchronously. |
| other | Worker time outside the five operation timers, including bookkeeping and instrumentation overhead. |
| block0 / block1 | Observed intervals with received slices waiting for prior output to be staged for sending, per direction. These overlap operation times and each other; do not add them to worker time. |
| empty% | Fraction of polls that return no completions. |

The raw JSON additionally contains separate send/receive posting times, buffer
size, completion counts, and the number of blocked intervals per direction.

Compare equal message sizes across two and three wires, looking at every rank:

- Each worker gets less data with three wires. Copy/reduce times that fail to
  decrease suggest a shared local processing limit. A local copy/reduce benchmark
  or hardware profiler is needed to separate memory throughput from execution.
- Large polling time means completions are not available quickly enough. This
  can reflect the network, a slow peer, or delayed posting; it does not prove a
  cable limit. Compare peer ranks' copy/reduce times too.
- Low CPU percentage or large start delays suggest scheduling delays. High CPU
  percentage alone cannot distinguish useful work from polling.
- Large blocked intervals indicate output reuse is delaying receive processing.
  They are overlapping observations, not independent lost time.

Detailed clocks, especially around fast empty polls, perturb execution. Compare
profiled latency against the unprofiled reference before drawing conclusions.
The benchmark's outer timing also includes profile printing. Use `dispatch` for
the local instrumented collective timing, and unprofiled runs for bandwidth.
These measurements narrow the cause; they do not automatically identify it.
