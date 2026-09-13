# Measure JACCL ring scaling

Rebuild and install this checkout on every benchmark node with your usual MLX
build command. The profiling code is in the native JACCL library, so copying the
Python benchmark alone is not enough.

Use the same tensor size and dtype for each run. From the repository root:

```sh
mlx.launch --backend jaccl-ring --hostfile hosts-2.json \
  benchmarks/python/grad_allreduce_bench.py \
  --cables-per-neighbor 2 --params 1000000000 --dtype float32 \
  --warmup 5 --iters 10 --profile > profile-2.log 2>&1

mlx.launch --backend jaccl-ring --hostfile hosts-3.json \
  benchmarks/python/grad_allreduce_bench.py \
  --cables-per-neighbor 3 --params 1000000000 --dtype float32 \
  --warmup 5 --iters 10 --profile > profile-3.log 2>&1

python3 benchmarks/python/summarize_jaccl_profile.py \
  profile-2.log profile-3.log --skip 5
```

Each hostfile must select the stated number of connections. The benchmark's
`--cables-per-neighbor` option only sets the bandwidth model.

`--profile` sets `JACCL_PROFILE=1` on every process. It records all ring all-reduce
operations larger than 32 KiB, including warmup operations. The summary skips
the first five calls for each rank and message configuration. If you change
`--warmup`, use the same value for `--skip`.

The timers run at worker and phase boundaries. There are no timers or log writes
inside the polling loop. Each rank prints its records after its workers finish.
`total_us` excludes printing, but the Python benchmark time includes it. Repeat
without `--profile` for the final bandwidth measurement.

| Field | Meaning |
| --- | --- |
| `start_us` | Phase start relative to this rank's local all-reduce dispatch. |
| `wall_us` | Elapsed time for the phase, including time off CPU. |
| `cpu_us` | CPU time consumed by this worker during the phase. Includes busy polling. |
| `total_us` | Local dispatch-to-join time for all workers, before printing. |
| `thread` | macOS thread ID, for matching a worker in Instruments. |
| `inplace` | Whether input and output share the same pointer. |
| `mean active CPU threads` | Sum of worker CPU times divided by local total time, averaged across calls. |

Times are local to each process. Do not compare `start_us` as an absolute time
across machines. Call IDs are local too; compare matching operations and sizes.
The summary's median rows do not represent one particular call. Use the raw
records to examine worker overlap within a call.

## Interpret the results

- A worker that starts late has dispatch or scheduling delay before its phase.
- A long phase with low CPU time spent part of its duration off CPU or blocked.
- A long phase with CPU time close to wall time was active. It may have been
  copying, reducing, or polling; CPU time alone does not distinguish these.
- If only reduce-scatter scales poorly, inspect reduction work and its memory
  accesses. If both phases scale poorly, inspect the work they share, including
  copies, RDMA progress, and memory access.
- If one worker or rank finishes late, inspect that worker and its peers. Waiting
  for a peer can appear as high CPU time in a busy-poll loop.

With two to three connections, each worker handles roughly two thirds as many
bytes. Compare phase durations against that expectation, using large messages.

## Separate copying, reduction, and polling

On a benchmark node, attach Instruments **Time Profiler** to the Python process
running the benchmark. Keep system libraries visible and separate results by
thread. Match the thread IDs from the raw profile records, then inspect time in
`ring_pass`, copy routines, reduction functions, and `ibv_poll_cq` or RDMA driver
code. Use a build with debug symbols if the stacks are not resolved.

For workers with a large wall/CPU gap, record **System Trace** to inspect thread
states and scheduling. A high empty-poll count does not prove that a worker was
descheduled.

Apple's [Instruments performance walkthrough](https://developer.apple.com/videos/play/wwdc2026/268/)
explains how Time Profiler and System Trace separate active CPU work from waiting.
