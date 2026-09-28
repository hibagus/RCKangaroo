# Phase 7: host and multi-GPU execution

## Status

Phase 7 is complete for the eight-GPU MI355X host. The host pipeline is also
architecture-independent and cross-builds for `gfx942`; physical MI300X
runtime validation remains part of the outstanding MI300X hardware pass.

The selected design keeps the normal hot path device-local. It does not use
peer memory, XGMI copies, or cross-device synchronization.

## Implementation

Each GPU owns the following resources:

- one non-blocking compute stream and one non-blocking transfer stream;
- two device DP output buffers;
- two matching `hipHostMalloc` pinned host buffers;
- completion events for compute and DMA;
- one GPU worker and one distinguished-point consumer thread.

The worker queues a compute launch into one slot, makes the transfer stream
wait on that slot's completion event, and copies the complete fixed-capacity
output asynchronously. It then queues the next compute launch before waiting
for the previous DMA. Copying the fixed-capacity buffer removes the old
device-to-host count round trip and costs only about 13 MiB per iteration.

A completed pinned slot is handed directly to that GPU's consumer. The slot is
not reusable until the consumer releases it. This bounded backpressure means a
slow consumer cannot overwrite or silently drop a host batch. Device-side
overflow is clamped, counted, and reported separately.

The old global staging buffer and its insertion lock are gone. `TFastBase`
uses 256 locks selected by the first key byte, matching its existing 256-way
memory-pool partition. Consumers therefore insert into independent database
shards concurrently. The record count is atomic, and allocation failures stop
all workers cleanly.

On Linux, each PCI device's `local_cpulist` is intersected with the process
CPU affinity. The GPU worker and consumer are pinned to distinct CPUs from
that local set. Pinned output allocation and large initialization allocations
occur after worker affinity is applied, so first-touch placement follows the
selected NUMA locality. The selected CPU IDs are printed at startup.
Unsupported or restricted topology discovery falls back to the process CPU
set; affinity failure is a warning rather than a solver failure.

GPU initialization now uses a deterministic per-device `mt19937_64` stream
derived from the run seed, solve index, and physical device index. This
removes the shared RNG lock and makes initialization independent of worker
scheduling.

The preferred selector is a comma-separated list:

```sh
build/mi355x/bin/rckangaroo --gpu 0,3,5 ...
```

The compact legacy spelling `-gpu 035` remains accepted. Use `--gpu 10` for a
multi-digit device index.

## MI355X scaling

Environment:

- Date: 2026-09-28
- ROCm: 7.2.0
- GPUs: eight AMD Instinct MI355X, `gfx950`, 256 CUs each
- Host: two NUMA nodes, 256 logical CPUs
- Solver defaults: 256 threads, 32 groups, 2,048 steps, 64 KiB KernelA LDS,
  workgroup-major state
- Workload: range 139, DP 32, seed 1, 30-second requested duration
- Profiling: `RCK_PROFILE=1`; medians are computed per GPU after KernelGen

Reproduction:

```sh
python3 scripts/benchmark/run_multi_gpu_scaling.py \
  --counts 1,2,4,8 --duration 30 --dp 32 \
  --output phase7-scaling.json
```

| GPUs | Aggregate MKeys/s | Per-GPU median MKeys/s | Efficiency | Process CPU |
|---:|---:|---:|---:|---:|
| 1 | 4,455.360 | 4,455.360 | 100.00% | 34.1% of one core |
| 2 | 8,948.004 | 4,474.002 | 100.42% | 38.6% of one core |
| 4 | 17,905.241 | 4,476.266 | 100.47% | 45.8% of one core |
| 8 | 35,698.616 | 4,469.269 | 100.16% | 59.4% of one core |

Efficiencies slightly above 100% are normal run-to-run clock and millisecond
sampling variation; the result should be read as linear scaling, not
superlinear speedup. The one-GPU result also matches the Phase 6 fully selected
pair result of 4,453.052 MKeys/s within 0.06%, so the host pipeline does not
reduce steady GPU throughput.

The first two-GPU attempt was rejected as a transient: GPU 0 ran KernelA at
1.100 seconds instead of the normal approximately 0.94 seconds and the pair
measured 8,154.769 MKeys/s with high dispersion. The table uses the repeated,
matched one-/two-GPU pair. The four- and eight-GPU rows come from the original
matrix, whose devices all had normal KernelA medians.

## Higher DP traffic

A 20-second DP16 stress pair used the same range, seed, and production kernel
defaults:

| GPUs | Aggregate MKeys/s | Efficiency | Process CPU |
|---:|---:|---:|---:|
| 1 | 4,459.987 | 100.00% | 9.9% of one core |
| 8 | 35,770.662 | 100.25% | 63.1% of one core |

DP16 therefore preserved both single-GPU throughput and eight-GPU scaling.
The process stayed below one logical CPU of aggregate host time even with one
consumer per GPU. No host-ring or device-output overflow was reported.

## Correctness and failure coverage

The CPU suite exercises:

- normal double-buffer ownership and ordered shutdown;
- producer backpressure while a consumer owns a slot;
- abort wakeups and invalid state transitions;
- DP overflow clamping and dropped-count accounting;
- Linux CPU-list parsing and topology CPU selection;
- modern and compact legacy GPU-list parsing, including duplicates, empty
  tokens, and out-of-range indices;
- concurrent unique and duplicate inserts across database shards.

The MI355X suite passes all ten tests, including both slow end-to-end known-key
vectors at DP14. Runtime failures now propagate through atomic error counts;
the solve loop detects when every GPU worker exits before a solution, stops the
remaining pipeline, joins every worker and consumer, and releases HIP and
pinned resources.

## Remaining validation

- Run the same correctness and performance protocols on physical MI300X.
- Repeat scaling under production power/clock policy when comparing systems;
  the current measurements used managed clocks.
- For unusually low DP values, watch the explicit device-overflow diagnostic.
  Host rings apply lossless backpressure, but the finite device output remains
  intentionally bounded.
