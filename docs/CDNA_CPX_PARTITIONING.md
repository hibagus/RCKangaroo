# CPX compute partitioning on MI300X

Measured with the node switched to `CPX` compute partitioning and `NPS4` memory
partitioning. This was the last untested item from the port plan's Phase 3; setting it
requires root (`rocm-smi --setcomputepartition CPX`).

## Result: worth about 5.5%

| Partition mode | Logical devices | CUs each | Throughput (puzzle #140) |
|---|---|---|---|
| SPX / NPS1 | 8 | 304 | 81,000 MKeys/s |
| **CPX / NPS4** | **64** | **38** | **85,452 MKeys/s** |

**+5.5%**, and it scales linearly across the whole node:

| Partitions | MKeys/s | per partition | vs 16 partitions |
|---|---|---|---|
| 16 | 21,177 | 1,323.6 | 100.0% |
| 32 | 42,085 | 1,315.2 | 99.4% |
| 48 | 63,837 | 1,329.9 | 100.5% |
| 64 | **85,452** | 1,335.2 | **100.9%** |

There is no contention penalty from running 64 logical devices - per-partition throughput at
64 is marginally *better* than at 16. Note this needs a long enough sampling window: the
solver averages speed over a 16-entry ring initialised to zero, so readings taken before
about nine reports are low. A first measurement of this configuration gave 83,404 purely
because it was sampled too early.

Correctness holds. Under a deliberately high DP rate (dp 16, range 80, 55 million
distinguished points collected across 64 devices) the solver solves with zero errors, so
the host copes with 64 worker threads contending on the shared DP database rather than 8.

Each logical device reports 24 GB and allocates 2,802 MB at the default
`PNT_GROUP_CNT=32` / 3 blocks per CU, for 933,888 kangaroos per XCD.

## Why it helps, and why only slightly

In CPX each logical device is one XCD with its own 4 MB L2, and all of its traffic stays
XCD-local instead of being distributed round-robin across eight XCDs sharing an address
space. Two consequences:

- No cross-XCD interconnect hops for the kangaroo state.
- Device-scope memory operations no longer bypass L2. Per the CDNA3 ISA load/store control
  tables, device-scope accesses are "Coherent Cache Bypass" when more than one L2 cache is
  present, which is always the case in SPX. In CPX there is one L2 per device, so they can
  hit it.

The gain is modest because neither of those is the bottleneck. The working set per device is
90 MB against a 4 MB L2, so it still misses to HBM; the state array remains
non-resident, which the residency experiments elsewhere showed does not matter anyway; and
the device-scope traffic is almost entirely the DP-table atomics, which fire once per
distinguished point - roughly one jump in 2^30 at dp=30.

## Cost and practicality

- Requires root to set, and it is a node-wide mode change.
- Device count goes from 8 to 64, so `MAX_GPU_CNT` had to rise from 32 to 64 on the AMD
  path. Anything that indexes GPUs by number (the `-gpu` mask) changes meaning.
- 64 host threads instead of 8. No measured problem - scaling is linear and a 55-million-DP
  run completed cleanly - but it is more CPU pressure on the DP database.
- **The `-gpu` option cannot address devices above 9.** It parses a string of digits, so
  "035" means devices 0, 3 and 5. That is adequate for 8 GPUs and unusable for 64. Selecting
  a subset currently requires `HIP_VISIBLE_DEVICES` instead.

Given +3-5% for a privileged node-wide reconfiguration, this is worth taking if the node is
dedicated to this workload and not otherwise.

## Why only 63 partitions enumerate: DRM card-minor exhaustion

Seven of the eight physical GPUs expose all 8 XCDs. The GPU on PCI bus `0xDD` exposes only
7 - function 7 is absent - giving 63 rather than 64.

**This is not a GPU fault.** It is the known Linux issue where the BMC virtual video
controller consumes a DRM card minor, leaving one too few for a full CPX node. On this
machine the BMC graphics is a Matrox G200eW3 driven by `mgag200` rather than the ASPEED AST
that AMD's documentation cites, but the mechanism is identical.

The evidence:

```
card nodes:    64      (card0 .. card63 - the DRM legacy minor range is 0-63, saturated)
card0:         mgag200 (Matrox G200eW3 BMC video controller, PCI 03:00.0)
render nodes:  63      (renderD128 .. renderD190)
ROCm agents:   63
```

DRM allocates card minors 0-63, i.e. exactly 64 slots. `mgag200` claims `card0` at boot, so
amdgpu's 64 CPX partitions can only be given `card1`-`card63`. The 64th has nowhere to go.

Two details confirm this rather than a hardware problem:

- The missing partition is bus `0xDD` function 7 - the **last** device in PCI enumeration
  order, which is what running out of minors predicts. A defective XCD would not
  preferentially be the last one enumerated.
- That same GPU reported all 304 CUs, all 8 XCDs, in SPX mode.

### Fix

Stop the BMC video driver from claiming a DRM minor, e.g. blacklist it:

```sh
echo 'blacklist mgag200' | sudo tee /etc/modprobe.d/blacklist-mgag200.conf
sudo update-initramfs -u    # or dracut -f, depending on distro
sudo reboot
```

or add `modprobe.blacklist=mgag200` to the kernel command line.

This costs the local VGA console. Out-of-band management is unaffected - IPMI and remote KVM
are implemented in BMC firmware and do not depend on the host driver - so on a headless
node this is usually acceptable.

Recovering the partition is worth about 1.6% of the node's compute, which on the CPX figures
above would take 83,468 to roughly 84,800 MKeys/s.
