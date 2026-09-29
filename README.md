<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GPLv3 License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">RCKangaroo — AMD CDNA Port</h3>

  <p align="center">
    RetiredCoder's SOTA v2 Kangaroo ECDLP solver, ported to AMD Instinct MI300X / MI355X
    <br />
    <a href="docs/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="#performance">View Results</a>
    &middot;
    <a href="https://github.com/hibagus/RCKangaroo/issues">Report Bug</a>
    &middot;
    <a href="https://github.com/hibagus/RCKangaroo/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#tuning">Tuning</a></li>
        <li><a href="#testing">Testing</a></li>
      </ul>
    </li>
    <li>
      <a href="#performance">Performance</a>
      <ul>
        <li><a href="#what-each-optimisation-was-worth">What each optimisation was worth</a></li>
        <li><a href="#measured-and-rejected">Measured and rejected</a></li>
        <li><a href="#why-hand-written-assembly-was-cancelled">Why hand-written assembly was cancelled</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This is a port of [RetiredCoder's RCKangaroo v4][rc-url] to AMD Instinct GPUs. The original
CUDA build is untouched and still works; this adds a parallel HIP path for CDNA3 (MI300X,
gfx942) and CDNA4 (MI355X, gfx950).

The algorithm is unchanged — SOTA v2 kangaroo with batched ("triple Montgomery") inversion,
K = 1.15. What had to be rebuilt is everything that touched NVIDIA-specific machinery:

* **Field arithmetic.** The CUDA original expresses every carry chain through PTX's implicit
  sticky carry flag. CDNA has no equivalent — carries live in VCC or an arbitrary SGPR pair —
  so the primitives were rewritten from scratch. Each formulation was chosen by measuring
  emitted instructions, and the winners are not consistent with one another: the multiply
  wants plain `__uint128_t` (which the backend lowers to exactly 64 `v_mad_u64_u32`, the
  theoretical minimum), carry chains want `__builtin_addcll`, and the reduction wants 64-bit
  limbs where RC uses 32-bit.
* **Targeted assembly.** The multiply, `SubModP` and the reduction are inline asm, generated
  by [tools/gen_asm.py](tools/gen_asm.py). The reason is specific: the hardware computes
  `{carry, acc64} = a.u32 * b.u32 + acc64` in one instruction with the carry delivered free
  into VCC, and C++ cannot consume a multiply-accumulate's carry-out, so it recovers it with
  a comparison instead.
* **Host side.** [include/cdna/cuda_compat.h](include/cdna/cuda_compat.h) maps the ~90
  mechanical CUDA→HIP renames so the host sources stay shared, with `#ifdef` seams only where
  the platforms genuinely differ.

Five latent bugs in the shared host code were fixed along the way, each in its own commit: an
unconditional cubin load that made the non-turbo fallback unreachable on any non-sm_89/120
GPU, a `CalcKangCnt`/`Prepare` mismatch that skewed the startup estimates, a signed-int
overflow in `L2size`, an absent `hipGetLastError` check, and an `L1S2` allocation whose two
wrong assumptions cancelled out at one block per CU.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![ROCm][rocm-shield]][rocm-url]
* [![HIP][hip-shield]][hip-url]
* [![C++][cpp-shield]][cpp-url]
* [![Python][python-shield]][python-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* ROCm 10 or later. The build auto-detects `/opt/rocm/core-10`, falling back to `/opt/rocm`.
* An AMD Instinct GPU — gfx942 (MI300X) and gfx950 (MI355X) are both tested.
* `g++` for the host sources. They stay on GCC because `utils.cpp` uses dialect-alternative
  inline asm that clang's integrated assembler rejects, and `Ec.cpp` uses x86 intrinsics.
* Python 3 for the assembly generator and the differential test's ground truth.

### Installation

```sh
git clone git@github.com:hibagus/RCKangaroo.git
cd RCKangaroo
make -f Makefile.hip                          # gfx942 (default)
make -f Makefile.hip OFFLOAD_ARCH=gfx950      # CDNA4
make -f Makefile.hip OFFLOAD_ARCH="gfx942;gfx950"
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

Command-line options are RC's and unchanged; see the original README below.

```sh
# benchmark mode, 76-bit range
./build-hip/rckangaroo-cdna -dp 16 -range 76

# solve a public key (puzzle #140)
./build-hip/rckangaroo-cdna -dp 30 -range 139 \
    -start 80000000000000000000000000000000000 \
    -pubkey 031f6a332d3c5c4f2de2378c012f429cd109ba07d69690c6c701b6bb87860d6640
```

### Tuning

| Setting | Default | Notes |
| --- | --- | --- |
| `RCK_BLOCKS_PER_CU` (env) | 3 | Workgroups launched per CU. Lower it for ranges below ~85 bits, where the extra kangaroos cost more in DP overhead than they gain in throughput. |
| `PNT_GROUP_CNT` (`defs.h`) | 32 on AMD, 24 on NVIDIA | Batched-inversion group size. 32 is the ceiling — `L1S2` carries one bit per group in a `u32`. |
| CPX compute partitioning | off | Worth +5.5%. Needs root: `rocm-smi --setcomputepartition CPX`. See [docs/CDNA_CPX_PARTITIONING.md](docs/CDNA_CPX_PARTITIONING.md). |

### Testing

```sh
make -f Makefile.hip test    # 10,000,000-vector arithmetic differential test
make -f Makefile.hip isa     # static codegen gate
make -f Makefile.hip bench   # instruction-rate microbenchmark
```

* The **differential test** checks all eight field primitives against Python
  arbitrary-precision ground truth, so the reference cannot share a bug with the code under
  test. Vectors lead with the full edge-case cross product — 0, 1, p−1, p, 2²⁵⁶−1, and values
  near 2²⁵⁶ that stress the Solinas fold.
* The **codegen gate** asserts the multiply still emits exactly 64 `v_mad_u64_u32` and that
  nothing spills, catching codegen regressions a throughput benchmark would blame on the
  algorithm.
* **K convergence** is the check that matters end to end. A solver can return correct keys
  while wasting most of its work; broken SOTA loop handling inflates K rather than producing
  wrong answers, so benchmark mode converging near K = 1.15 is the real signal.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- PERFORMANCE -->
## Performance

Puzzle #140 (139-bit range, K = 1.15, 0% DP overhead). MI300X on ROCm 10, MI355X on
ROCm 7.2.

| Configuration | Throughput |
| --- | --- |
| 8× MI355X, SPX partitioning | **126,180 MKeys/s** |
| 8× MI300X, CPX partitioning | 85,452 MKeys/s |
| 8× MI300X, SPX partitioning | 81,000 MKeys/s |
| Per physical MI355X | 15.9 GH/s |
| Per physical MI300X | 10.7 GH/s |
| RTX 4090 (RC's figure) | 14.5 GH/s |
| RTX 5090 (RC's figure) | 19.3 GH/s |

MI355X is 57% faster per GPU than MI300X even though its nominal vector-integer peak is 4%
*lower* — the gain is HBM3E bandwidth and sustained clock, not ALUs, and CDNA4 adds no new
integer instructions. It is also power-limited rather than thermally limited, which makes
bytes-per-point-addition worth more than instructions-per-point-addition. See
[docs/CDNA4_MI355X.md](docs/CDNA4_MI355X.md).

The node is about 5.9× a single RTX 4090. **Per watt the consumer NVIDIA parts win this
workload outright**, by roughly 2.5×, and the reason is a single instruction-set feature
rather than ALU count or cache size — see [docs/CDNA_VS_NVIDIA.md](docs/CDNA_VS_NVIDIA.md).

### What each optimisation was worth

| Step | Effect |
| --- | --- |
| Working HIP port | 6.25 GH/s per GPU |
| Both jump tables in LDS, avoiding generic `flat_load` for the per-kangaroo table select | required for the fast path, not optional |
| 2 workgroups per CU — the grid, not LDS or registers, was capping occupancy at 1 wave/SIMD | +21% |
| 256-bit multiply in assembly | +14% |
| `PNT_GROUP_CNT` 24 → 32, 3 blocks/CU, `jmp_x`-only in LDS | +4% |
| `SubModP` in assembly | +0.5% |
| Modular reduction in assembly | +6% |
| CPX compute partitioning | +5.5% |
| **Total** | **6.25 → 10.7 GH/s per GPU** |

### Measured and rejected

Most ideas that sounded good did not survive measurement. They are listed so nobody repeats
them.

| Idea | Result |
| --- | --- |
| Fit the kangaroo state in the 256 MB Infinity Cache | **Slower.** Tested two ways. Configurations that fit have worse inverse amortisation; a 239 MB working set runs at 8,566 MKeys/s against 9,957 for a 717 MB one. |
| Workgroup-contiguous state layout, for DRAM row locality | **37% slower.** The 7.47 MB group stride is what spreads accesses across HBM channels. RC's layout is load-bearing. |
| Software-pipeline the state loads one group ahead | **2.2% slower.** No latency to hide — `MemUnitStalled` is 0.65%. |
| Dedicated squaring exploiting symmetry | Not worth it. Doubling an off-diagonal product needs two MADs, which is what the general multiply already costs. |
| Halve `JMP_CNT` to free LDS | Entangled with the flag bit layout and `JmpDists12` offsets; abandoned as correctness-critical for little gain. |
| Fuse `SubModP` into `MulModP` | **Recovers nothing.** The pair together costs *less* than the two separately — the compiler already optimises across the boundary. |

### Why hand-written assembly was cancelled

A full hand-scheduled kernel — what RC's `main.asm` is — was planned and projected at
+12–35%. Three measurements closed it.

1. **Most register moves are structurally required.** Of 347 per group body, 268 are inside
   the primitives and only 79 are kernel glue. The multiply's 240 cannot be removed:
   `v_mad_u64_u32` needs an even-aligned VGPR pair for its 64-bit addend, so a
   32-bit-granular sliding accumulator costs ~2 moves per column however arranged. The
   compiler manages 40 per multiply where a naive hand-written version needs ~60 —
   hand-writing it would be *worse*.
2. **`s_nop` is free at this occupancy**, which invalidated the cost metric the plan rested
   on. Moving the reduction's last C++ fold into assembly cut `MulModP` from 240 VALU + 30
   wait states to 238 + 17 — 5.6% by the "issue slots" figure used throughout. Back-to-back
   A/B/A measured 84,341 / 84,232 / 84,531 MKeys/s: noise 0.23%, effect −0.24%, no gain.
   Thirteen of the fifteen removed slots were `s_nop`, and CDNA issues scalar and vector
   instructions independently, so at 3 waves/SIMD another wave's VALU fills that cycle.
3. **The real prize is ~4%.** Counting VALU only, the removable budget is 79 glue moves out
   of 1,971 per point-addition — roughly 84,400 → 87,600 MKeys/s for the node, in exchange
   for hand-scheduling ~2,500 lines of assembly that **no differential test can cover**. A
   monolithic kernel would have only K values and solve counts to go on, which are
   statistical and would let a subtle distance-accounting bug hide as "slightly worse K".

For context, the arithmetic already runs at **87% of VALU peak** measured in isolation
([bench/arith_rate.hip](bench/arith_rate.hip)), so little remains in the primitives anyway.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Phase 0 — measure CDNA instruction rates before designing anything
- [x] Phase 1 — working HIP port of RCKangaroo v4
- [x] Phase 2 — assembly for the arithmetic primitives
- [x] Phase 3 — architecture and occupancy tuning, CPX partitioning
- [x] ~~Phase 4 — hand-written kernel assembly~~ *cancelled; measured at ~4% for a ~2,500-line rewrite*
- [x] Verify on CDNA4 (gfx950) hardware — [measured on 8× MI355X](docs/CDNA4_MI355X.md)
- [x] Move `jmp_y` back into LDS — free on CDNA4's 160 KB, +2.0%; still costs a wave on CDNA3
- [ ] Measure CPX partitioning on MI355X; it was worth +5.5% on MI300X
- [ ] Rework `-gpu` to address more than 10 devices, which CPX exposes
- [ ] Host-side usability: dynamic DP validation, per-GPU statistics, progress display

See the [open issues](https://github.com/hibagus/RCKangaroo/issues) for a full list.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are welcome. Two conventions matter here more than usual:

1. **Measure before building.** Nearly every promising idea in this port turned out neutral
   or negative, and the cheap experiment that revealed it saved days each time. Benchmark
   A/B comparisons must run back to back in one session, taking the maximum over 13+ samples
   with the baseline re-measured, because the speed display averages over a 16-entry ring
   initialised to zero and reads low before then.
2. **Keep the primitives testable.** Anything touching field arithmetic must still pass
   `make -f Makefile.hip test` — 10,000,000 vectors against independent ground truth. That
   harness caught a real operand-numbering bug in generated assembly that would otherwise
   have shipped.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the GPLv3 License. See `LICENSE.TXT` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Bagus Hanindhito — [@hibagus](https://github.com/hibagus)

Project Link: [https://github.com/hibagus/RCKangaroo](https://github.com/hibagus/RCKangaroo)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [RetiredCoder][rc-url] — the original RCKangaroo, the SOTA v2 method, and the hand-written
  SASS that set the bar this port was measured against
* [AMD Instinct MI300 CDNA3 ISA Reference][cdna3-url] and [CDNA4 ISA Reference][cdna4-url] —
  the hazard tables and cache-control semantics that several decisions turned on
* [Bernstein & Yang, *Fast constant-time gcd computation and modular inversion*][by-url] —
  the divsteps inversion used by `InvModP`
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/hibagus/RCKangaroo.svg?style=for-the-badge
[contributors-url]: https://github.com/hibagus/RCKangaroo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/hibagus/RCKangaroo.svg?style=for-the-badge
[forks-url]: https://github.com/hibagus/RCKangaroo/network/members
[stars-shield]: https://img.shields.io/github/stars/hibagus/RCKangaroo.svg?style=for-the-badge
[stars-url]: https://github.com/hibagus/RCKangaroo/stargazers
[issues-shield]: https://img.shields.io/github/issues/hibagus/RCKangaroo.svg?style=for-the-badge
[issues-url]: https://github.com/hibagus/RCKangaroo/issues
[license-shield]: https://img.shields.io/github/license/hibagus/RCKangaroo.svg?style=for-the-badge
[license-url]: https://github.com/hibagus/RCKangaroo/blob/master/LICENSE.TXT
[rocm-shield]: https://img.shields.io/badge/ROCm-10-ED1C24?style=for-the-badge&logo=amd&logoColor=white
[rocm-url]: https://rocm.docs.amd.com/
[hip-shield]: https://img.shields.io/badge/HIP-CDNA3%20%7C%20CDNA4-ED1C24?style=for-the-badge&logo=amd&logoColor=white
[hip-url]: https://rocm.docs.amd.com/projects/HIP/
[cpp-shield]: https://img.shields.io/badge/C++-17-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white
[cpp-url]: https://isocpp.org/
[python-shield]: https://img.shields.io/badge/Python-3-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[rc-url]: https://github.com/RetiredC
[cdna3-url]: docs/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
[cdna4-url]: docs/amd-instinct-cdna4-instruction-set-architecture.pdf
[by-url]: https://tches.iacr.org/index.php/TCHES/article/download/8298/7648/4494

---

# Original README (RetiredCoder)

(c) 2024-2026, RetiredCoder (RC)

RCKangaroo is free and open-source (GPLv3).
This software demonstrates efficient GPU implementation of SOTA v2 Kangaroo method for solving ECDLP. 
It's part #3 of my research, you can find more details here: https://github.com/RetiredC

Discussion thread: https://bitcointalk.org/index.php?topic=5517607


<b>Features:</b>

- Lowest K=1.15, it means 1.8 times less required operations compared to classic method with K=2.1, also it means that you need 1.8 times less memory to store DPs.
- Fastest: about 14.5GH/s for 4090 and 19.3GH/s for 5090 (turbo kernels).
- Keeps DP overhead as small as possible.
- Supports ranges up to 170 bits.
- Both Windows and Linux are supported.


<b>Turbo kernels for 4xxx and 5xxx cards:</b>
- Written in pure assembler (using RCAsm) to use full power of GPUs.
- "Triple Montgomery trick" (let's name it like that) is applied to reduce GPU resources used for inverse calculation to about 3% only.
- Inverse calculation is masked completely and excluded from the main loop.


<b>Limitations:</b>

- No advanced features like networking, saving/loading DPs, etc.


<b>Command line parameters:</b>

<b>-gpu</b>		which GPUs are used, for example, "035" means that GPUs #0, #3 and #5 are used. If not specified, all available GPUs are used. 

<b>-pubkey</b>		public key to solve, both compressed and uncompressed keys are supported. If not specified, software starts in benchmark mode and solves random keys. 

<b>-start</b>		start offset of the key, in hex. Mandatory if "-pubkey" option is specified. For example, for puzzle #85 start offset is "1000000000000000000000". 

<b>-range</b>		bit range of private the key. Mandatory if "-pubkey" option is specified. For example, for puzzle #85 bit range is "84" (84 bits). Must be in range 32...170. 

<b>-dp</b>		DP bits. Must be in range 14...32. Low DP bits values cause larger DB but reduces DP overhead and vice versa. 

<b>-max</b>		option to limit max number of operations. For example, value 5.5 limits number of operations to 5.5 * 1.15 * sqrt(range), software stops when the limit is reached. 

<b>-tames</b>		filename with tames. If file not found, software generates tames (option "-max" is required) and saves them to the file. If the file is found, software loads tames to speedup solving. 

When public key is solved, software displays it and also writes it to "RESULTS.TXT" file. 

Sample command line for puzzle #85:

RCKangaroo.exe -dp 16 -range 84 -start 1000000000000000000000 -pubkey 0329c4574a4fd8c810b7e42a4b398882b381bcd85e40c6883712912d167c83e73a

Sample command to generate tames:

RCKangaroo.exe -dp 16 -range 76 -tames tames76.dat -max 10

Then you can restart software with same parameters to see less K in benchmark mode or add "-tames tames76.dat" to solve some public key in 76-bit range faster.


<b>Some notes:</b>

Fastest ECDLP solvers will always use SOTA/SOTA+ method, as it's 1.4/1.5 times faster and requires less memory for DPs compared to the best 3-way kangaroos with K=1.6. 
Even if you already have a faster implementation of kangaroo jumps, incorporating SOTA method will improve it further. 
While adding the necessary loop-handling code will cause you to lose about 5–15% of your current speed, the SOTA method itself will provide a 40% performance increase. 
Overall, this translates to roughly a 25% net improvement, which should not be ignored if your goal is to build a truly fast solver. 


<b>Changelog:</b>

v4.0:

- added turbo kernels (asm) for 4xxx and 5xxx cards, about 14.5GH/s for 4090 and 19.3GH for 5090.
- this version is optimized for 4xxx and 5xxx cards, for older cards use previous versions for best performance.
- added estimation for K with DP overhead.
- some minor changes, fixed some bugs.

v3.1:

- fixed "gpu illegal memory access" bug.
- some small improvements.

v3.0:

- added "-tames" and "-max" options.
- fixed some bugs.

v2.0:

- added support for 30xx, 20xx and 1xxx cards.
- some minor changes.

v1.1:

- added ability to start software on 30xx cards.

v1.0:

- initial release.