#!/usr/bin/env python3
"""Build and benchmark group-major and workgroup-major GPU state layouts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROFILE_SCRIPTS = Path(__file__).resolve().parents[1] / "profile"
sys.path.insert(0, str(PROFILE_SCRIPTS))

from common import (  # noqa: E402
    ROOT,
    collect_gpu_metrics,
    collect_metadata,
    create_output_directory,
    prefixed_json,
    run,
    write_json,
)


ARCHITECTURES = {"mi300x": "gfx942", "mi355x": "gfx950"}
LAYOUTS = {"group": 0, "workgroup": 1}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(ARCHITECTURES), default="mi355x")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--point-groups", type=int, default=32)
    parser.add_argument("--warmup-ms", type=int, default=1000)
    parser.add_argument("--sample-ms", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path, help="exact output directory")
    parser.add_argument("--output-root", type=Path, default=ROOT / "profiles")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def write_markdown(path: Path, records: list[dict[str, object]]) -> None:
    ordered = sorted(records, key=lambda record: float(record["median_mkeys_per_second"]), reverse=True)
    lines = [
        "# KernelA state-layout matrix",
        "",
        "| Rank | Layout | MKeys/s | MAD | ms/launch |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, record in enumerate(ordered, 1):
        lines.append(
            f"| {rank} | {record['state_layout_name']} | "
            f"{float(record['median_mkeys_per_second']):.3f} | "
            f"{float(record['mad_mkeys_per_second']):.3f} | "
            f"{float(record['median_kernel_ms']):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_arguments()
    if args.device < 0 or min(args.warmup_ms, args.sample_ms, args.samples) <= 0:
        raise SystemExit("device and timing values must be positive")
    if args.point_groups < 2 or args.point_groups > 32 or args.point_groups & 1:
        raise SystemExit("point groups must be an even value between 2 and 32")

    output = args.output.resolve() if args.output else create_output_directory(
        args.output_root.resolve(), "state-layout", args.preset
    )
    output.mkdir(parents=True, exist_ok=True)
    argument_record = vars(args).copy()
    for key in ("output", "output_root"):
        argument_record[key] = str(argument_record[key])
    metadata = collect_metadata(args.device, args.preset, argument_record)
    metadata["gpu_metrics_before"] = collect_gpu_metrics(args.device)
    write_json(output / "metadata.json", metadata)

    records: list[dict[str, object]] = []
    for layout, layout_value in LAYOUTS.items():
        build = ROOT / "build" / "tuning" / f"{args.preset}-layout-{layout}"
        if not args.skip_build:
            run([
                "cmake", "-S", ROOT, "-B", build, "-G", "Ninja",
                "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=ON",
                "-DRCK_BUILD_HIP=ON",
                f"-DRCK_HIP_ARCHITECTURES={ARCHITECTURES[args.preset]}",
                "-DRCK_BLOCK_SIZE=256", "-DRCK_BLOCKS_PER_CU=1",
                "-DRCK_POINT_GROUP_COUNT=24", "-DRCK_STEP_COUNT=1000",
                "-DRCK_USE_ARCH_TABLE_DEFAULTS=ON", f"-DRCK_STATE_LAYOUT={layout}",
                "-DRCK_USE_ARCH_STATE_LAYOUT_DEFAULTS=OFF",
                "-DRCK_ENABLE_SLOW_GPU_TESTS=OFF",
            ], log=output / f"configure_{layout}.log")
            run([
                "cmake", "--build", build, "--target", "rckangaroo_hip_benchmark",
                "rckangaroo_gpu_kernel_tests", "-j",
            ], log=output / f"build_{layout}.log")

        if not args.skip_tests:
            run(
                [build / "tests" / "rckangaroo_gpu_kernel_tests"],
                log=output / f"correctness_{layout}.log",
            )
        command = [
            build / "benchmarks" / "rckangaroo_hip_benchmark",
            "--device", str(args.device), "--groups", str(args.point_groups),
            "--warmup-ms", str(args.warmup_ms), "--sample-ms", str(args.sample_ms),
            "--samples", str(args.samples),
        ]
        result = run(command, log=output / f"benchmark_{layout}.log")
        record = prefixed_json(result.stdout, "RCK_BENCHMARK_JSON=")
        if int(record["state_layout"]) != layout_value:
            raise RuntimeError("benchmark reported a different state layout")
        record["state_layout_name"] = layout
        records.append(record)
        write_json(output / "results.json", {"schema": 1, "results": records})
        write_markdown(output / "summary.md", records)

    winner = max(records, key=lambda record: float(record["median_mkeys_per_second"]))
    metadata["gpu_metrics_after"] = collect_gpu_metrics(args.device)
    write_json(output / "metadata.json", metadata)
    write_json(output / "results.json", {"schema": 1, "winner": winner, "results": records})
    print(
        f"KernelA state-layout winner: {winner['state_layout_name']}, "
        f"{float(winner['median_mkeys_per_second']):.3f} MKeys/s"
    )
    print(f"KernelA state-layout artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
