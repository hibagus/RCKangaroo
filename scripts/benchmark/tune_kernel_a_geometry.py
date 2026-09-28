#!/usr/bin/env python3
"""Build and benchmark the KernelA workgroup/point-group matrix."""

from __future__ import annotations

import argparse
import csv
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


def integer_list(text: str, *, allowed: set[int]) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        try:
            value = int(item)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"invalid integer list: {text}") from error
        if value not in allowed:
            raise argparse.ArgumentTypeError(
                f"{value} is not one of {','.join(str(entry) for entry in sorted(allowed))}"
            )
        if value not in values:
            values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(ARCHITECTURES), default="mi355x")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--block-sizes",
        type=lambda value: integer_list(value, allowed={64, 128, 256}),
        default=[64, 128, 256],
        help="comma-separated workgroup sizes",
    )
    parser.add_argument(
        "--group-counts",
        type=lambda value: integer_list(value, allowed={8, 12, 16, 24, 32}),
        default=[8, 12, 16, 24, 32],
        help="comma-separated point counts per lane",
    )
    parser.add_argument("--warmup-ms", type=int, default=500)
    parser.add_argument("--sample-ms", type=int, default=500)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, help="exact output directory")
    parser.add_argument("--output-root", type=Path, default=ROOT / "profiles")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    columns = (
        "threads", "groups", "blocks", "configured_blocks_per_cu", "kangaroos",
        "median_kernel_ms", "median_mkeys_per_second", "mad_mkeys_per_second",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_markdown(path: Path, records: list[dict[str, object]]) -> None:
    ordered = sorted(records, key=lambda record: float(record["median_mkeys_per_second"]), reverse=True)
    lines = [
        "# KernelA geometry matrix",
        "",
        "| Rank | Threads | Groups/lane | Blocks/CU | Blocks | MKeys/s | MAD | ms/launch |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, record in enumerate(ordered, 1):
        lines.append(
            f"| {rank} | {record['threads']} | {record['groups']} | "
            f"{record['configured_blocks_per_cu']} | {record['blocks']} | "
            f"{float(record['median_mkeys_per_second']):.3f} | "
            f"{float(record['mad_mkeys_per_second']):.3f} | "
            f"{float(record['median_kernel_ms']):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_arguments()
    if args.device < 0 or min(args.warmup_ms, args.sample_ms, args.samples) <= 0:
        raise SystemExit("device must be non-negative and timing values must be positive")

    output = args.output.resolve() if args.output else create_output_directory(
        args.output_root.resolve(), "kernel-a-geometry", args.preset
    )
    output.mkdir(parents=True, exist_ok=True)
    argument_record = vars(args).copy()
    for key in ("output", "output_root"):
        argument_record[key] = str(argument_record[key])
    metadata = collect_metadata(args.device, args.preset, argument_record)
    metadata["gpu_metrics_before"] = collect_gpu_metrics(args.device)
    write_json(output / "metadata.json", metadata)

    architecture = ARCHITECTURES[args.preset]
    records: list[dict[str, object]] = []
    for block_size in args.block_sizes:
        blocks_per_cu = 256 // block_size
        build = ROOT / "build" / "tuning" / f"{args.preset}-b{block_size}"
        if not args.skip_build:
            configure = [
                "cmake", "-S", ROOT, "-B", build, "-G", "Ninja",
                "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=ON",
                "-DRCK_BUILD_HIP=ON", f"-DRCK_HIP_ARCHITECTURES={architecture}",
                f"-DRCK_BLOCK_SIZE={block_size}",
                f"-DRCK_BLOCKS_PER_CU={blocks_per_cu}",
                "-DRCK_POINT_GROUP_COUNT=24", "-DRCK_STEP_COUNT=1000",
                "-DRCK_ENABLE_SLOW_GPU_TESTS=OFF",
            ]
            run(configure, log=output / f"configure_b{block_size}.log")
            run([
                "cmake", "--build", build, "--target", "rckangaroo_hip_benchmark",
                "rckangaroo_gpu_kernel_tests", "-j",
            ], log=output / f"build_b{block_size}.log")

        if not args.skip_tests:
            run(
                [build / "tests" / "rckangaroo_gpu_kernel_tests"],
                log=output / f"correctness_b{block_size}.log",
            )

        benchmark = build / "benchmarks" / "rckangaroo_hip_benchmark"
        for group_count in args.group_counts:
            command = [
                benchmark, "--device", str(args.device), "--groups", str(group_count),
                "--warmup-ms", str(args.warmup_ms), "--sample-ms", str(args.sample_ms),
                "--samples", str(args.samples),
            ]
            result = run(command, log=output / f"b{block_size}_g{group_count}.log")
            record = prefixed_json(result.stdout, "RCK_BENCHMARK_JSON=")
            if int(record["threads"]) != block_size or int(record["groups"]) != group_count:
                raise RuntimeError("benchmark reported geometry different from the requested matrix point")
            records.append(record)
            write_json(output / "results.json", {"schema": 1, "results": records})
            write_csv(output / "results.csv", records)
            write_markdown(output / "summary.md", records)

    winner = max(records, key=lambda record: float(record["median_mkeys_per_second"]))
    metadata["gpu_metrics_after"] = collect_gpu_metrics(args.device)
    write_json(output / "metadata.json", metadata)
    write_json(output / "results.json", {"schema": 1, "winner": winner, "results": records})
    print(
        "KernelA geometry winner: "
        f"{winner['threads']} threads, {winner['groups']} groups, "
        f"{winner['configured_blocks_per_cu']} blocks/CU, "
        f"{float(winner['median_mkeys_per_second']):.3f} MKeys/s"
    )
    print(f"KernelA geometry artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
