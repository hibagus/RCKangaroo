#!/usr/bin/env python3
"""Capture reproducible pure-kernel and end-to-end RCKangaroo baselines."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROFILE_SCRIPTS = Path(__file__).resolve().parents[1] / "profile"
sys.path.insert(0, str(PROFILE_SCRIPTS))

from common import (  # noqa: E402
    DEFAULT_PUBLIC_KEY,
    DEFAULT_START,
    ROOT,
    build_targets,
    collect_gpu_metrics,
    collect_metadata,
    create_output_directory,
    median_and_mad,
    prefixed_json,
    run,
    solver_command,
    write_json,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("mi300x", "mi355x", "rocm-fat"), default="mi355x")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, help="exact output directory")
    parser.add_argument("--output-root", type=Path, default=ROOT / "profiles")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--warmup-ms", type=int, default=2000)
    parser.add_argument("--sample-ms", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--solver-warmup-seconds", type=int, default=60)
    parser.add_argument("--solver-sample-seconds", type=int, default=90)
    parser.add_argument("--solver-samples", type=int, default=3)
    parser.add_argument("--range", dest="range_bits", type=int, default=139)
    parser.add_argument("--dp", dest="dp_bits", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--pubkey", default=DEFAULT_PUBLIC_KEY)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if min(args.warmup_ms, args.sample_ms, args.samples) <= 0:
        raise SystemExit("benchmark timing values must be positive")
    if args.solver_samples < 0 or args.solver_warmup_seconds < 0 or args.solver_sample_seconds <= 0:
        raise SystemExit("solver timing values are invalid")

    output = args.output.resolve() if args.output else create_output_directory(
        args.output_root.resolve(), "baseline", args.preset)
    output.mkdir(parents=True, exist_ok=True)
    argument_record = vars(args).copy()
    argument_record["output"] = str(output)
    argument_record["output_root"] = str(args.output_root)
    metadata = collect_metadata(args.device, args.preset, argument_record)
    metadata["gpu_metrics_before"] = collect_gpu_metrics(args.device)
    write_json(output / "metadata.json", metadata)

    if not args.skip_build:
        build_targets(args.preset, ["rckangaroo", "rckangaroo_hip_benchmark"])

    benchmark = ROOT / "build" / args.preset / "benchmarks" / "rckangaroo_hip_benchmark"
    benchmark_command = [
        str(benchmark), "--device", str(args.device), "--warmup-ms", str(args.warmup_ms),
        "--sample-ms", str(args.sample_ms), "--samples", str(args.samples),
    ]
    benchmark_result = run(
        benchmark_command, log=output / "kernel_a.log", telemetry_device=args.device,
        telemetry_log=output / "kernel_a_telemetry.json", telemetry_interval=2.0,
    )
    kernel_record = prefixed_json(benchmark_result.stdout, "RCK_BENCHMARK_JSON=")
    kernel_record["command"] = benchmark_command
    write_json(output / "kernel_a.json", kernel_record)

    environment = os.environ.copy()
    environment["RCK_PROFILE"] = "1"
    if args.solver_samples and args.solver_warmup_seconds:
        warmup_command = solver_command(
            args.preset, args.device, args.solver_warmup_seconds,
            range_bits=args.range_bits, dp_bits=args.dp_bits, seed=args.seed,
            start=args.start, public_key=args.pubkey,
        )
        run(
            warmup_command, env=environment, log=output / "solver_warmup.log",
            telemetry_device=args.device,
            telemetry_log=output / "solver_warmup_telemetry.json",
        )

    solver_records = []
    for sample in range(args.solver_samples):
        command = solver_command(
            args.preset, args.device, args.solver_sample_seconds,
            range_bits=args.range_bits, dp_bits=args.dp_bits, seed=args.seed,
            start=args.start, public_key=args.pubkey,
        )
        result = run(
            command, env=environment, log=output / f"solver_sample_{sample + 1}.log",
            telemetry_device=args.device,
            telemetry_log=output / f"solver_sample_{sample + 1}_telemetry.json",
        )
        record = prefixed_json(result.stdout, "RCK_PROFILE_JSON=")
        if int(record["sample_count"]) == 0:
            raise RuntimeError(
                "solver sample did not complete a KernelA/B/C iteration; increase "
                "--solver-sample-seconds and inspect its log (portable MI355X startup takes about 37 seconds)"
            )
        record["command"] = command
        record["sample"] = sample + 1
        solver_records.append(record)

    write_json(output / "solver_samples.json", solver_records)
    summary: dict[str, object] = {
        "schema": 1,
        "metadata": "metadata.json",
        "pure_kernel_a": kernel_record,
        "solver_sample_count": len(solver_records),
    }
    if solver_records:
        summary["kernel_gen_ms"] = median_and_mad([
            float(record["kernel_gen_ms"]) for record in solver_records
        ])
        for kernel in ("kernel_a", "kernel_b", "kernel_c"):
            summary[f"{kernel}_median_ms"] = median_and_mad([
                float(record[kernel]["median_ms"]) for record in solver_records
            ])
        summary["end_to_end_mkeys_per_second"] = median_and_mad([
            float(record["end_to_end"]["median_mkeys_per_second"])
            for record in solver_records
        ])
    metadata["gpu_metrics_after"] = collect_gpu_metrics(args.device)
    write_json(output / "metadata.json", metadata)
    write_json(output / "summary.json", summary)
    print(f"Baseline artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
