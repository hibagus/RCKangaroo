#!/usr/bin/env python3
"""Benchmark Phase 4 field variants and archive their code objects."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROFILE_SCRIPTS = Path(__file__).resolve().parents[1] / "profile"
sys.path.insert(0, str(PROFILE_SCRIPTS))

from common import (  # noqa: E402
    ROOT,
    build_targets,
    collect_gpu_metrics,
    collect_metadata,
    create_output_directory,
    prefixed_json,
    run,
    write_json,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("mi300x", "mi355x", "rocm-fat"), default="mi355x")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--multiply-iterations", type=int, default=256)
    parser.add_argument("--inverse-iterations", type=int, default=1)
    parser.add_argument("--output", type=Path, help="exact output directory")
    parser.add_argument("--output-root", type=Path, default=ROOT / "profiles")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-isa", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    values = [args.samples, args.multiply_iterations, args.inverse_iterations]
    if any(value <= 0 for value in values) or (args.blocks is not None and args.blocks <= 0):
        raise SystemExit("samples, blocks, and iteration counts must be positive")
    if args.device < 0:
        raise SystemExit("device must be non-negative")

    output = args.output.resolve() if args.output else create_output_directory(
        args.output_root.resolve(), "field-variants", args.preset)
    output.mkdir(parents=True, exist_ok=True)
    argument_record = vars(args).copy()
    argument_record["output"] = str(output)
    argument_record["output_root"] = str(args.output_root)
    metadata = collect_metadata(args.device, args.preset, argument_record)
    metadata["gpu_metrics_before"] = collect_gpu_metrics(args.device)
    write_json(output / "metadata.json", metadata)

    if not args.skip_build:
        build_targets(args.preset, ["rckangaroo_field_benchmark"])

    benchmark = ROOT / "build" / args.preset / "benchmarks" / "rckangaroo_field_benchmark"
    command = [
        str(benchmark), "--device", str(args.device), "--samples", str(args.samples),
        "--multiply-iterations", str(args.multiply_iterations),
        "--inverse-iterations", str(args.inverse_iterations),
    ]
    if args.blocks is not None:
        command.extend(("--blocks", str(args.blocks)))
    result = run(command, log=output / "benchmark.log")
    record = prefixed_json(result.stdout, "RCK_FIELD_BENCHMARK_JSON=")
    record["command"] = command
    write_json(output / "benchmark.json", record)

    if not args.skip_isa:
        extractor = PROFILE_SCRIPTS / "extract_field_variants.py"
        run([
            sys.executable, extractor, "--preset", args.preset, "--skip-build",
            "--output", output / "isa",
        ], log=output / "isa_extraction.log")

    metadata["gpu_metrics_after"] = collect_gpu_metrics(args.device)
    write_json(output / "metadata.json", metadata)
    print(f"Field-variant artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
