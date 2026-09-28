#!/usr/bin/env python3
"""Benchmark solver launch lengths with a fixed point-group geometry."""

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


def step_list(text: str) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        try:
            value = int(item)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"invalid step list: {text}") from error
        if value <= 0 or value > 65535:
            raise argparse.ArgumentTypeError("steps must be between 1 and 65535")
        if value not in values:
            values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("step list must not be empty")
    return values


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("mi300x", "mi355x", "rocm-fat"), default="mi355x")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--steps", type=step_list, default=[256, 512, 1000, 2048])
    parser.add_argument("--point-groups", type=int, default=32)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--duration", type=int, default=15)
    parser.add_argument("--warmup-duration", type=int, default=10)
    parser.add_argument("--range", dest="range_bits", type=int, default=139)
    parser.add_argument("--dp", dest="dp_bits", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--pubkey", default=DEFAULT_PUBLIC_KEY)
    parser.add_argument("--output", type=Path, help="exact output directory")
    parser.add_argument("--output-root", type=Path, default=ROOT / "profiles")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def profiled_command(args: argparse.Namespace, steps: int, duration: int) -> list[str]:
    command = solver_command(
        args.preset, args.device, duration, range_bits=args.range_bits,
        dp_bits=args.dp_bits, seed=args.seed, start=args.start,
        public_key=args.pubkey,
    )
    command.extend(("--point-groups", str(args.point_groups), "--kernel-steps", str(steps)))
    return command


def write_markdown(path: Path, summaries: list[dict[str, object]]) -> None:
    ordered = sorted(
        summaries,
        key=lambda summary: float(summary["end_to_end_mkeys_per_second"]["median"]),
        reverse=True,
    )
    lines = [
        "# Solver launch-length matrix",
        "",
        "| Rank | Steps | End-to-end MKeys/s | MAD | KernelA ms | KernelB ms | KernelC ms |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, summary in enumerate(ordered, 1):
        lines.append(
            f"| {rank} | {summary['steps']} | "
            f"{float(summary['end_to_end_mkeys_per_second']['median']):.3f} | "
            f"{float(summary['end_to_end_mkeys_per_second']['mad']):.3f} | "
            f"{float(summary['kernel_a_ms']['median']):.3f} | "
            f"{float(summary['kernel_b_ms']['median']):.3f} | "
            f"{float(summary['kernel_c_ms']['median']):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_arguments()
    if args.device < 0 or args.samples <= 0 or args.duration <= 0 or args.warmup_duration < 0:
        raise SystemExit("device, sample, and duration arguments are invalid")
    if args.point_groups < 2 or args.point_groups > 32 or args.point_groups & 1:
        raise SystemExit("point groups must be an even value between 2 and 32")

    output = args.output.resolve() if args.output else create_output_directory(
        args.output_root.resolve(), "solver-steps", args.preset
    )
    output.mkdir(parents=True, exist_ok=True)
    argument_record = vars(args).copy()
    for key in ("output", "output_root"):
        argument_record[key] = str(argument_record[key])
    metadata = collect_metadata(args.device, args.preset, argument_record)
    metadata["gpu_metrics_before"] = collect_gpu_metrics(args.device)
    write_json(output / "metadata.json", metadata)

    if not args.skip_build:
        build_targets(args.preset, ["rckangaroo"])

    environment = os.environ.copy()
    environment["RCK_PROFILE"] = "1"
    if args.warmup_duration:
        run(
            profiled_command(args, args.steps[0], args.warmup_duration), env=environment,
            log=output / "warmup.log",
        )

    records: list[dict[str, object]] = []
    for sample in range(1, args.samples + 1):
        for steps in args.steps:
            command = profiled_command(args, steps, args.duration)
            result = run(
                command, env=environment,
                log=output / f"steps_{steps}_sample_{sample}.log",
            )
            record = prefixed_json(result.stdout, "RCK_PROFILE_JSON=")
            if int(record["sample_count"]) == 0:
                raise RuntimeError(f"{steps}-step sample completed no solver iterations")
            record["steps_candidate"] = steps
            record["sample"] = sample
            record["command"] = command
            records.append(record)
            write_json(output / "samples.json", records)

    summaries: list[dict[str, object]] = []
    for steps in args.steps:
        selected = [record for record in records if int(record["steps_candidate"]) == steps]
        summary: dict[str, object] = {
            "steps": steps,
            "process_samples": len(selected),
            "solver_iterations": sum(int(record["sample_count"]) for record in selected),
            "end_to_end_mkeys_per_second": median_and_mad([
                float(record["end_to_end"]["median_mkeys_per_second"])
                for record in selected
            ]),
        }
        for kernel in ("kernel_a", "kernel_b", "kernel_c"):
            summary[f"{kernel}_ms"] = median_and_mad([
                float(record[kernel]["median_ms"]) for record in selected
            ])
        summaries.append(summary)

    winner = max(
        summaries,
        key=lambda summary: float(summary["end_to_end_mkeys_per_second"]["median"]),
    )
    metadata["gpu_metrics_after"] = collect_gpu_metrics(args.device)
    write_json(output / "metadata.json", metadata)
    write_json(output / "summary.json", {"schema": 1, "winner": winner, "results": summaries})
    write_markdown(output / "summary.md", summaries)
    print(
        f"Solver launch-length winner: {winner['steps']} steps, "
        f"{float(winner['end_to_end_mkeys_per_second']['median']):.3f} MKeys/s"
    )
    print(f"Solver launch-length artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
