#!/usr/bin/env python3
"""Run rocprofv3 on a fixed solver workload and summarize every hot kernel."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
from pathlib import Path

from common import (
    DEFAULT_PUBLIC_KEY,
    DEFAULT_START,
    ROOT,
    build_targets,
    collect_gpu_metrics,
    collect_metadata,
    create_output_directory,
    prefixed_json,
    run,
    solver_command,
    write_json,
)


HOT_KERNELS = ("KernelGen", "KernelA", "KernelB", "KernelC")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("mi300x", "mi355x", "rocm-fat"), default="mi355x")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--duration", type=int, default=90)
    parser.add_argument("--output", type=Path, help="exact output directory")
    parser.add_argument("--output-root", type=Path, default=ROOT / "profiles")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--range", dest="range_bits", type=int, default=139)
    parser.add_argument("--dp", dest="dp_bits", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--pubkey", default=DEFAULT_PUBLIC_KEY)
    return parser.parse_args()


def locate_output(output: Path, filename: str) -> Path:
    matches = sorted(output.rglob(filename))
    if not matches:
        raise RuntimeError(f"rocprofv3 did not produce {filename}")
    return matches[0]


def summarize_kernels(stats_path: Path, trace_path: Path) -> list[dict[str, object]]:
    percentages: dict[str, float] = {}
    with stats_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            percentages[row.get("Name", "")] = float(row["Percentage"])

    dispatches: dict[str, list[dict[str, str]]] = {name: [] for name in HOT_KERNELS}
    with trace_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row.get("Kernel_Name", "")
            if name in dispatches:
                dispatches[name].append(row)

    records: list[dict[str, object]] = []
    for name in HOT_KERNELS:
        rows = dispatches[name]
        if not rows:
            continue
        durations = [int(row["End_Timestamp"]) - int(row["Start_Timestamp"]) for row in rows]
        median = statistics.median(durations)
        record: dict[str, object] = {
            "name": name,
            "calls": len(rows),
            "total_ns": sum(durations),
            "average_ns": statistics.mean(durations),
            "median_ns": median,
            "mad_ns": statistics.median(abs(duration - median) for duration in durations),
            "percentage": percentages.get(name, 0.0),
            "min_ns": min(durations),
            "max_ns": max(durations),
            "stddev_ns": statistics.pstdev(durations),
            "lds_block_bytes": int(rows[0]["LDS_Block_Size"]),
            "scratch_bytes": int(rows[0]["Scratch_Size"]),
            "allocated_vgprs": int(rows[0]["VGPR_Count"]),
            "allocated_agprs": int(rows[0]["Accum_VGPR_Count"]),
            "allocated_sgprs": int(rows[0]["SGPR_Count"]),
            "workgroup": [int(rows[0][f"Workgroup_Size_{axis}"]) for axis in "XYZ"],
            "grid": [int(rows[0][f"Grid_Size_{axis}"]) for axis in "XYZ"],
        }
        records.append(record)
    found = {record["name"] for record in records}
    missing = set(HOT_KERNELS) - found
    if missing:
        raise RuntimeError(f"rocprofv3 trace is missing hot kernels: {', '.join(sorted(missing))}")
    return records


def main() -> int:
    args = parse_arguments()
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")
    if not shutil.which("rocprofv3"):
        raise SystemExit("rocprofv3 is not installed or not in PATH")
    output = args.output.resolve() if args.output else create_output_directory(
        args.output_root.resolve(), "rocprof", args.preset)
    output.mkdir(parents=True, exist_ok=True)
    arguments = vars(args).copy()
    arguments["output"] = str(output)
    arguments["output_root"] = str(args.output_root)
    metadata = collect_metadata(args.device, args.preset, arguments)
    metadata["gpu_metrics_before"] = collect_gpu_metrics(args.device)
    write_json(output / "metadata.json", metadata)
    if not args.skip_build:
        build_targets(args.preset, ["rckangaroo"])

    application = solver_command(
        args.preset, args.device, args.duration, range_bits=args.range_bits,
        dp_bits=args.dp_bits, seed=args.seed, start=args.start, public_key=args.pubkey,
    )
    command = [
        "rocprofv3", "--kernel-trace", "--memory-copy-trace", "--scratch-memory-trace",
        "--stats", "--summary", "--output-directory", str(output),
        "--output-file", "trace", "--output-format", "csv", "--", *application,
    ]
    environment = os.environ.copy()
    environment["RCK_PROFILE"] = "1"
    result = run(
        command, env=environment, log=output / "rocprof.log",
        telemetry_device=args.device, telemetry_log=output / "telemetry.json",
    )
    event_profile = prefixed_json(result.stdout, "RCK_PROFILE_JSON=")
    if int(event_profile["sample_count"]) == 0:
        raise RuntimeError(
            "profile did not complete a KernelA/B/C iteration; increase --duration "
            "and inspect rocprof.log (portable MI355X startup takes about 37 seconds)"
        )
    write_json(output / "hip_event_profile.json", event_profile)
    kernel_summary = summarize_kernels(
        locate_output(output, "trace_kernel_stats.csv"),
        locate_output(output, "trace_kernel_trace.csv"),
    )
    write_json(output / "hot_kernel_summary.json", {
        "schema": 1,
        "command": command,
        "hot_kernels": kernel_summary,
        "hip_event_profile": event_profile,
    })
    metadata["gpu_metrics_after"] = collect_gpu_metrics(args.device)
    write_json(output / "metadata.json", metadata)
    print(f"rocprofv3 artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
