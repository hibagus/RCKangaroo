#!/usr/bin/env python3
"""Measure RCKangaroo host-pipeline scaling on selected GPU counts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import time
from typing import Any


PROFILE_PREFIX = "RCK_PROFILE_JSON="


def parse_counts(text: str) -> list[int]:
    counts: list[int] = []
    for token in text.split(","):
        try:
            count = int(token)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"invalid GPU count: {token}") from error
        if count <= 0:
            raise argparse.ArgumentTypeError("GPU counts must be positive")
        if count in counts:
            raise argparse.ArgumentTypeError(f"duplicate GPU count: {count}")
        counts.append(count)
    return counts


def child_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def child_max_rss_kib() -> int:
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss


def extract_profiles(output: str) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for line in output.splitlines():
        if line.startswith(PROFILE_PREFIX):
            profiles.append(json.loads(line[len(PROFILE_PREFIX) :]))
    return profiles


def run_sample(arguments: argparse.Namespace, gpu_count: int) -> dict[str, Any]:
    gpu_list = ",".join(str(index) for index in range(gpu_count))
    command = [
        str(arguments.executable),
        "--gpu",
        gpu_list,
        "-range",
        str(arguments.range),
        "-dp",
        str(arguments.dp),
        "--seed",
        str(arguments.seed),
        "--duration",
        str(arguments.duration),
    ]
    environment = os.environ.copy()
    environment["RCK_PROFILE"] = "1"

    cpu_before = child_cpu_seconds()
    wall_start = time.monotonic()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    wall_seconds = time.monotonic() - wall_start
    cpu_seconds = child_cpu_seconds() - cpu_before
    profiles = extract_profiles(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{gpu_count}-GPU sample failed with status {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    if len(profiles) != gpu_count:
        raise RuntimeError(
            f"expected {gpu_count} profile records, found {len(profiles)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    per_gpu = [
        profile["end_to_end"]["median_mkeys_per_second"]
        for profile in profiles
    ]
    return {
        "gpu_count": gpu_count,
        "gpu_list": gpu_list,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "cpu_percent_one_core": 100.0 * cpu_seconds / wall_seconds,
        "max_rss_kib": child_max_rss_kib(),
        "per_gpu_mkeys_per_second": per_gpu,
        "per_gpu_median_mkeys_per_second": statistics.median(per_gpu),
        "aggregate_mkeys_per_second": sum(per_gpu),
        "profiles": profiles,
        "command": command,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path("build/mi355x/bin/rckangaroo"),
    )
    parser.add_argument("--counts", type=parse_counts, default=parse_counts("1,2,4,8"))
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--range", type=int, default=139)
    parser.add_argument("--dp", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    if arguments.duration <= 0:
        parser.error("--duration must be positive")
    if not arguments.executable.is_file():
        parser.error(f"executable does not exist: {arguments.executable}")

    results: list[dict[str, Any]] = []
    for gpu_count in arguments.counts:
        sample = run_sample(arguments, gpu_count)
        results.append(sample)
        print(
            f"{gpu_count} GPU: {sample['aggregate_mkeys_per_second']:.3f} MKeys/s, "
            f"CPU {sample['cpu_percent_one_core']:.1f}% of one core"
        )

    one_gpu = next(
        (sample["aggregate_mkeys_per_second"] for sample in results
         if sample["gpu_count"] == 1),
        None,
    )
    for sample in results:
        sample["scaling_efficiency_percent"] = (
            100.0 * sample["aggregate_mkeys_per_second"] /
            (one_gpu * sample["gpu_count"])
            if one_gpu else None
        )

    report = {
        "schema": 1,
        "protocol": {
            "duration_seconds": arguments.duration,
            "range": arguments.range,
            "dp": arguments.dp,
            "seed": arguments.seed,
            "profile_environment": "RCK_PROFILE=1",
        },
        "results": results,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"wrote {arguments.output}")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
