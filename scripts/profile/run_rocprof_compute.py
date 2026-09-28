#!/usr/bin/env python3
"""Preflight rocprof-compute and print or run the fixed RCKangaroo profile command."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from common import DEFAULT_PUBLIC_KEY, DEFAULT_START, ROOT, solver_command


HOT_KERNELS = ("KernelGen", "KernelA", "KernelB", "KernelC")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("mi300x", "mi355x", "rocm-fat"), default="mi355x")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--duration", type=int, default=90)
    parser.add_argument("--name", default="rckangaroo")
    parser.add_argument("--output-root", type=Path, default=ROOT / "profiles" / "rocprof-compute")
    parser.add_argument("--kernel", nargs="+", choices=HOT_KERNELS, default=list(HOT_KERNELS))
    parser.add_argument("--range", dest="range_bits", type=int, default=139)
    parser.add_argument("--dp", dest="dp_bits", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--pubkey", default=DEFAULT_PUBLIC_KEY)
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    executable = shutil.which("rocprof-compute")
    if not executable:
        raise SystemExit("rocprof-compute is not installed or not in PATH")
    preflight = subprocess.run(
        [executable, "--help"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, errors="replace", check=False,
    )
    if preflight.returncode:
        print("rocprof-compute cannot start. Install the Python dependencies reported below "
              "for the active ROCm installation:")
        print(preflight.stdout, end="" if preflight.stdout.endswith("\n") else "\n")
        return 2
    application = solver_command(
        args.preset, args.device, args.duration, range_bits=args.range_bits,
        dp_bits=args.dp_bits, seed=args.seed, start=args.start, public_key=args.pubkey,
    )
    command = [
        executable, "profile", "-n", args.name, "--path", str(args.output_root.resolve()),
        "--device", str(args.device), "--kernel", *args.kernel, "--", *application,
    ]
    print("+", shlex.join(command), flush=True)
    if args.print_command:
        return 0
    environment = os.environ.copy()
    environment["RCK_PROFILE"] = "1"
    return subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
