#!/usr/bin/env python3
"""Shared, dependency-free helpers for RCKangaroo profiling scripts."""

from __future__ import annotations

import datetime as dt
import json
import platform
import shlex
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_START = "80000000000000000000000000000000000"
DEFAULT_PUBLIC_KEY = "031f6a332d3c5c4f2de2378c012f429cd109ba07d69690c6c701b6bb87860d6640"


def run(
    command: Iterable[str | Path],
    *,
    env: dict[str, str] | None = None,
    log: Path | None = None,
    check: bool = True,
    echo: bool = True,
    cwd: Path = ROOT,
    telemetry_device: int | None = None,
    telemetry_log: Path | None = None,
    telemetry_interval: float = 5.0,
) -> subprocess.CompletedProcess[str]:
    argv = [str(item) for item in command]
    if echo:
        print("+", shlex.join(argv), flush=True)
    if telemetry_device is None:
        result = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            errors="replace",
            check=False,
        )
    else:
        if telemetry_log is None or telemetry_interval <= 0:
            raise ValueError("telemetry_log and a positive telemetry_interval are required")
        samples: list[dict[str, Any]] = []

        def capture_metrics() -> None:
            samples.append({
                "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "metrics": collect_gpu_metrics(telemetry_device),
            })

        with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as output:
            process = subprocess.Popen(
                argv, cwd=cwd, env=env, text=True, stdout=output,
                stderr=subprocess.STDOUT, errors="replace",
            )
            capture_metrics()
            while True:
                try:
                    returncode = process.wait(timeout=telemetry_interval)
                    break
                except subprocess.TimeoutExpired:
                    capture_metrics()
            capture_metrics()
            output.seek(0)
            result = subprocess.CompletedProcess(argv, returncode, output.read())
        write_json(telemetry_log, {
            "schema": 1,
            "device": telemetry_device,
            "interval_seconds": telemetry_interval,
            "command": argv,
            "samples": samples,
        })
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(result.stdout, encoding="utf-8")
    if echo and result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if check and result.returncode:
        raise RuntimeError(f"command exited with status {result.returncode}: {shlex.join(argv)}")
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prefixed_json(output: str, prefix: str) -> dict[str, Any]:
    records = []
    for line in output.splitlines():
        if line.startswith(prefix):
            records.append(json.loads(line[len(prefix) :]))
    if not records:
        raise RuntimeError(f"output did not contain {prefix}")
    return records[-1]


def command_text(command: list[str]) -> str | None:
    executable = shutil.which(command[0])
    if not executable:
        return None
    result = run([executable, *command[1:]], check=False, echo=False)
    return result.stdout.strip() or None


def command_json(command: list[str]) -> dict[str, Any] | None:
    executable = shutil.which(command[0])
    if not executable:
        return None
    result = run([executable, *command[1:]], check=False, echo=False)
    if result.returncode:
        return {"error": result.stdout.strip(), "returncode": result.returncode}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "tool did not return JSON", "output": result.stdout.strip()}


def collect_metadata(device: int, preset: str, arguments: dict[str, Any]) -> dict[str, Any]:
    revision = run(["git", "rev-parse", "HEAD"], echo=False).stdout.strip()
    status = run(["git", "status", "--short"], echo=False).stdout.splitlines()
    rocm_version_path = Path("/opt/rocm/.info/version")
    rocm_version = rocm_version_path.read_text(encoding="utf-8").strip() \
        if rocm_version_path.exists() else None
    return {
        "schema": 1,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository": {
            "revision": revision,
            "short_revision": revision[:12],
            "dirty": bool(status),
            "status": status,
        },
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "kernel": platform.release(),
            "processor": platform.processor(),
        },
        "toolchain": {
            "preset": preset,
            "rocm_version": rocm_version,
            "hipcc_version": command_text(["hipcc", "--version"]),
            "rocprofv3_version": command_text(["rocprofv3", "--version"]),
            "rocprof_compute_version": command_text(["rocprof-compute", "--version"]),
        },
        "gpu": command_json(["amd-smi", "static", "-g", str(device), "--json"]),
        "arguments": arguments,
    }


def collect_gpu_metrics(device: int) -> dict[str, Any] | None:
    return command_json([
        "amd-smi", "metric", "-g", str(device), "--clock", "--temperature",
        "--power", "--mem-usage", "--json",
    ])


def create_output_directory(base: Path, prefix: str, preset: str) -> Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    revision = run(["git", "rev-parse", "--short=12", "HEAD"], echo=False).stdout.strip()
    output = base / f"{timestamp}-{prefix}-{preset}-{revision}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def build_targets(preset: str, targets: list[str]) -> None:
    run(["cmake", "--build", "--preset", preset, "--target", *targets, "-j"])


def solver_command(
    preset: str,
    device: int,
    duration: int,
    *,
    range_bits: int,
    dp_bits: int,
    seed: int,
    start: str,
    public_key: str,
) -> list[str]:
    if not 0 <= device <= 9:
        raise ValueError("the legacy -gpu syntax supports device indices 0 through 9")
    return [
        str(ROOT / "build" / preset / "bin" / "rckangaroo"),
        "-gpu", str(device), "-dp", str(dp_bits), "-range", str(range_bits),
        "-start", start, "-pubkey", public_key, "--seed", str(seed),
        "--duration", str(duration),
    ]


def median_and_mad(values: list[float]) -> dict[str, float]:
    if not values:
        return {"median": 0.0, "mad": 0.0, "minimum": 0.0, "maximum": 0.0}
    median = statistics.median(values)
    return {
        "median": median,
        "mad": statistics.median(abs(value - median) for value in values),
        "minimum": min(values),
        "maximum": max(values),
    }
