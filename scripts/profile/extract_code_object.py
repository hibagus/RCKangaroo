#!/usr/bin/env python3
"""Archive AMD code objects, disassembly, instruction mix, and kernel resources."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter
from pathlib import Path

from common import ROOT, build_targets, create_output_directory, run, write_json


HOT_KERNELS = ("KernelGen", "KernelA", "KernelB", "KernelC")
DYNAMIC_LDS = {"KernelGen": 0, "KernelA": 32768, "KernelB": 49152, "KernelC": 49152}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("mi300x", "mi355x", "rocm-fat"), default="mi355x")
    parser.add_argument("--output", type=Path, help="exact output directory")
    parser.add_argument("--output-root", type=Path, default=ROOT / "profiles")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def parse_metadata(notes: str, code_object: str) -> list[dict[str, object]]:
    kernels: list[dict[str, object]] = []
    keys = {
        ".sgpr_count": "sgpr_count",
        ".vgpr_count": "vgpr_count",
        ".agpr_count": "agpr_count",
        ".sgpr_spill_count": "sgpr_spill_count",
        ".vgpr_spill_count": "vgpr_spill_count",
        ".private_segment_fixed_size": "private_segment_bytes",
        ".group_segment_fixed_size": "static_lds_bytes",
        ".kernarg_segment_size": "kernarg_bytes",
        ".max_flat_workgroup_size": "max_workgroup_size",
        ".wavefront_size": "wavefront_size",
    }
    blocks = re.split(r"(?m)^  - (?=\.agpr_count:)", notes)
    for block in blocks:
        name_match = re.search(r"(?m)^\s*\.name:\s*'?([^'\s]+)'?", block)
        if not name_match:
            continue
        name = name_match.group(1)
        if name not in HOT_KERNELS:
            continue
        current: dict[str, object] = {"kernel": name, "code_object": code_object}
        for metadata_key, output_key in keys.items():
            match = re.search(rf"(?m)^\s*{re.escape(metadata_key)}:\s*(\d+)", block)
            if match:
                current[output_key] = int(match.group(1))
        kernels.append(current)
    for kernel in kernels:
        kernel["dynamic_lds_bytes"] = DYNAMIC_LDS[str(kernel["kernel"])]
        kernel["total_lds_bytes"] = int(kernel.get("static_lds_bytes", 0)) + int(kernel["dynamic_lds_bytes"])
    return kernels


def instruction_category(mnemonic: str) -> str:
    if "atomic" in mnemonic:
        return "atomic"
    if mnemonic.startswith("ds_"):
        return "lds"
    if mnemonic.startswith(("global_", "flat_", "buffer_", "scratch_")):
        return "vmem"
    if mnemonic.startswith("v_"):
        return "valu"
    if mnemonic.startswith("s_"):
        return "salu"
    return "other"


def parse_instruction_mix(disassembly: str, code_object: str) -> list[dict[str, object]]:
    mixes: dict[str, Counter[str]] = {kernel: Counter() for kernel in HOT_KERNELS}
    current: str | None = None
    for line in disassembly.splitlines():
        label = re.search(r"<([^>]+)>:$", line)
        if label:
            current = label.group(1) if label.group(1) in HOT_KERNELS else None
            continue
        if current is None:
            continue
        match = re.search(r"\t([a-zA-Z][a-zA-Z0-9_.]+)(?:\s|$)", line)
        if not match:
            continue
        mnemonic = match.group(1)
        mixes[current][instruction_category(mnemonic)] += 1
        mixes[current]["total"] += 1
    records = []
    for kernel, counts in mixes.items():
        if counts["total"]:
            records.append({"kernel": kernel, "code_object": code_object, **counts})
    return records


def main() -> int:
    args = parse_arguments()
    output = args.output.resolve() if args.output else create_output_directory(
        args.output_root.resolve(), "isa", args.preset)
    output.mkdir(parents=True, exist_ok=True)
    if not args.skip_build:
        build_targets(args.preset, ["rckangaroo_hip_backend"])
    llvm_bin = Path("/opt/rocm/lib/llvm/bin")
    objdump = llvm_bin / "llvm-objdump"
    readelf = llvm_bin / "llvm-readelf"
    source = ROOT / "build" / args.preset / "CMakeFiles" / "rckangaroo_hip_backend.dir" / "src" / "hip" / "kernels.hip.o"
    if not source.exists():
        raise SystemExit(f"HIP object not found: {source}")
    if not objdump.exists() or not readelf.exists():
        raise SystemExit("ROCm llvm-objdump and llvm-readelf are required")

    archive = output / "code_object"
    archive.mkdir(parents=True, exist_ok=True)
    host_object = archive / "kernels.hip.o"
    shutil.copy2(source, host_object)
    extraction = run([objdump, "--offloading", host_object.name], cwd=archive,
                     log=output / "offload_extraction.log")
    code_objects = sorted(path for path in archive.iterdir() if ".hipv" in path.name)
    if not code_objects:
        raise RuntimeError(f"no AMD code object extracted; llvm-objdump said: {extraction.stdout}")

    resources: list[dict[str, object]] = []
    instruction_mix: list[dict[str, object]] = []
    for code_object in code_objects:
        notes = run([readelf, "--notes", code_object], echo=False).stdout
        notes_path = output / f"{code_object.name}.metadata.txt"
        notes_path.write_text(notes, encoding="utf-8")
        resources.extend(parse_metadata(notes, code_object.name))
        disassembly = run([objdump, "--disassemble", "--demangle", code_object], echo=False).stdout
        assembly_path = output / f"{code_object.name}.s"
        assembly_path.write_text(disassembly, encoding="utf-8")
        instruction_mix.extend(parse_instruction_mix(disassembly, code_object.name))

    write_json(output / "resources.json", {"schema": 1, "kernels": resources})
    write_json(output / "instruction_mix.json", {"schema": 1, "kernels": instruction_mix})
    if resources:
        columns = sorted({key for row in resources for key in row})
        with (output / "resources.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(resources)
    print(f"ISA and resource artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
