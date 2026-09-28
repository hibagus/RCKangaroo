#!/usr/bin/env python3
"""Archive code objects, ISA, and resources for Phase 4 field kernels."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter
from pathlib import Path

from common import ROOT, build_targets, create_output_directory, run, write_json


FIELD_KERNELS = (
    "FieldMulPortable",
    "FieldMulCombaMad",
    "FieldMulCombaExplicit",
    "FieldSquarePortable",
    "FieldSquareCombaMad",
    "FieldSquareCombaExplicit",
    "FieldInverseBinaryExponent",
    "FieldInverseChainPortable",
    "FieldInverseChainCombaMad",
    "FieldInverseChainCombaExplicit",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("mi300x", "mi355x", "rocm-fat"), default="mi355x")
    parser.add_argument("--output", type=Path, help="exact output directory")
    parser.add_argument("--output-root", type=Path, default=ROOT / "profiles")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def parse_metadata(notes: str, code_object: str) -> list[dict[str, object]]:
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
    records: list[dict[str, object]] = []
    for block in re.split(r"(?m)^  - (?=\.agpr_count:)", notes):
        name_match = re.search(r"(?m)^\s*\.name:\s*'?([^'\s]+)'?", block)
        if not name_match or name_match.group(1) not in FIELD_KERNELS:
            continue
        record: dict[str, object] = {
            "kernel": name_match.group(1),
            "code_object": code_object,
        }
        for metadata_key, output_key in keys.items():
            match = re.search(rf"(?m)^\s*{re.escape(metadata_key)}:\s*(\d+)", block)
            if match:
                record[output_key] = int(match.group(1))
        records.append(record)
    return records


def parse_instruction_mix(disassembly: str, code_object: str) -> list[dict[str, object]]:
    mixes = {kernel: Counter() for kernel in FIELD_KERNELS}
    current: str | None = None
    for line in disassembly.splitlines():
        label = re.search(r"<([^>]+)>:$", line)
        if label:
            current = label.group(1) if label.group(1) in FIELD_KERNELS else None
            continue
        if current is None:
            continue
        match = re.search(r"\t([a-zA-Z][a-zA-Z0-9_.]+)(?:\s|$)", line)
        if not match:
            continue
        mnemonic = match.group(1)
        mixes[current]["total"] += 1
        if mnemonic.startswith("v_"):
            mixes[current]["valu"] += 1
        elif mnemonic.startswith("s_"):
            mixes[current]["salu"] += 1
        elif mnemonic.startswith(("global_", "flat_", "buffer_")):
            mixes[current]["vmem"] += 1
        elif mnemonic.startswith("scratch_"):
            mixes[current]["scratch"] += 1
        elif mnemonic.startswith("ds_"):
            mixes[current]["lds"] += 1
        else:
            mixes[current]["other"] += 1
        for selected in (
            "v_mad_u64_u32", "v_mul_lo_u32", "v_mul_hi_u32",
            "v_add_co_u32", "v_addc_co_u32",
        ):
            if mnemonic.startswith(selected):
                mixes[current][selected] += 1
    return [
        {"kernel": kernel, "code_object": code_object, **counts}
        for kernel, counts in mixes.items() if counts["total"]
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_arguments()
    output = args.output.resolve() if args.output else create_output_directory(
        args.output_root.resolve(), "field-isa", args.preset)
    output.mkdir(parents=True, exist_ok=True)
    if not args.skip_build:
        build_targets(args.preset, ["rckangaroo_field_benchmark"])

    llvm_bin = Path("/opt/rocm/lib/llvm/bin")
    objdump = llvm_bin / "llvm-objdump"
    readelf = llvm_bin / "llvm-readelf"
    source = (ROOT / "build" / args.preset / "benchmarks" / "CMakeFiles" /
              "rckangaroo_field_benchmark.dir" / "field_ops_benchmark.hip.o")
    if not source.exists():
        raise SystemExit(f"HIP object not found: {source}")
    if not objdump.exists() or not readelf.exists():
        raise SystemExit("ROCm llvm-objdump and llvm-readelf are required")

    archive = output / "code_object"
    archive.mkdir(parents=True, exist_ok=True)
    host_object = archive / source.name
    shutil.copy2(source, host_object)
    run([objdump, "--offloading", host_object.name], cwd=archive,
        log=output / "offload_extraction.log")
    code_objects = sorted(path for path in archive.iterdir() if ".hipv" in path.name)
    if not code_objects:
        raise RuntimeError("no AMD code object was extracted")

    resources: list[dict[str, object]] = []
    instruction_mix: list[dict[str, object]] = []
    for code_object in code_objects:
        notes = run([readelf, "--notes", code_object], echo=False).stdout
        (output / f"{code_object.name}.metadata.txt").write_text(notes, encoding="utf-8")
        resources.extend(parse_metadata(notes, code_object.name))
        disassembly = run([objdump, "--disassemble", "--demangle", code_object], echo=False).stdout
        (output / f"{code_object.name}.s").write_text(disassembly, encoding="utf-8")
        instruction_mix.extend(parse_instruction_mix(disassembly, code_object.name))

    write_json(output / "resources.json", {"schema": 1, "kernels": resources})
    write_json(output / "instruction_mix.json", {"schema": 1, "kernels": instruction_mix})
    write_csv(output / "resources.csv", resources)
    write_csv(output / "instruction_mix.csv", instruction_mix)
    print(f"Field ISA and resource artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
