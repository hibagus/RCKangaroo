#!/usr/bin/env python3
"""Gate handwritten AMD ISA work behind measured primitive improvements."""

from __future__ import annotations

import argparse
import json
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
    run,
    write_json,
)


COMPARISONS = (
    ("multiply", "multiply_comba_mad", "multiply_comba_explicit"),
    ("square", "square_comba_mad", "square_comba_explicit"),
    ("inverse", "inverse_chain_comba_mad", "inverse_chain_comba_explicit"),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("mi300x", "mi355x"), default="mi355x")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--multiply-iterations", type=int, default=256)
    parser.add_argument("--inverse-iterations", type=int, default=1)
    parser.add_argument(
        "--minimum-primitive-speedup-percent",
        type=float,
        default=3.0,
        help="speedup an explicit-ISA primitive must exceed before integration",
    )
    parser.add_argument("--output", type=Path, help="exact output directory")
    parser.add_argument("--output-root", type=Path, default=ROOT / "profiles")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def kernel_records(path: Path) -> dict[str, dict[str, object]]:
    document = read_json(path)
    records = document.get("kernels")
    if not isinstance(records, list):
        raise RuntimeError(f"invalid kernel artifact: {path}")
    indexed: dict[str, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("kernel"), str):
            raise RuntimeError(f"invalid kernel record: {path}")
        name = str(record["kernel"])
        if name in indexed:
            raise RuntimeError(f"duplicate kernel {name} in single-architecture artifact: {path}")
        indexed[name] = record
    return indexed


def comparison_records(
    benchmark: dict[str, object], minimum_speedup_percent: float
) -> list[dict[str, object]]:
    raw_results = benchmark.get("results")
    if not isinstance(raw_results, list):
        raise RuntimeError("field benchmark has no result list")
    results = {
        str(record["name"]): record
        for record in raw_results
        if isinstance(record, dict) and "name" in record
    }
    comparisons: list[dict[str, object]] = []
    required_ratio = 1.0 + minimum_speedup_percent / 100.0
    for operation, selected_name, candidate_name in COMPARISONS:
        if selected_name not in results or candidate_name not in results:
            raise RuntimeError(f"field benchmark is missing {selected_name} or {candidate_name}")
        selected_gops = float(results[selected_name]["median_gops"])
        candidate_gops = float(results[candidate_name]["median_gops"])
        ratio = candidate_gops / selected_gops
        comparisons.append({
            "operation": operation,
            "selected_kernel": selected_name,
            "candidate_kernel": candidate_name,
            "selected_median_gops": selected_gops,
            "candidate_median_gops": candidate_gops,
            "candidate_over_selected": ratio,
            "candidate_speedup_percent": (ratio - 1.0) * 100.0,
            "required_ratio": required_ratio,
            "passed": ratio >= required_ratio,
        })
    return comparisons


def write_markdown(
    path: Path,
    *,
    gate: dict[str, object],
    benchmark: dict[str, object],
    field_resources: dict[str, dict[str, object]],
    field_mix: dict[str, dict[str, object]],
) -> None:
    passed = bool(gate["primitive_gate_passed"])
    decision = "integrate and test the candidate" if passed else "retain compiler-generated HIP"
    lines = [
        "# Phase 8 AMD ISA gate",
        "",
        f"- Decision: **{decision}**",
        f"- GPU: {benchmark.get('gpu', 'unknown')} ({benchmark.get('arch', 'unknown')})",
        f"- Samples: {gate['samples']}",
        "- Primitive threshold: an explicit-ISA candidate must exceed the selected "
        f"compiler path by {float(gate['minimum_primitive_speedup_percent']):.2f}%",
        "",
        "| Operation | Compiler Gop/s | Explicit ISA Gop/s | Candidate / compiler | Change | Pass |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    comparisons = gate["comparisons"]
    assert isinstance(comparisons, list)
    for record in comparisons:
        assert isinstance(record, dict)
        lines.append(
            f"| {str(record['operation']).title()} | "
            f"{float(record['selected_median_gops']):.6f} | "
            f"{float(record['candidate_median_gops']):.6f} | "
            f"{float(record['candidate_over_selected']):.3f}x | "
            f"{float(record['candidate_speedup_percent']):+.2f}% | "
            f"{'yes' if record['passed'] else 'no'} |"
        )

    lines.extend([
        "",
        "## Static resources",
        "",
        "| Kernel | VGPR | SGPR | Private bytes | VGPR spills | SGPR spills |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    selected_kernels: list[str] = []
    for _, selected_name, candidate_name in COMPARISONS:
        for name in (selected_name, candidate_name):
            kernel_name = {
                "multiply_comba_mad": "FieldMulCombaMad",
                "multiply_comba_explicit": "FieldMulCombaExplicit",
                "square_comba_mad": "FieldSquareCombaMad",
                "square_comba_explicit": "FieldSquareCombaExplicit",
                "inverse_chain_comba_mad": "FieldInverseChainCombaMad",
                "inverse_chain_comba_explicit": "FieldInverseChainCombaExplicit",
            }[name]
            selected_kernels.append(kernel_name)
            record = field_resources[kernel_name]
            lines.append(
                f"| `{kernel_name}` | {record.get('vgpr_count', 0)} | "
                f"{record.get('sgpr_count', 0)} | {record.get('private_segment_bytes', 0)} | "
                f"{record.get('vgpr_spill_count', 0)} | {record.get('sgpr_spill_count', 0)} |"
            )

    lines.extend([
        "",
        "## Static instruction mix",
        "",
        "| Kernel | Total | VALU | SALU | MAD64 | MUL lo | MUL hi | ADD co | ADDC co |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for kernel_name in selected_kernels:
        record = field_mix[kernel_name]
        lines.append(
            f"| `{kernel_name}` | {record.get('total', 0)} | {record.get('valu', 0)} | "
            f"{record.get('salu', 0)} | {record.get('v_mad_u64_u32', 0)} | "
            f"{record.get('v_mul_lo_u32', 0)} | {record.get('v_mul_hi_u32', 0)} | "
            f"{record.get('v_add_co_u32', 0)} | {record.get('v_addc_co_u32', 0)} |"
        )

    lines.extend(["", "## Interpretation", ""])
    if passed:
        lines.append(
            "The primitive gate passed. This does not accept the assembly candidate: the next "
            "step is a source-built architecture-specific integration with full correctness and "
            "end-to-end performance validation."
        )
    else:
        lines.append(
            "The candidate failed the primitive gate, so a full-kernel assembly integration is "
            "not justified by this result. Keep the directly compiled HIP path and rerun this "
            "gate after a material candidate or toolchain change."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_arguments()
    if args.device < 0:
        raise SystemExit("device must be non-negative")
    if min(args.samples, args.multiply_iterations, args.inverse_iterations) <= 0:
        raise SystemExit("samples and iteration counts must be positive")
    if args.blocks is not None and args.blocks <= 0:
        raise SystemExit("blocks must be positive")
    if args.minimum_primitive_speedup_percent < 0:
        raise SystemExit("minimum primitive speedup must be non-negative")

    output = args.output.resolve() if args.output else create_output_directory(
        args.output_root.resolve(), "phase8-isa-gate", args.preset
    )
    output.mkdir(parents=True, exist_ok=True)
    argument_record = vars(args).copy()
    for key in ("output", "output_root"):
        argument_record[key] = str(argument_record[key])
    metadata = collect_metadata(args.device, args.preset, argument_record)
    metadata["gpu_metrics_before"] = collect_gpu_metrics(args.device)
    write_json(output / "metadata.json", metadata)

    if not args.skip_build:
        targets = ["rckangaroo_field_benchmark", "rckangaroo_hip_backend"]
        if not args.skip_tests:
            targets.append("rckangaroo_gpu_field_variant_tests")
        build_targets(args.preset, targets)
    if not args.skip_tests:
        run(
            [ROOT / "build" / args.preset / "tests" / "rckangaroo_gpu_field_variant_tests"],
            log=output / "correctness.log",
        )

    field_output = output / "field"
    field_command: list[str | Path] = [
        sys.executable,
        ROOT / "scripts" / "benchmark" / "run_field_variants.py",
        "--preset", args.preset,
        "--device", str(args.device),
        "--samples", str(args.samples),
        "--multiply-iterations", str(args.multiply_iterations),
        "--inverse-iterations", str(args.inverse_iterations),
        "--skip-build",
        "--output", field_output,
    ]
    if args.blocks is not None:
        field_command.extend(("--blocks", str(args.blocks)))
    run(field_command, log=output / "field_runner.log")

    hot_output = output / "hot-kernel-isa"
    run([
        sys.executable,
        ROOT / "scripts" / "profile" / "extract_code_object.py",
        "--preset", args.preset,
        "--skip-build",
        "--output", hot_output,
    ], log=output / "hot_kernel_isa.log")

    benchmark = read_json(field_output / "benchmark.json")
    comparisons = comparison_records(benchmark, args.minimum_primitive_speedup_percent)
    field_resources = kernel_records(field_output / "isa" / "resources.json")
    field_mix = kernel_records(field_output / "isa" / "instruction_mix.json")
    primitive_gate_passed = any(bool(record["passed"]) for record in comparisons)
    gate: dict[str, object] = {
        "schema": 1,
        "preset": args.preset,
        "device": args.device,
        "samples": args.samples,
        "minimum_primitive_speedup_percent": args.minimum_primitive_speedup_percent,
        "primitive_gate_passed": primitive_gate_passed,
        "end_to_end_gate_required": primitive_gate_passed,
        "decision": (
            "integrate_candidate_for_end_to_end_validation"
            if primitive_gate_passed
            else "retain_compiler_generated_hip"
        ),
        "comparisons": comparisons,
        "field_resources": list(field_resources.values()),
        "field_instruction_mix": list(field_mix.values()),
        "hot_kernel_resources": read_json(hot_output / "resources.json")["kernels"],
        "hot_kernel_instruction_mix": read_json(hot_output / "instruction_mix.json")["kernels"],
        "artifacts": {
            "field": "field",
            "hot_kernel_isa": "hot-kernel-isa",
            "correctness_log": None if args.skip_tests else "correctness.log",
        },
    }
    write_json(output / "gate.json", gate)
    write_markdown(
        output / "summary.md",
        gate=gate,
        benchmark=benchmark,
        field_resources=field_resources,
        field_mix=field_mix,
    )
    metadata["gpu_metrics_after"] = collect_gpu_metrics(args.device)
    write_json(output / "metadata.json", metadata)
    print(f"Phase 8 decision: {gate['decision']}")
    print(f"Phase 8 ISA-gate artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
