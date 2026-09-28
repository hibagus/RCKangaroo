#!/usr/bin/env python3
"""Generate a Markdown numeric comparison for two Phase 3 artifact directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PREFERRED_FILES = ("summary.json", "hot_kernel_summary.json", "resources.json", "instruction_mix.json")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def flatten(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            result.update(flatten(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, dict) and "name" in child:
                label = str(child["name"])
            elif isinstance(child, dict) and "kernel" in child:
                label = str(child["kernel"])
                if "code_object" in child:
                    label += f"@{child['code_object']}"
            else:
                label = str(index)
            result.update(flatten(child, f"{prefix}.{label}"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = float(value)
    return result


def load_artifacts(directory: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for filename in PREFERRED_FILES:
        path = directory / filename
        if path.exists():
            values.update(flatten(json.loads(path.read_text(encoding="utf-8")), filename))
    if not values:
        raise SystemExit(f"no comparable Phase 3 JSON artifacts found in {directory}")
    return values


def main() -> int:
    args = parse_arguments()
    before = load_artifacts(args.before)
    after = load_artifacts(args.after)
    common = sorted(set(before) & set(after))
    lines = [
        "# RCKangaroo profile comparison",
        "",
        f"Before: `{args.before.resolve()}`  ",
        f"After: `{args.after.resolve()}`",
        "",
        "| Metric | Before | After | Delta | Change |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in common:
        old = before[metric]
        new = after[metric]
        delta = new - old
        change = delta / abs(old) * 100.0 if old else float("nan")
        change_text = f"{change:+.3f}%" if old else "n/a"
        lines.append(f"| `{metric}` | {old:.6g} | {new:.6g} | {delta:+.6g} | {change_text} |")
    report = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"Comparison report: {args.output}")
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
