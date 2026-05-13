#!/usr/bin/env python3
"""Collect multiple speculative diagnosis CSV files into a gamma overview."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


FIELDS = [
    "gamma",
    "speculative_vs_true_speedup",
    "speculative_vs_true_speedup_median",
    "speculative_vs_zero_draft_speedup",
    "speculative_vs_zero_draft_speedup_median",
    "framework_overhead_ms",
    "framework_overhead_ms_median",
    "target_verify_reduction_ms",
    "target_verify_reduction_ms_median",
    "draft_overhead_ms",
    "net_time_saved_ms",
    "accepted_tokens_per_drafted_token",
    "accepted_tokens_per_sample",
]

METRICS = FIELDS[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--gammas", required=True, help="Comma-separated gamma values, e.g. 1,2,4,8,16")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def read_diagnosis(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"metric", "value"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path}: missing required column(s): {', '.join(sorted(missing))}")
        for row in reader:
            metric = row.get("metric", "")
            value = row.get("value", "")
            if metric:
                values[metric] = value
    return values


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_markdown(rows: list[dict[str, str]]) -> None:
    print("| " + " | ".join(FIELDS) + " |")
    print("| " + " | ".join(["---"] * len(FIELDS)) + " |")
    for row in rows:
        print("| " + " | ".join(row.get(field, "") for field in FIELDS) + " |")


def main() -> int:
    args = parse_args()
    gammas = [gamma.strip() for gamma in args.gammas.split(",")]
    if any(not gamma for gamma in gammas):
        raise SystemExit("--gammas contains an empty value")
    if len(gammas) != len(args.inputs):
        raise SystemExit(f"--gammas count ({len(gammas)}) does not match --inputs count ({len(args.inputs)})")

    rows: list[dict[str, str]] = []
    for gamma, path in zip(gammas, args.inputs):
        values = read_diagnosis(path)
        row = {"gamma": gamma}
        for metric in METRICS:
            if metric not in values:
                print(f"warning: {path}: missing metric {metric}", file=sys.stderr)
            row[metric] = values.get(metric, "")
        rows.append(row)

    write_csv(args.out, rows)
    print_markdown(rows)
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
