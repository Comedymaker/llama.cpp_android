#!/usr/bin/env python3
"""Summarize llama-speculative-simple diagnostic CSV output."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev


MODES = ["true_target_only", "spec_draft_max_0", "speculative"]
METRICS = [
    "total_decode_time_ms",
    "draft_time_ms",
    "target_verify_time_ms",
    "accept_reject_time_ms",
    "generated_tokens",
    "drafted_tokens",
    "accepted_tokens",
    "acceptance_rate",
    "decode_tok_per_s",
]
SAMPLE_METRICS = [
    "total_decode_time_ms",
    "target_verify_time_ms",
    "draft_time_ms",
    "decode_tok_per_s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def to_float(value: str, row_num: int, field: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        print(f"warning: row {row_num}: invalid numeric value for {field}: {value!r}", file=sys.stderr)
        return None


def load_rows(path: Path) -> dict[str, list[dict[str, float]]]:
    by_mode: dict[str, list[dict[str, float]]] = defaultdict(list)

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"mode", *METRICS} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"missing required column(s): {', '.join(sorted(missing))}")

        for row_num, row in enumerate(reader, start=2):
            mode = row.get("mode", "")
            if mode not in MODES:
                print(f"warning: row {row_num}: skipping unknown mode: {mode!r}", file=sys.stderr)
                continue

            parsed: dict[str, float] = {}
            ok = True
            for metric in METRICS:
                value = to_float(row.get(metric, ""), row_num, metric)
                if value is None:
                    ok = False
                    break
                parsed[metric] = value
            if ok:
                by_mode[mode].append(parsed)

    for mode in MODES:
        if mode not in by_mode:
            print(f"warning: mode missing from input: {mode}", file=sys.stderr)

    return by_mode


def metric_mean(rows: list[dict[str, float]], metric: str) -> float | None:
    if not rows:
        return None
    return mean(row[metric] for row in rows)


def metric_std(rows: list[dict[str, float]], metric: str) -> float | None:
    if not rows:
        return None
    if len(rows) == 1:
        return 0.0
    return stdev(row[metric] for row in rows)


def metric_median(rows: list[dict[str, float]], metric: str) -> float | None:
    if not rows:
        return None
    return median(row[metric] for row in rows)


def metric_min(rows: list[dict[str, float]], metric: str) -> float | None:
    if not rows:
        return None
    return min(row[metric] for row in rows)


def metric_max(rows: list[dict[str, float]], metric: str) -> float | None:
    if not rows:
        return None
    return max(row[metric] for row in rows)


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def print_summary(summary_rows: list[dict[str, str]]) -> None:
    columns = ["mode", "count"]
    for metric in METRICS:
        columns.append(f"{metric}_mean")
        columns.append(f"{metric}_std")
        columns.append(f"{metric}_median")
        columns.append(f"{metric}_min")
        columns.append(f"{metric}_max")

    widths = {col: len(col) for col in columns}
    for row in summary_rows:
        for col in columns:
            widths[col] = max(widths[col], len(row.get(col, "")))

    print("Summary by mode")
    print("  ".join(col.ljust(widths[col]) for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in summary_rows:
        print("  ".join(row.get(col, "").ljust(widths[col]) for col in columns))


def print_samples(by_mode: dict[str, list[dict[str, float]]]) -> None:
    columns = ["mode", "index", *SAMPLE_METRICS]
    widths = {col: len(col) for col in columns}
    rows_out: list[dict[str, str]] = []
    for mode in MODES:
        for index, row in enumerate(by_mode.get(mode, [])):
            out = {"mode": mode, "index": str(index)}
            for metric in SAMPLE_METRICS:
                out[metric] = fmt(row[metric])
            rows_out.append(out)
            for col in columns:
                widths[col] = max(widths[col], len(out.get(col, "")))

    print("\nPer-sample metrics")
    print("  ".join(col.ljust(widths[col]) for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in rows_out:
        print("  ".join(row.get(col, "").ljust(widths[col]) for col in columns))


def print_variance_warnings(summary_rows: list[dict[str, str]]) -> None:
    for row in summary_rows:
        mode = row["mode"]
        for metric in METRICS:
            mean_value = row.get(f"{metric}_mean", "")
            std_value = row.get(f"{metric}_std", "")
            if not mean_value or not std_value:
                continue
            m = float(mean_value)
            s = float(std_value)
            if s > m:
                print(f"warning: high variance detected: mode={mode} metric={metric} std={s:.6f} mean={m:.6f}", file=sys.stderr)


def safe_sub(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0.0:
        return None
    return a / b


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    by_mode = load_rows(args.input)

    summary_rows: list[dict[str, str]] = []
    means: dict[str, dict[str, float | None]] = {}
    medians: dict[str, dict[str, float | None]] = {}

    for mode in MODES:
        rows = by_mode.get(mode, [])
        row: dict[str, str] = {"mode": mode, "count": str(len(rows))}
        means[mode] = {}
        medians[mode] = {}
        for metric in METRICS:
            m = metric_mean(rows, metric)
            s = metric_std(rows, metric)
            med = metric_median(rows, metric)
            mn = metric_min(rows, metric)
            mx = metric_max(rows, metric)
            means[mode][metric] = m
            medians[mode][metric] = med
            row[f"{metric}_mean"] = fmt(m)
            row[f"{metric}_std"] = fmt(s)
            row[f"{metric}_median"] = fmt(med)
            row[f"{metric}_min"] = fmt(mn)
            row[f"{metric}_max"] = fmt(mx)
        summary_rows.append(row)

    print_summary(summary_rows)
    print_samples(by_mode)
    print_variance_warnings(summary_rows)

    diagnosis = {
        "framework_overhead_ms": safe_sub(
            means["spec_draft_max_0"]["total_decode_time_ms"],
            means["true_target_only"]["total_decode_time_ms"],
        ),
        "speculative_vs_true_speedup": safe_div(
            means["speculative"]["decode_tok_per_s"],
            means["true_target_only"]["decode_tok_per_s"],
        ),
        "speculative_vs_zero_draft_speedup": safe_div(
            means["speculative"]["decode_tok_per_s"],
            means["spec_draft_max_0"]["decode_tok_per_s"],
        ),
        "target_verify_reduction_ms": safe_sub(
            means["spec_draft_max_0"]["target_verify_time_ms"],
            means["speculative"]["target_verify_time_ms"],
        ),
        "framework_overhead_ms_median": safe_sub(
            medians["spec_draft_max_0"]["total_decode_time_ms"],
            medians["true_target_only"]["total_decode_time_ms"],
        ),
        "speculative_vs_true_speedup_median": safe_div(
            medians["speculative"]["decode_tok_per_s"],
            medians["true_target_only"]["decode_tok_per_s"],
        ),
        "speculative_vs_zero_draft_speedup_median": safe_div(
            medians["speculative"]["decode_tok_per_s"],
            medians["spec_draft_max_0"]["decode_tok_per_s"],
        ),
        "target_verify_reduction_ms_median": safe_sub(
            medians["spec_draft_max_0"]["target_verify_time_ms"],
            medians["speculative"]["target_verify_time_ms"],
        ),
        "draft_overhead_ms": means["speculative"]["draft_time_ms"],
        "net_time_saved_ms": safe_sub(
            means["true_target_only"]["total_decode_time_ms"],
            means["speculative"]["total_decode_time_ms"],
        ),
        "accepted_tokens_per_drafted_token": safe_div(
            means["speculative"]["accepted_tokens"],
            means["speculative"]["drafted_tokens"],
        ),
        "accepted_tokens_per_sample": means["speculative"]["accepted_tokens"],
    }

    print("\nDiagnosis")
    for name, value in diagnosis.items():
        print(f"{name},{fmt(value)}")

    columns = ["mode", "count"]
    for metric in METRICS:
        columns.append(f"{metric}_mean")
        columns.append(f"{metric}_std")
        columns.append(f"{metric}_median")
        columns.append(f"{metric}_min")
        columns.append(f"{metric}_max")
    write_csv(args.out, summary_rows, columns)

    diagnosis_out = args.out.with_name(args.out.stem.replace("_summary", "") + "_diagnosis.csv")
    write_csv(
        diagnosis_out,
        [{"metric": name, "value": fmt(value)} for name, value in diagnosis.items()],
        ["metric", "value"],
    )

    print(f"\nwrote {args.out}")
    print(f"wrote {diagnosis_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
