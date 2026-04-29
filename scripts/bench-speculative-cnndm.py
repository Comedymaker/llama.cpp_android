#!/usr/bin/env python3
"""Benchmark target, draft, and speculative decoding on CNN/DailyMail prompts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from statistics import mean
from typing import Any


DECODED_RE = re.compile(
    r"decoded\s+(?P<tokens>\d+)\s+tokens\s+in\s+(?P<seconds>[0-9.]+)\s+seconds,\s+speed:\s+(?P<tps>[0-9.]+)\s+t/s"
)
EVAL_RE = re.compile(
    r"common_perf_print:\s+eval time\s+=\s+(?P<ms>[0-9.]+)\s+ms\s+/\s+(?P<runs>\d+)\s+runs.*\(\s*(?P<ms_tok>[0-9.]+)\s+ms per token,\s+(?P<tps>[0-9.]+)\s+tokens per second\)"
)
PROMPT_EVAL_RE = re.compile(
    r"common_perf_print:\s+prompt eval time\s+=\s+(?P<ms>[0-9.]+)\s+ms\s+/\s+(?P<tokens>\d+)\s+tokens.*\(\s*(?P<ms_tok>[0-9.]+)\s+ms per token,\s+(?P<tps>[0-9.]+)\s+tokens per second\)"
)
SPEC_INT_RE = {
    "n_predict": re.compile(r"n_predict\s+=\s+(?P<value>\d+)"),
    "n_drafted": re.compile(r"n_drafted\s+=\s+(?P<value>\d+)"),
    "n_accept": re.compile(r"n_accept\s+=\s+(?P<value>\d+)"),
}
ACCEPT_RE = re.compile(r"accept\s+=\s+(?P<value>[0-9.]+)%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CNN/DailyMail summarization prompts through llama-cli target, "
            "llama-cli draft, and llama-speculative, then aggregate throughput, "
            "speedup, and acceptance rate."
        )
    )
    parser.add_argument("--target-model", required=True, help="Path to the target GGUF model")
    parser.add_argument("--draft-model", required=True, help="Path to the draft GGUF model")
    parser.add_argument("--llama-cli", default="build/bin/llama-completion", help="Path to llama-completion or another non-interactive completion binary")
    parser.add_argument("--llama-speculative", default="build/bin/llama-speculative", help="Path to llama-speculative")
    parser.add_argument("--samples", type=int, default=16, help="Number of CNN/DM examples to benchmark")
    parser.add_argument("--skip", type=int, default=0, help="Number of dataset examples to skip before sampling")
    parser.add_argument("--hf-dataset", default="abisee/cnn_dailymail", help="Hugging Face dataset id for CNN/DM")
    parser.add_argument("--hf-config", default="3.0.0", help="Hugging Face dataset config for CNN/DM")
    parser.add_argument("--split", default="test", help="CNN/DM split when using --dataset-source hf")
    parser.add_argument("--dataset-source", choices=["local", "hf", "hf-viewer"], default="local")
    parser.add_argument("--dataset-file", help="Local CNN/DM JSONL/JSON/CSV file with article and highlights fields")
    parser.add_argument("--article-field", default="article", help="Field name for the article text")
    parser.add_argument("--summary-field", default="highlights", help="Field name for the reference summary")
    parser.add_argument("--max-article-chars", type=int, default=3500, help="Truncate each article before prompt formatting")
    parser.add_argument("--n-predict", type=int, default=128, help="Generated tokens per run")
    parser.add_argument("--ctx-size", type=int, default=4096, help="Context size passed to all binaries")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size passed to all binaries")
    parser.add_argument("--threads", type=int, default=0, help="Threads passed to all binaries; 0 keeps llama.cpp default")
    parser.add_argument("--seed", type=int, default=1234, help="Sampler seed passed to all binaries")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature; 0.0 keeps decoding deterministic")
    parser.add_argument("--common-args", default="", help="Extra shell-style args passed to all three runs")
    parser.add_argument("--target-args", default="", help="Extra shell-style args passed only to target completion run")
    parser.add_argument("--draft-args", default="", help="Extra shell-style args passed only to draft completion run")
    parser.add_argument("--spec-args", default="", help="Extra shell-style args passed only to llama-speculative")
    parser.add_argument("--out-jsonl", default="bench-cnndm-speculative.jsonl", help="Per-sample metrics output")
    parser.add_argument("--summary-json", default="bench-cnndm-speculative-summary.json", help="Aggregate metrics output")
    parser.add_argument("--keep-logs-dir", help="Directory to keep raw stdout/stderr logs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands for the first sample without running them")
    return parser.parse_args()


def format_prompt(article: str) -> str:
    return (
        "Summarize the following CNN/DailyMail news article in a concise paragraph.\n\n"
        "Article:\n"
        f"{article.strip()}\n\n"
        "Summary:"
    )


def load_local_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "rows", "examples"):
                if isinstance(data.get(key), list):
                    return data[key]
        raise ValueError(f"unsupported JSON structure in {path}")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"unsupported dataset file suffix: {suffix}")


def load_hf_rows(dataset: str, config: str, split: str, samples: int, skip: int) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError("install the 'datasets' package or use --dataset-source local --dataset-file ...") from exc

    ds = load_dataset(dataset, config, split=f"{split}[{skip}:{skip + samples}]")
    return [dict(row) for row in ds]


def load_hf_viewer_rows(dataset: str, config: str, split: str, samples: int, skip: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = skip
    remaining = samples

    while remaining > 0:
        length = min(100, remaining)
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))

        batch = [item["row"] for item in payload.get("rows", []) if "row" in item]
        if not batch:
            break
        rows.extend(batch)
        got = len(batch)
        offset += got
        remaining -= got
        if got < length:
            break

    return rows


def load_prompts(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.dataset_source == "hf":
        rows = load_hf_rows(args.hf_dataset, args.hf_config, args.split, args.samples, args.skip)
    elif args.dataset_source == "hf-viewer":
        rows = load_hf_viewer_rows(args.hf_dataset, args.hf_config, args.split, args.samples, args.skip)
    else:
        if not args.dataset_file:
            raise ValueError("--dataset-file is required when --dataset-source local")
        rows = load_local_rows(Path(args.dataset_file))
        rows = rows[args.skip : args.skip + args.samples]

    prompts = []
    for idx, row in enumerate(rows):
        article = str(row.get(args.article_field, "")).strip()
        if not article:
            continue
        article = article[: args.max_article_chars]
        prompts.append(
            {
                "sample_id": args.skip + idx,
                "prompt": format_prompt(article),
                "article_chars": len(article),
                "reference": str(row.get(args.summary_field, "")).strip(),
            }
        )
    if not prompts:
        raise ValueError("no usable CNN/DM rows found")
    return prompts


def base_args(args: argparse.Namespace) -> list[str]:
    out = ["-n", str(args.n_predict), "-c", str(args.ctx_size), "-b", str(args.batch_size), "--seed", str(args.seed)]
    if args.threads > 0:
        out += ["-t", str(args.threads), "-tb", str(args.threads)]
    if args.temp >= 0.0:
        out += ["--temp", str(args.temp)]
    out += shlex.split(args.common_args)
    return out


def command_for(kind: str, prompt_file: Path, args: argparse.Namespace) -> list[str]:
    common = base_args(args) + ["-f", str(prompt_file)]
    completion_common = ["--no-warmup", "--no-display-prompt", "--no-conversation", "--single-turn", "--simple-io"]
    if kind == "target":
        return [args.llama_cli, "-m", args.target_model] + common + completion_common + shlex.split(args.target_args)
    if kind == "draft":
        return [args.llama_cli, "-m", args.draft_model] + common + completion_common + shlex.split(args.draft_args)
    if kind == "spec":
        return (
            [args.llama_speculative, "-m", args.target_model, "-md", args.draft_model]
            + common
            + shlex.split(args.spec_args)
        )
    raise ValueError(kind)


def run_command(cmd: list[str]) -> tuple[int, str, float]:
    t0 = time.monotonic()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return proc.returncode, proc.stdout, time.monotonic() - t0


def parse_metrics(kind: str, text: str, wall_seconds: float) -> dict[str, Any]:
    metrics: dict[str, Any] = {"wall_seconds": wall_seconds}

    decoded = DECODED_RE.search(text)
    if decoded:
        metrics["decoded_tokens"] = int(decoded.group("tokens"))
        metrics["decode_seconds"] = float(decoded.group("seconds"))
        metrics["decode_tps"] = float(decoded.group("tps"))

    eval_match = None
    for eval_match in EVAL_RE.finditer(text):
        pass
    if eval_match:
        metrics["eval_ms"] = float(eval_match.group("ms"))
        metrics["eval_runs"] = int(eval_match.group("runs"))
        metrics["eval_tps"] = float(eval_match.group("tps"))

    prompt_match = None
    for prompt_match in PROMPT_EVAL_RE.finditer(text):
        pass
    if prompt_match:
        metrics["prompt_eval_ms"] = float(prompt_match.group("ms"))
        metrics["prompt_tokens"] = int(prompt_match.group("tokens"))
        metrics["prompt_eval_tps"] = float(prompt_match.group("tps"))

    if kind == "spec":
        for key, pattern in SPEC_INT_RE.items():
            match = pattern.search(text)
            if match:
                metrics[key] = int(match.group("value"))
        accept = ACCEPT_RE.search(text)
        if accept:
            metrics["accept_pct"] = float(accept.group("value"))

    if "decode_tps" not in metrics and "eval_tps" in metrics:
        metrics["decode_tps"] = metrics["eval_tps"]
    return metrics


def write_log(logs_dir: Path | None, sample_id: int, kind: str, text: str) -> None:
    if logs_dir is None:
        return
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / f"sample-{sample_id:05d}-{kind}.log").write_text(text, encoding="utf-8")


def run_sample(sample: dict[str, Any], args: argparse.Namespace, logs_dir: Path | None) -> dict[str, Any]:
    sample_id = int(sample["sample_id"])
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".prompt", delete=False) as f:
        f.write(sample["prompt"])
        prompt_path = Path(f.name)

    try:
        result: dict[str, Any] = {
            "sample_id": sample_id,
            "article_chars": sample["article_chars"],
            "reference_chars": len(sample.get("reference", "")),
        }

        for kind in ("draft", "target", "spec"):
            cmd = command_for(kind, prompt_path, args)
            if args.dry_run:
                result[f"{kind}_cmd"] = cmd
                continue
            rc, output, wall_seconds = run_command(cmd)
            write_log(logs_dir, sample_id, kind, output)
            metrics = parse_metrics(kind, output, wall_seconds)
            metrics["returncode"] = rc
            result[kind] = metrics
            if rc != 0:
                result["error"] = f"{kind} failed with exit code {rc}"
                break

        if not args.dry_run and all(k in result for k in ("draft", "target", "spec")):
            target_tps = result["target"].get("decode_tps")
            spec_tps = result["spec"].get("decode_tps")
            draft_tps = result["draft"].get("decode_tps")
            if target_tps and spec_tps:
                result["speedup_vs_target"] = spec_tps / target_tps
            if draft_tps and spec_tps:
                result["spec_vs_draft_tps_ratio"] = spec_tps / draft_tps
        return result
    finally:
        try:
            os.unlink(prompt_path)
        except OSError:
            pass


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("error") and all(k in r for k in ("draft", "target", "spec"))]

    def avg(path: tuple[str, str]) -> float | None:
        vals = []
        for row in ok:
            value = row.get(path[0], {}).get(path[1])
            if isinstance(value, (int, float)):
                vals.append(float(value))
        return mean(vals) if vals else None

    speedups = [float(r["speedup_vs_target"]) for r in ok if isinstance(r.get("speedup_vs_target"), (int, float))]
    return {
        "samples_total": len(rows),
        "samples_ok": len(ok),
        "draft_decode_tps_mean": avg(("draft", "decode_tps")),
        "target_decode_tps_mean": avg(("target", "decode_tps")),
        "spec_decode_tps_mean": avg(("spec", "decode_tps")),
        "speedup_vs_target_mean": mean(speedups) if speedups else None,
        "spec_accept_pct_mean": avg(("spec", "accept_pct")),
        "spec_n_predict_mean": avg(("spec", "n_predict")),
        "spec_n_drafted_mean": avg(("spec", "n_drafted")),
        "spec_n_accept_mean": avg(("spec", "n_accept")),
    }


def main() -> int:
    args = parse_args()
    prompts = load_prompts(args)
    logs_dir = Path(args.keep_logs_dir) if args.keep_logs_dir else None

    rows = []
    out_jsonl = Path(args.out_jsonl)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for i, sample in enumerate(prompts, start=1):
            print(f"[{i}/{len(prompts)}] sample_id={sample['sample_id']}", file=sys.stderr, flush=True)
            row = run_sample(sample, args, logs_dir)
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if args.dry_run:
                print(json.dumps(row, ensure_ascii=False, indent=2))
                break

    summary = summarize(rows)
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    return 0 if summary["samples_ok"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
