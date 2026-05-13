#!/usr/bin/env python3
"""Run Qwen3 CNN/DailyMail raw-prompt llama-speculative-simple experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


PROMPT_TEMPLATE = """Summarize the following CNN/DailyMail news article in 2-3 concise sentences.
Return only the summary. Do not include reasoning.

Article:
{article}

Summary:"""

DEFAULT_DATASET = Path("datasets/cnn_dailymail_3.0.0_test_128.jsonl")
DEFAULT_SPEC_BIN = Path("build/bin/llama-speculative-simple")
DEFAULT_DRAFT_MODEL = Path("/data/data/com.termux/files/home/models/Qwen3-0.6B-Q4_K_M.gguf")
DEFAULT_TARGET_MODEL = Path("/data/data/com.termux/files/home/models/Qwen3-8B-Q4_K_M.gguf")
DEFAULT_OUT_DIR = Path("run-results")

ENCODED_RE = re.compile(
    r"encoded\s+(?P<tokens>\d+)\s+tokens\s+in\s+(?P<seconds>[0-9.]+)\s+seconds,\s+speed:\s+(?P<tps>[0-9.]+)\s+t/s"
)
DECODED_RE = re.compile(
    r"decoded\s+(?P<tokens>\d+)\s+tokens\s+in\s+(?P<seconds>[0-9.]+)\s+seconds,\s+speed:\s+(?P<tps>[0-9.]+)\s+t/s"
)
PROMPT_EVAL_RE = re.compile(
    r"common_perf_print:\s+prompt eval time\s+=\s+(?P<ms>[0-9.]+)\s+ms\s+/\s+(?P<tokens>\d+)\s+tokens.*?token,\s+(?P<tps>[0-9.]+)\s+tokens per second\)"
)
EVAL_RE = re.compile(
    r"common_perf_print:\s+eval time\s+=\s+(?P<ms>[0-9.]+)\s+ms\s+/\s+(?P<runs>\d+)\s+runs.*?token,\s+(?P<tps>[0-9.]+)\s+tokens per second\)"
)
SPEC_INT_RE = {
    "n_draft": re.compile(r"n_draft\s+=\s+(?P<value>\d+)"),
    "n_predict": re.compile(r"n_predict\s+=\s+(?P<value>\d+)"),
    "n_drafted": re.compile(r"n_drafted\s+=\s+(?P<value>\d+)"),
    "n_accept": re.compile(r"n_accept\s+=\s+(?P<value>\d+)"),
}
ACCEPT_RE = re.compile(r"accept\s+=\s+(?P<value>[0-9.]+|nan)%")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--llama-speculative-simple", type=Path, default=DEFAULT_SPEC_BIN)
    parser.add_argument("--draft-model", type=Path, default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--target-model", type=Path, default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--max-article-chars", type=int, default=3500)
    parser.add_argument("--n-predict", type=int, default=160)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gpu-layers", default="auto")
    parser.add_argument("--draft-gpu-layers", default="auto")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--threads-batch", type=int, default=0)
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra arg passed to every run; repeat for multiple args")
    parser.add_argument("--logs-dir", type=Path)
    parser.add_argument("--allow-missing-q4km-draft", action="store_true")
    parser.add_argument(
        "--run",
        action="append",
        choices=["draft-only", "target-only", "speculative"],
        help="Run only the selected experiment; repeat for multiple. Default runs all three.",
    )
    parser.add_argument("--resume", action="store_true", help="Append only missing samples for the selected run(s)")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_rows(path: Path, samples: int, skip: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows[skip : skip + samples]


def make_prompt(article: str, max_chars: int) -> str:
    return PROMPT_TEMPLATE.format(article=article[:max_chars].strip())


def check_paths(args: argparse.Namespace) -> None:
    missing = [
        str(path)
        for path in (args.dataset, args.llama_speculative_simple, args.draft_model, args.target_model)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("missing required file(s):\n  " + "\n  ".join(missing))
    if "Qwen3-0.6B-Q4_K_M.gguf" not in args.draft_model.name and not args.allow_missing_q4km_draft:
        raise ValueError(
            f"draft model is not Qwen3-0.6B-Q4_K_M.gguf: {args.draft_model}\n"
            "Pass the exact Q4_K_M file, or add --allow-missing-q4km-draft if you intentionally want a different quant."
        )


def command_for(run: dict[str, Any], prompt_path: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        str(args.llama_speculative_simple),
        "-m",
        str(run["target_model"]),
        "-md",
        str(run["draft_model"]),
        "-f",
        str(prompt_path),
        "-n",
        str(args.n_predict),
        "-c",
        str(args.ctx_size),
        "-cd",
        str(args.ctx_size),
        "-b",
        str(args.batch_size),
        "--temp",
        str(args.temp),
        "--seed",
        str(args.seed),
        "--draft-max",
        str(run["draft_max"]),
        "-ngl",
        str(args.gpu_layers),
        "-ngld",
        str(args.draft_gpu_layers),
    ]
    if args.threads > 0:
        cmd += ["-t", str(args.threads)]
    if args.threads_batch > 0:
        cmd += ["-tb", str(args.threads_batch)]
    cmd += args.extra_arg
    return cmd


def run_cmd(cmd: list[str]) -> tuple[int, str, str, float]:
    t0 = time.monotonic()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return proc.returncode, proc.stdout, proc.stderr, time.monotonic() - t0


def clean_prediction(stdout: str, prompt: str) -> str:
    text = ANSI_RE.sub("", stdout).replace("\r\n", "\n")
    pos = text.find(prompt)
    if pos >= 0:
        text = text[pos + len(prompt) :]
    text = re.sub(r"<\|im_end\|>|<\|endoftext\|>|</s>", "", text)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append(line)
            continue
        if re.match(r"^(build:|llama_|common_|load_|print_info:|encoded\s+|decoded\s+|n_draft\s*=|n_predict\s*=|n_drafted\s*=|n_accept\s*=|accept\s*=)", stripped):
            continue
        if re.match(r"^(Loading model|Exiting|Available devices:)", stripped, re.IGNORECASE):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def parse_metrics(stderr: str, wall_seconds: float) -> dict[str, Any]:
    metrics: dict[str, Any] = {"wall_seconds": wall_seconds}
    text = ANSI_RE.sub("", stderr)
    if match := ENCODED_RE.search(text):
        metrics["encoded_tokens"] = int(match.group("tokens"))
        metrics["encode_seconds"] = float(match.group("seconds"))
        metrics["encode_tps"] = float(match.group("tps"))
    if match := DECODED_RE.search(text):
        metrics["decoded_tokens"] = int(match.group("tokens"))
        metrics["decode_seconds"] = float(match.group("seconds"))
        metrics["decode_tps"] = float(match.group("tps"))
    prompt_match = None
    for prompt_match in PROMPT_EVAL_RE.finditer(text):
        pass
    if prompt_match:
        metrics["prompt_eval_tokens"] = int(prompt_match.group("tokens"))
        metrics["prompt_eval_tps"] = float(prompt_match.group("tps"))
        metrics.setdefault("encode_tps", metrics["prompt_eval_tps"])
    eval_match = None
    for eval_match in EVAL_RE.finditer(text):
        pass
    if eval_match:
        metrics["eval_runs"] = int(eval_match.group("runs"))
        metrics["eval_tps"] = float(eval_match.group("tps"))
        metrics.setdefault("decode_tps", metrics["eval_tps"])
    for key, pattern in SPEC_INT_RE.items():
        if match := pattern.search(text):
            metrics[key] = int(match.group("value"))
    if match := ACCEPT_RE.search(text):
        metrics["accept_pct"] = None if match.group("value") == "nan" else float(match.group("value"))
    return metrics


def tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def ngrams(items: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(items[i : i + n]) for i in range(max(0, len(items) - n + 1)))


def score_ngram(pred: list[str], ref: list[str], n: int) -> dict[str, float]:
    pred_counts = ngrams(pred, n)
    ref_counts = ngrams(ref, n)
    overlap = sum((pred_counts & ref_counts).values())
    pred_total = sum(pred_counts.values())
    ref_total = sum(ref_counts.values())
    precision = overlap / pred_total if pred_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    fmeasure = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "fmeasure": fmeasure}


def lcs_len(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def score_l(pred: list[str], ref: list[str]) -> dict[str, float]:
    overlap = lcs_len(pred, ref)
    precision = overlap / len(pred) if pred else 0.0
    recall = overlap / len(ref) if ref else 0.0
    fmeasure = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "fmeasure": fmeasure}


def rouge(prediction: str, reference: str) -> dict[str, dict[str, float]]:
    pred_tokens = tokens(prediction)
    ref_tokens = tokens(reference)
    return {
        "rouge1": score_ngram(pred_tokens, ref_tokens, 1),
        "rouge2": score_ngram(pred_tokens, ref_tokens, 2),
        "rougeL": score_l(pred_tokens, ref_tokens),
    }


def avg(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return mean(vals) if vals else None


def summarize(rows: list[dict[str, Any]], run: dict[str, Any], args: argparse.Namespace, devices: str) -> dict[str, Any]:
    ok = [r for r in rows if r.get("ok")]
    summary: dict[str, Any] = {
        "samples_total": len(rows),
        "samples_ok": len(ok),
        "rouge1_f": avg(ok, "rouge1_fmeasure"),
        "rouge2_f": avg(ok, "rouge2_fmeasure"),
        "rougeL_f": avg(ok, "rougeL_fmeasure"),
        "rouge1_recall": avg(ok, "rouge1_recall"),
        "rouge2_recall": avg(ok, "rouge2_recall"),
        "rougeL_recall": avg(ok, "rougeL_recall"),
        "decode_tps_mean": avg(ok, "decode_tps"),
        "encode_tps_mean": avg(ok, "encode_tps"),
        "wall_seconds_mean": avg(ok, "wall_seconds"),
        "config": {
            "run_name": run["name"],
            "binary": str(args.llama_speculative_simple),
            "dataset": str(args.dataset),
            "target_model": str(run["target_model"]),
            "draft_model": str(run["draft_model"]),
            "prompt_template": PROMPT_TEMPLATE,
            "max_article_chars": args.max_article_chars,
            "n_predict": args.n_predict,
            "ctx_size": args.ctx_size,
            "ctx_size_draft": args.ctx_size,
            "batch_size_actual": args.batch_size,
            "temp": args.temp,
            "seed": args.seed,
            "draft_max": run["draft_max"],
            "gpu_layers_arg": args.gpu_layers,
            "draft_gpu_layers_arg": args.draft_gpu_layers,
            "list_devices_output": devices.strip(),
            "effective_offload_note": "no devices listed; CPU only" if devices.strip() == "Available devices:" else "see list_devices_output and llama.cpp logs",
        },
    }
    if run["draft_max"] > 0:
        summary.update(
            {
                "accept_pct_mean": avg(ok, "accept_pct"),
                "n_drafted_mean": avg(ok, "n_drafted"),
                "n_accept_mean": avg(ok, "n_accept"),
                "accepted_per_drafted_mean": avg(ok, "accepted_per_drafted"),
            }
        )
    return summary


def flatten_row(base: dict[str, Any], prediction: str, metrics: dict[str, Any], scores: dict[str, dict[str, float]]) -> dict[str, Any]:
    row = {
        **base,
        "prediction": prediction,
        "ok": True,
        "rouge1_precision": scores["rouge1"]["precision"],
        "rouge1_recall": scores["rouge1"]["recall"],
        "rouge1_fmeasure": scores["rouge1"]["fmeasure"],
        "rouge2_precision": scores["rouge2"]["precision"],
        "rouge2_recall": scores["rouge2"]["recall"],
        "rouge2_fmeasure": scores["rouge2"]["fmeasure"],
        "rougeL_precision": scores["rougeL"]["precision"],
        "rougeL_recall": scores["rougeL"]["recall"],
        "rougeL_fmeasure": scores["rougeL"]["fmeasure"],
    }
    row.update(metrics)
    if row.get("n_drafted"):
        row["accepted_per_drafted"] = float(row.get("n_accept", 0)) / float(row["n_drafted"])
    return row


def run_one_experiment(run: dict[str, Any], rows: list[dict[str, Any]], args: argparse.Namespace, devices: str) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.out_dir / f"{run['prefix']}-predictions.jsonl"
    summary_path = args.out_dir / f"{run['prefix']}-summary.json"
    logs_dir = args.logs_dir / run["name"] if args.logs_dir else None
    if logs_dir:
        logs_dir.mkdir(parents=True, exist_ok=True)

    out_rows: list[dict[str, Any]] = []
    done_indices: set[int] = set()
    if args.resume and pred_path.exists():
        with pred_path.open("r", encoding="utf-8") as existing:
            for line in existing:
                if not line.strip():
                    continue
                row = json.loads(line)
                out_rows.append(row)
                if row.get("ok") and isinstance(row.get("sample_index"), int):
                    done_indices.add(int(row["sample_index"]))

    mode = "a" if args.resume else "w"
    with pred_path.open(mode, encoding="utf-8") as out:
        for i, source in enumerate(rows, start=1):
            article = str(source.get("article", ""))
            reference = str(source.get("highlights", ""))
            sample_id = source.get("id", args.skip + i - 1)
            sample_index = args.skip + i - 1
            if sample_index in done_indices:
                continue
            prompt = make_prompt(article, args.max_article_chars)
            base = {
                "id": sample_id,
                "sample_index": sample_index,
                "article_chars": len(article[: args.max_article_chars].strip()),
                "reference": reference,
            }
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".prompt", delete=False) as f:
                f.write(prompt)
                prompt_path = Path(f.name)
            try:
                cmd = command_for(run, prompt_path, args)
                print(f"[{run['name']}] {i}/{len(rows)} id={sample_id}", file=sys.stderr, flush=True)
                if args.dry_run:
                    row = {**base, "ok": True, "command": cmd}
                else:
                    rc, stdout, stderr, wall = run_cmd(cmd)
                    if logs_dir:
                        (logs_dir / f"{i:05d}-{sample_id}.stdout.txt").write_text(stdout, encoding="utf-8")
                        (logs_dir / f"{i:05d}-{sample_id}.stderr.txt").write_text(stderr, encoding="utf-8")
                    prediction = clean_prediction(stdout, prompt)
                    metrics = parse_metrics(stderr, wall)
                    metrics["returncode"] = rc
                    if rc != 0:
                        row = {**base, "ok": False, "error": f"exit code {rc}", **metrics, "prediction": prediction}
                    else:
                        row = flatten_row(base, prediction, metrics, rouge(prediction, reference))
                out_rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
            finally:
                try:
                    os.unlink(prompt_path)
                except OSError:
                    pass

    summary = summarize(out_rows, run, args, devices)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run": run["name"], **summary}, ensure_ascii=False, indent=2), flush=True)


def list_devices(binary: Path) -> str:
    proc = subprocess.run([str(binary), "--list-devices"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return proc.stdout


def main() -> int:
    args = parse_args()
    check_paths(args)
    rows = load_rows(args.dataset, args.samples, args.skip)
    devices = list_devices(args.llama_speculative_simple)
    runs = [
        {
            "name": "draft-only",
            "prefix": "cnndm-qwen3-0.6b-raw-specsimple-phone",
            "target_model": args.draft_model,
            "draft_model": args.draft_model,
            "draft_max": 0,
        },
        {
            "name": "target-only",
            "prefix": "cnndm-qwen3-8b-raw-specsimple-phone",
            "target_model": args.target_model,
            "draft_model": args.draft_model,
            "draft_max": 0,
        },
        {
            "name": "speculative",
            "prefix": "cnndm-qwen3-8b-0.6b-specsimple-phone",
            "target_model": args.target_model,
            "draft_model": args.draft_model,
            "draft_max": 16,
        },
    ]
    for run in runs:
        if args.run and run["name"] not in args.run:
            continue
        run_one_experiment(run, rows, args, devices)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
