# CNN/DailyMail Speculative Decoding Benchmark

Date: 2026-04-29

## Setup

Dataset:

- Local file: `datasets/cnn_dailymail_3.0.0_test_128.jsonl`
- Source: `abisee/cnn_dailymail`, config `3.0.0`, split `test`
- Evaluated sample ids: `0..7`
- Prompt article truncation: `--max-article-chars 1000`

Prompt template:

```text
Summarize the following CNN/DailyMail news article in a concise paragraph.

Article:
{article}

Summary:
```

Models:

- Draft: `/data/data/com.termux/files/home/models/Qwen3-0.6B-Q4_K_S.gguf`
- Target: `/data/data/com.termux/files/home/models/Qwen3-4B-Q4_K_S.gguf`

Common generation settings:

- `--samples 8`
- `--n-predict 64`
- `--ctx-size 2048`
- `--threads 8`
- `--temp 0.0`
- `--seed 1234`

The benchmark script used for all runs is:

```sh
python3 scripts/bench-speculative-cnndm.py
```

Single-model throughput uses `build/bin/llama-completion` instead of `llama-cli`, because this build of `llama-cli` enters conversation mode for Qwen chat-template models and does not support `--no-conversation`. `llama-completion` supports non-interactive completion mode and produces stable `common_perf_print` throughput output.

## Full Benchmark Command

This command runs draft-only, target-only, and speculative decoding on the same samples and writes per-sample JSONL plus a summary JSON:

```sh
python3 scripts/bench-speculative-cnndm.py \
  --target-model /data/data/com.termux/files/home/models/Qwen3-4B-Q4_K_S.gguf \
  --draft-model /data/data/com.termux/files/home/models/Qwen3-0.6B-Q4_K_S.gguf \
  --dataset-source local \
  --dataset-file datasets/cnn_dailymail_3.0.0_test_128.jsonl \
  --samples 8 \
  --max-article-chars 1000 \
  --n-predict 64 \
  --ctx-size 2048 \
  --threads 8 \
  --out-jsonl bench-cnndm-speculative-8-complete.jsonl \
  --summary-json bench-cnndm-speculative-8-complete-summary.json \
  --keep-logs-dir bench-cnndm-logs-8
```

Output files:

- Per-sample metrics: `bench-cnndm-speculative-8-complete.jsonl`
- Summary metrics: `bench-cnndm-speculative-8-complete-summary.json`
- Raw logs: `bench-cnndm-logs-8/`

## Individual Test Commands

For one prompt file, the benchmark script executes the following three command forms.

### Draft Model Throughput

```sh
build/bin/llama-completion \
  -m /data/data/com.termux/files/home/models/Qwen3-0.6B-Q4_K_S.gguf \
  -n 64 \
  -c 2048 \
  -b 2048 \
  --seed 1234 \
  -t 8 \
  -tb 8 \
  --temp 0.0 \
  -f <prompt-file> \
  --no-warmup \
  --no-display-prompt \
  --no-conversation \
  --single-turn \
  --simple-io
```

Metric parsed from output:

```text
common_perf_print:        eval time = ... / 63 runs (..., XX.XX tokens per second)
```

### Target Model Throughput

```sh
build/bin/llama-completion \
  -m /data/data/com.termux/files/home/models/Qwen3-4B-Q4_K_S.gguf \
  -n 64 \
  -c 2048 \
  -b 2048 \
  --seed 1234 \
  -t 8 \
  -tb 8 \
  --temp 0.0 \
  -f <prompt-file> \
  --no-warmup \
  --no-display-prompt \
  --no-conversation \
  --single-turn \
  --simple-io
```

Metric parsed from output:

```text
common_perf_print:        eval time = ... / 63 runs (..., XX.XX tokens per second)
```

### Speculative Decoding Throughput

```sh
build/bin/llama-speculative \
  -m /data/data/com.termux/files/home/models/Qwen3-4B-Q4_K_S.gguf \
  -md /data/data/com.termux/files/home/models/Qwen3-0.6B-Q4_K_S.gguf \
  -n 64 \
  -c 2048 \
  -b 2048 \
  --seed 1234 \
  -t 8 \
  -tb 8 \
  --temp 0.0 \
  -f <prompt-file>
```

Metrics parsed from output:

```text
decoded   NN tokens in   SS.SSS seconds, speed: XX.XXX t/s
n_predict = NN
n_drafted = NN
n_accept  = NN
accept    = XX.XXX%
```

## Example Result

Example: `sample_id=0`

```json
{
  "sample_id": 0,
  "article_chars": 1000,
  "reference_chars": 233,
  "draft_decode_tps": 52.42,
  "target_decode_tps": 17.15,
  "spec_decode_tps": 6.502,
  "speedup_vs_target": 0.379,
  "spec_accept_pct": 17.279,
  "spec_n_predict": 65,
  "spec_n_drafted": 272,
  "spec_n_accept": 47
}
```

Raw parsed record:

```json
{"sample_id":0,"article_chars":1000,"reference_chars":233,"draft":{"wall_seconds":2.829496977996314,"eval_ms":1201.92,"eval_runs":63,"eval_tps":52.42,"prompt_eval_ms":660.23,"prompt_tokens":213,"prompt_eval_tps":322.62,"decode_tps":52.42,"returncode":0},"target":{"wall_seconds":9.416817235993221,"eval_ms":3673.85,"eval_runs":63,"eval_tps":17.15,"prompt_eval_ms":3087.37,"prompt_tokens":213,"prompt_eval_tps":68.99,"decode_tps":17.15,"returncode":0},"spec":{"wall_seconds":21.041271242007497,"decoded_tokens":65,"decode_seconds":9.997,"decode_tps":6.502,"prompt_eval_ms":8618.08,"prompt_tokens":502,"prompt_eval_tps":58.25,"n_predict":65,"n_drafted":272,"n_accept":47,"accept_pct":17.279,"returncode":0},"speedup_vs_target":0.3791253644314869,"spec_vs_draft_tps_ratio":0.12403662724151086}
```

## Per-Sample Results

| sample_id | draft t/s | target t/s | spec t/s | spec / target | accept % |
|---:|---:|---:|---:|---:|---:|
| 0 | 52.42 | 17.15 | 6.502 | 0.379 | 17.279 |
| 1 | 64.85 | 16.67 | 5.431 | 0.326 | 13.750 |
| 2 | 63.78 | 16.38 | 6.277 | 0.383 | 17.279 |
| 3 | 66.24 | 14.35 | 5.735 | 0.400 | 15.000 |
| 4 | 67.87 | 17.17 | 5.565 | 0.324 | 14.375 |
| 5 | 65.27 | 16.52 | 4.907 | 0.297 | 12.500 |
| 6 | 46.82 | 16.73 | 4.877 | 0.292 | 15.972 |
| 7 | 61.82 | 17.25 | 8.105 | 0.470 | 23.661 |

## Summary

```json
{
  "samples_total": 8,
  "samples_ok": 8,
  "sample_ids": [0, 1, 2, 3, 4, 5, 6, 7],
  "draft_decode_tps_mean": 61.13375,
  "target_decode_tps_mean": 16.5275,
  "spec_decode_tps_mean": 5.924875,
  "speedup_vs_target_mean": 0.3587870067214447,
  "spec_accept_pct_mean": 16.227,
  "spec_n_predict_mean": 66.75,
  "spec_n_drafted_mean": 298.0,
  "spec_n_accept_mean": 47.125
}
```

Interpreted results:

- Draft-only throughput: `61.13 t/s`
- Target-only throughput: `16.53 t/s`
- Speculative decoding throughput: `5.92 t/s`
- Speculative speedup over target: `0.36x`
- Draft-token acceptance rate: `16.23%`

Under this CNN/DailyMail setup and model pair, speculative decoding is slower than target-only decoding. The main observed cause is low acceptance rate combined with the extra draft and verification overhead.
