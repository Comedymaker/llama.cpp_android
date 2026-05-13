# CNN/DailyMail Qwen3 Phone Experiment Commands

This document records the exact command forms used for the phone-side
CNN/DailyMail raw-prompt experiment. The older
`bench-cnndm-speculative-report.md` belongs to a different experiment
(`Qwen3-4B-Q4_K_S`, 1000 article chars, 64 generated tokens, and mixed
`llama-completion` / speculative binaries), so it should not be used as the
command record for this run.

## Shared Setup

Dataset:

```text
datasets/cnn_dailymail_3.0.0_test_128.jsonl
```

Prompt template:

```text
Summarize the following CNN/DailyMail news article in 2-3 concise sentences.
Return only the summary. Do not include reasoning.

Article:
{article}

Summary:
```

The `article` field is truncated to the first 3500 characters before prompt
formatting.

Models:

```text
Draft:  /data/data/com.termux/files/home/models/Qwen3-0.6B-Q4_K_M.gguf
Target: /data/data/com.termux/files/home/models/Qwen3-8B-Q4_K_M.gguf
```

Common generation settings:

```text
n-predict: 160
ctx-size: 4096
ctx-size-draft: 4096
batch-size: 2048
temp: 0.0
seed: 1234
gpu-layers: auto
draft-gpu-layers: auto
backend: CPU only on this phone (`llama-speculative-simple --list-devices`
         printed only `Available devices:`)
```

The wrapper script used for all runs is:

```sh
python3 scripts/run-cnndm-qwen3-phone.py
```

It writes prompt text to a temporary prompt file and calls
`build/bin/llama-speculative-simple` for every sample.

## Full Run Commands

Run all three experiment groups from scratch:

```sh
python3 scripts/run-cnndm-qwen3-phone.py --logs-dir run-results/logs
```

Run or resume only one group:

```sh
python3 scripts/run-cnndm-qwen3-phone.py --run draft-only --resume --logs-dir run-results/logs
python3 scripts/run-cnndm-qwen3-phone.py --run target-only --resume --logs-dir run-results/logs
python3 scripts/run-cnndm-qwen3-phone.py --run speculative --resume --logs-dir run-results/logs
```

The `--resume` mode appends only samples whose `sample_index` is not already
present as a successful row in that group's predictions JSONL.

## Per-Sample Command Forms

`<prompt-file>` below is the temporary file containing the raw prompt shown in
the shared setup section.

### A. draft-only 0.6B

This uses `llama-speculative-simple` with draft disabled:

```sh
build/bin/llama-speculative-simple \
  -m /data/data/com.termux/files/home/models/Qwen3-0.6B-Q4_K_M.gguf \
  -md /data/data/com.termux/files/home/models/Qwen3-0.6B-Q4_K_M.gguf \
  -f <prompt-file> \
  -n 160 \
  -c 4096 \
  -cd 4096 \
  -b 2048 \
  --temp 0.0 \
  --seed 1234 \
  --draft-max 0 \
  -ngl auto \
  -ngld auto
```

Outputs:

```text
run-results/cnndm-qwen3-0.6b-raw-specsimple-phone-predictions.jsonl
run-results/cnndm-qwen3-0.6b-raw-specsimple-phone-summary.json
run-results/logs/draft-only/
```

### B. target-only 8B

This also uses `llama-speculative-simple` with draft disabled, and still loads
the draft model to keep the control path consistent with the experiment
definition:

```sh
build/bin/llama-speculative-simple \
  -m /data/data/com.termux/files/home/models/Qwen3-8B-Q4_K_M.gguf \
  -md /data/data/com.termux/files/home/models/Qwen3-0.6B-Q4_K_M.gguf \
  -f <prompt-file> \
  -n 160 \
  -c 4096 \
  -cd 4096 \
  -b 2048 \
  --temp 0.0 \
  --seed 1234 \
  --draft-max 0 \
  -ngl auto \
  -ngld auto
```

Outputs:

```text
run-results/cnndm-qwen3-8b-raw-specsimple-phone-predictions.jsonl
run-results/cnndm-qwen3-8b-raw-specsimple-phone-summary.json
run-results/logs/target-only/
```

### C. 8B+0.6B speculative

This uses `llama-speculative-simple` with `draft-max=16`:

```sh
build/bin/llama-speculative-simple \
  -m /data/data/com.termux/files/home/models/Qwen3-8B-Q4_K_M.gguf \
  -md /data/data/com.termux/files/home/models/Qwen3-0.6B-Q4_K_M.gguf \
  -f <prompt-file> \
  -n 160 \
  -c 4096 \
  -cd 4096 \
  -b 2048 \
  --temp 0.0 \
  --seed 1234 \
  --draft-max 16 \
  -ngl auto \
  -ngld auto
```

Outputs:

```text
run-results/cnndm-qwen3-8b-0.6b-specsimple-phone-predictions.jsonl
run-results/logs/speculative/
```

The speculative run was intentionally stopped after enough rows were available
for the phone-side analysis. The reported speculative summary used only the
first 29 rows from:

```text
run-results/cnndm-qwen3-8b-0.6b-specsimple-phone-predictions.jsonl
```

## Prediction Cleaning

The wrapper captures stdout and stderr separately. Predictions are cleaned by
removing the complete raw prompt prefix once if it is echoed in stdout, then
removing llama.cpp log/banner/perf lines and common special end tokens. It does
not use `rsplit("Summary:")`, so model-generated `Summary:` or `Article:` text
inside the answer is not used as a truncation point.

