#!/usr/bin/env bash
set -euo pipefail

BIN="build/bin/llama-speculative-simple"
REPEAT=5
DRAFT_MAX=4
DRAFT_MAX_LIST=""
OUT="outputs/spec_diag.csv"
OUT_PREFIX=""
TARGET_MODEL=""
DRAFT_MODEL=""
PROMPT=""
PROMPT_FILE=""
N_PREDICT=""

usage() {
    cat <<'EOF'
Usage: scripts/run_spec_diagnostics.sh \
  --target-model PATH \
  --draft-model PATH \
  (--prompt TEXT | --prompt-file PATH) \
  --n-predict N \
  [--repeat N] \
  [--draft-max N] \
  [--draft-max-list 1,2,4,8,16] \
  [--out PATH] \
  [--out-prefix PREFIX]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-model)
            TARGET_MODEL="$2"
            shift 2
            ;;
        --draft-model)
            DRAFT_MODEL="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --prompt-file)
            PROMPT_FILE="$2"
            shift 2
            ;;
        --n-predict)
            N_PREDICT="$2"
            shift 2
            ;;
        --repeat)
            REPEAT="$2"
            shift 2
            ;;
        --draft-max)
            DRAFT_MAX="$2"
            shift 2
            ;;
        --draft-max-list)
            DRAFT_MAX_LIST="$2"
            shift 2
            ;;
        --out)
            OUT="$2"
            shift 2
            ;;
        --out-prefix)
            OUT_PREFIX="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$TARGET_MODEL" || -z "$DRAFT_MODEL" || -z "$N_PREDICT" ]]; then
    echo "missing required argument" >&2
    usage >&2
    exit 1
fi

if [[ -n "$PROMPT" && -n "$PROMPT_FILE" ]]; then
    echo "use only one of --prompt or --prompt-file" >&2
    exit 1
fi

if [[ -z "$PROMPT" && -z "$PROMPT_FILE" ]]; then
    echo "one of --prompt or --prompt-file is required" >&2
    exit 1
fi

if [[ ! -x "$BIN" ]]; then
    echo "missing executable: $BIN" >&2
    exit 1
fi

if [[ ! -f "$TARGET_MODEL" ]]; then
    echo "missing target model: $TARGET_MODEL" >&2
    exit 1
fi

if [[ ! -f "$DRAFT_MODEL" ]]; then
    echo "missing draft model: $DRAFT_MODEL" >&2
    exit 1
fi

if [[ -n "$PROMPT_FILE" && ! -f "$PROMPT_FILE" ]]; then
    echo "missing prompt file: $PROMPT_FILE" >&2
    exit 1
fi

OUT_DIR="$(dirname "$OUT")"
CSV_HEADER="sample_id,mode,total_decode_time_ms,draft_time_ms,target_verify_time_ms,accept_reject_time_ms,generated_tokens,drafted_tokens,accepted_tokens,acceptance_rate,decode_tok_per_s,draft_vocab_convert_time_ms,draft_prompt_sync_time_ms,draft_decode_time_ms,draft_sampling_time_ms,draft_other_time_ms,draft_returned_tokens,draft_effective_tokens,target_verify_batch_tokens,draft_prompt_decode_calls,draft_prompt_decode_tokens,draft_decode_calls,draft_decode_tokens,target_verify_calls,draft_tok_per_s_in_spec,draft_decode_tok_per_s,verify_tok_per_s,draft_reuse_scan_time_ms,draft_kv_cache_edit_time_ms,draft_prompt_catchup_decode_time_ms,draft_eager_kv_decode_time_ms,draft_eager_kv_decode_calls,draft_eager_kv_decode_tokens"
LOG_DIR=""

prompt_args=()
if [[ -n "$PROMPT_FILE" ]]; then
    prompt_args=(-f "$PROMPT_FILE")
else
    prompt_args=(-p "$PROMPT")
fi

run_one() {
    local sample_id="$1"
    local mode="$2"
    local draft_max_arg="$3"
    local log_file="$LOG_DIR/${sample_id}-${mode}.log"
    local cmd=("$BIN" -m "$TARGET_MODEL" "${prompt_args[@]}" -n "$N_PREDICT")

    case "$mode" in
        true_target_only)
            ;;
        spec_draft_max_0|speculative)
            cmd+=(-md "$DRAFT_MODEL" --draft-max "$draft_max_arg")
            ;;
        *)
            echo "internal error: unknown mode $mode" >&2
            exit 1
            ;;
    esac

    echo "running sample_id=$sample_id mode=$mode" >&2
    if ! LLAMA_SPEC_PROFILE_SAMPLE_ID="$sample_id" "${cmd[@]}" > "$log_file" 2>&1; then
        echo "run failed: sample_id=$sample_id mode=$mode" >&2
        echo "log: $log_file" >&2
        return 1
    fi

    local row
    row="$(grep -E "^[0-9]+,(true_target_only|spec_draft_max_0|speculative)," "$log_file" | tail -n 1 || true)"
    if [[ -z "$row" ]]; then
        echo "CSV row not found: sample_id=$sample_id mode=$mode" >&2
        echo "log: $log_file" >&2
        return 1
    fi

    printf '%s\n' "$row" >> "$OUT"
}

prepare_output() {
    OUT_DIR="$(dirname "$OUT")"
    mkdir -p "$OUT_DIR"

    if [[ ! -s "$OUT" ]]; then
        printf '%s\n' "$CSV_HEADER" > "$OUT"
    fi

    LOG_DIR="${OUT}.logs"
    mkdir -p "$LOG_DIR"
}

run_suite() {
    local draft_max="$1"
    local sample_id=0

    prepare_output

    for ((i = 0; i < REPEAT; ++i)); do
        run_one "$sample_id" true_target_only 0
        sample_id=$((sample_id + 1))

        run_one "$sample_id" spec_draft_max_0 0
        sample_id=$((sample_id + 1))

        run_one "$sample_id" speculative "$draft_max"
        sample_id=$((sample_id + 1))
    done

    echo "wrote $OUT" >&2
}

if [[ -n "$DRAFT_MAX_LIST" ]]; then
    if [[ -z "$OUT_PREFIX" ]]; then
        OUT_PREFIX="${OUT%.csv}"
    fi

    IFS=',' read -r -a draft_max_values <<< "$DRAFT_MAX_LIST"
    for draft_max in "${draft_max_values[@]}"; do
        if [[ -z "$draft_max" ]]; then
            echo "empty value in --draft-max-list" >&2
            exit 1
        fi
        OUT="${OUT_PREFIX}_g${draft_max}.csv"
        run_suite "$draft_max"
    done
else
    mkdir -p "$OUT_DIR"
    run_suite "$DRAFT_MAX"
fi
