#!/usr/bin/env bash
set -euo pipefail

# Compare Transformer and Griffin on the exact same single-train-shard setup,
# evaluated on a different shard by default.
#
# Examples:
#   bash compare_same_shard_overfit.sh gpt
#   bash compare_same_shard_overfit.sh griffin
#   bash compare_same_shard_overfit.sh both
#
# Override any default with environment variables, for example:
#   STEPS=2000 LR=3e-4 bash compare_same_shard_overfit.sh gpt
# Logs are written to logs/<run_id>.txt by each trainer; this wrapper also
# prints the final eval line and a compact summary.

MODE="${1:-both}"

TRAIN_DATASET="${TRAIN_DATASET:-./data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin}"
VAL_DATASET="${VAL_DATASET:-./data/datasets/fineweb10B_sp1024/fineweb_train_000001.bin}"
TOKENIZER="${TOKENIZER:-./data/tokenizers/fineweb_1024_bpe.model}"
DEVICE="${DEVICE:-auto}"
STEPS="${STEPS:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-4096}"
VAL_BATCH_TOKENS="${VAL_BATCH_TOKENS:-4096}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
GPT_LAYERS="${GPT_LAYERS:-2}"
GPT_HEADS="${GPT_HEADS:-4}"
GPT_EMBD="${GPT_EMBD:-128}"
GRIFFIN_LAYERS="${GRIFFIN_LAYERS:-2}"
GRIFFIN_HEADS="${GRIFFIN_HEADS:-4}"
GRIFFIN_EMBD="${GRIFFIN_EMBD:-104}"
DROPOUT="${DROPOUT:-0.0}"
LR="${LR:-1e-3}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-2}"
GPT_CHECKPOINT="${GPT_CHECKPOINT:-checkpoints/gpt_overfit_{step}.pt}"
GRIFFIN_CHECKPOINT="${GRIFFIN_CHECKPOINT:-checkpoints/griffin_overfit_{step}.pt}"
GRIFFIN_LOCAL_WINDOW="${GRIFFIN_LOCAL_WINDOW:-32}"
RUN_TAG="${RUN_TAG:-same_shard_overfit}"
GPT_RUN_ID="${GPT_RUN_ID:-gpt_${RUN_TAG}}"
GRIFFIN_RUN_ID="${GRIFFIN_RUN_ID:-griffin_${RUN_TAG}}"

COMMON_ARGS=(
  --mode backbone
  --device "${DEVICE}"
  --train-pattern "${TRAIN_DATASET}"
  --val-pattern "${VAL_DATASET}"
  --tokenizer-path "${TOKENIZER}"
  --report-bpb
  --block-size "${BLOCK_SIZE}"
  --dropout "${DROPOUT}"
  --train-batch-tokens "${TRAIN_BATCH_TOKENS}"
  --val-batch-tokens "${VAL_BATCH_TOKENS}"
  --max-steps "${STEPS}"
  --eval-interval "${EVAL_INTERVAL}"
  --log-interval "${LOG_INTERVAL}"
  --save-every 0
  --max-val-batches "${MAX_VAL_BATCHES}"
  --backbone-lr "${LR}"
)

run_gpt() {
  echo "== Running Transformer overfit baseline =="
  python3 train_gpt.py \
    "${COMMON_ARGS[@]}" \
    --n-layer "${GPT_LAYERS}" \
    --n-head "${GPT_HEADS}" \
    --n-embd "${GPT_EMBD}" \
    --run-id "${GPT_RUN_ID}" \
    --checkpoint-path "${GPT_CHECKPOINT}"
}

run_griffin() {
  echo "== Running Griffin overfit baseline =="
  python3 train_griffin.py \
    "${COMMON_ARGS[@]}" \
    --n-layer "${GRIFFIN_LAYERS}" \
    --n-head "${GRIFFIN_HEADS}" \
    --n-embd "${GRIFFIN_EMBD}" \
    --run-id "${GRIFFIN_RUN_ID}" \
    --local-window "${GRIFFIN_LOCAL_WINDOW}" \
    --checkpoint-path "${GRIFFIN_CHECKPOINT}"
}

last_eval_line() {
  local log_path="$1"
  if [[ ! -f "${log_path}" ]]; then
    echo "missing_log:${log_path}"
    return
  fi
  grep '^eval ' "${log_path}" | tail -n 1 || true
}

extract_metric() {
  local line="$1"
  local key="$2"
  awk -v key="${key}" '
    {
      for (i = 1; i <= NF; i++) {
        split($i, a, ":")
        if (a[1] == key) {
          print a[2]
          exit
        }
      }
    }
  ' <<< "${line}"
}

extract_backbone_params() {
  local log_path="$1"
  if [[ ! -f "${log_path}" ]]; then
    echo "n/a"
    return
  fi
  local griffin_line
  griffin_line="$(grep '^model_params ' "${log_path}" | tail -n 1 || true)"
  if [[ -n "${griffin_line}" ]]; then
    extract_metric "${griffin_line}" "backbone"
    return
  fi
  local trainable_line
  trainable_line="$(grep '^mode:backbone trainable_params:' "${log_path}" | tail -n 1 || true)"
  if [[ -n "${trainable_line}" ]]; then
    awk '
      {
        for (i = 1; i <= NF; i++) {
          split($i, a, ":")
          if (a[1] == "trainable_params") {
            print a[2]
            exit
          }
        }
      }
    ' <<< "${trainable_line}"
    return
  fi
  echo "n/a"
}

print_summary() {
  local gpt_log="logs/${GPT_RUN_ID}.txt"
  local griffin_log="logs/${GRIFFIN_RUN_ID}.txt"
  local gpt_eval=""
  local griffin_eval=""

  if [[ "${MODE}" == "gpt" || "${MODE}" == "both" ]]; then
    gpt_eval="$(last_eval_line "${gpt_log}")"
    echo
    echo "== Transformer final eval =="
    echo "${gpt_eval}"
  fi

  if [[ "${MODE}" == "griffin" || "${MODE}" == "both" ]]; then
    griffin_eval="$(last_eval_line "${griffin_log}")"
    echo
    echo "== Griffin final eval =="
    echo "${griffin_eval}"
  fi

  if [[ "${MODE}" == "both" ]]; then
    local gpt_val_loss griffin_val_loss gpt_val_bpb griffin_val_bpb gpt_params griffin_params
    gpt_val_loss="$(extract_metric "${gpt_eval}" "val_loss")"
    griffin_val_loss="$(extract_metric "${griffin_eval}" "val_loss")"
    gpt_val_bpb="$(extract_metric "${gpt_eval}" "val_bpb")"
    griffin_val_bpb="$(extract_metric "${griffin_eval}" "val_bpb")"
    gpt_params="$(extract_backbone_params "${gpt_log}")"
    griffin_params="$(extract_backbone_params "${griffin_log}")"

    echo
    echo "== Side-by-side summary =="
    printf "%-12s %-14s %-12s %-12s %-24s\n" "model" "backbone_params" "val_loss" "val_bpb" "log"
    printf "%-12s %-14s %-12s %-12s %-24s\n" "gpt" "${gpt_params:-n/a}" "${gpt_val_loss:-n/a}" "${gpt_val_bpb:-n/a}" "${gpt_log}"
    printf "%-12s %-14s %-12s %-12s %-24s\n" "griffin" "${griffin_params:-n/a}" "${griffin_val_loss:-n/a}" "${griffin_val_bpb:-n/a}" "${griffin_log}"
  fi
}

case "${MODE}" in
  gpt)
    run_gpt
    ;;
  griffin)
    run_griffin
    ;;
  both)
    run_gpt
    run_griffin
    ;;
  *)
    echo "Usage: bash compare_same_shard_overfit.sh [gpt|griffin|both]" >&2
    exit 1
    ;;
esac

print_summary
