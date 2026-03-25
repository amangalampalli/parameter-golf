#!/usr/bin/env bash
set -euo pipefail

# Compare Transformer and the default Griffin path at roughly matched
# quantized-size budgets on the full dataset with longer fixed-budget training.
#
# Examples:
#   bash compare_budgeted_backbones.sh all
#   PRESET=budget_6mb bash compare_budgeted_backbones.sh all
#   PRESET=budget_13mb bash compare_budgeted_backbones.sh all
#   PRESET=ultra_fast bash compare_budgeted_backbones.sh all
#
# Notes:
# - Presets are approximate starting points. The summary prints estimated
#   quantized MB from parameter counts so you can tighten the match.
# - `train_griffin.py` is the single/default Griffin path.
# - Runs are reused from logs by default; set FORCE_RERUN=1 to retrain.

MODE="${1:-all}"
PRESET="${PRESET:-budget_6mb}"

DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
TRAIN_PATTERN="${TRAIN_PATTERN:-${DATA_PATH}/fineweb_train_*.bin}"
VAL_PATTERN="${VAL_PATTERN:-${DATA_PATH}/fineweb_val_*.bin}"
TOKENIZER="${TOKENIZER:-./data/tokenizers/fineweb_1024_bpe.model}"
DEVICE="${DEVICE:-auto}"
STEPS="${STEPS:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
LR="${LR:-3e-4}"
DROPOUT="${DROPOUT:-0.0}"
QAT_BITS="${QAT_BITS:-6}"
FORCE_RERUN="${FORCE_RERUN:-0}"

case "${PRESET}" in
  budget_6mb)
    DEFAULT_TRAIN_BATCH_TOKENS=4096
    DEFAULT_VAL_BATCH_TOKENS=4096
    DEFAULT_BLOCK_SIZE=128
    DEFAULT_MAX_VAL_BATCHES=6
    DEFAULT_GPT_LAYERS=6
    DEFAULT_GPT_HEADS=8
    DEFAULT_GPT_EMBD=320
    DEFAULT_GRIFFIN_LAYERS=5
    DEFAULT_GRIFFIN_HEADS=8
    DEFAULT_GRIFFIN_EMBD=256
    ;;
  budget_13mb)
    DEFAULT_TRAIN_BATCH_TOKENS=4096
    DEFAULT_VAL_BATCH_TOKENS=4096
    DEFAULT_BLOCK_SIZE=128
    DEFAULT_MAX_VAL_BATCHES=6
    DEFAULT_GPT_LAYERS=8
    DEFAULT_GPT_HEADS=8
    DEFAULT_GPT_EMBD=416
    DEFAULT_GRIFFIN_LAYERS=8
    DEFAULT_GRIFFIN_HEADS=8
    DEFAULT_GRIFFIN_EMBD=320
    ;;
  fast_compare)
    DEFAULT_TRAIN_BATCH_TOKENS=8192
    DEFAULT_VAL_BATCH_TOKENS=8192
    DEFAULT_BLOCK_SIZE=128
    DEFAULT_MAX_VAL_BATCHES=8
    DEFAULT_GPT_LAYERS=4
    DEFAULT_GPT_HEADS=8
    DEFAULT_GPT_EMBD=256
    DEFAULT_GRIFFIN_LAYERS=4
    DEFAULT_GRIFFIN_HEADS=8
    DEFAULT_GRIFFIN_EMBD=256
    STEPS="${STEPS:-5000}"
    EVAL_INTERVAL="${EVAL_INTERVAL:-250}"
    LOG_INTERVAL="${LOG_INTERVAL:-25}"
    ;;
  ultra_fast)
    DEFAULT_TRAIN_BATCH_TOKENS=4096
    DEFAULT_VAL_BATCH_TOKENS=4096
    DEFAULT_BLOCK_SIZE=64
    DEFAULT_MAX_VAL_BATCHES=4
    DEFAULT_GPT_LAYERS=3
    DEFAULT_GPT_HEADS=4
    DEFAULT_GPT_EMBD=192
    DEFAULT_GRIFFIN_LAYERS=3
    DEFAULT_GRIFFIN_HEADS=4
    DEFAULT_GRIFFIN_EMBD=192
    STEPS="${STEPS:-3000}"
    EVAL_INTERVAL="${EVAL_INTERVAL:-150}"
    LOG_INTERVAL="${LOG_INTERVAL:-20}"
    ;;
  *)
    echo "Unknown PRESET=${PRESET}. Use one of: budget_6mb, budget_13mb, fast_compare, ultra_fast" >&2
    exit 1
    ;;
esac

TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-${DEFAULT_TRAIN_BATCH_TOKENS}}"
VAL_BATCH_TOKENS="${VAL_BATCH_TOKENS:-${DEFAULT_VAL_BATCH_TOKENS}}"
BLOCK_SIZE="${BLOCK_SIZE:-${DEFAULT_BLOCK_SIZE}}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-${DEFAULT_MAX_VAL_BATCHES}}"

GPT_LAYERS="${GPT_LAYERS:-${DEFAULT_GPT_LAYERS}}"
GPT_HEADS="${GPT_HEADS:-${DEFAULT_GPT_HEADS}}"
GPT_EMBD="${GPT_EMBD:-${DEFAULT_GPT_EMBD}}"
GRIFFIN_LAYERS="${GRIFFIN_LAYERS:-${DEFAULT_GRIFFIN_LAYERS}}"
GRIFFIN_HEADS="${GRIFFIN_HEADS:-${DEFAULT_GRIFFIN_HEADS}}"
GRIFFIN_EMBD="${GRIFFIN_EMBD:-${DEFAULT_GRIFFIN_EMBD}}"

RUN_TAG="${RUN_TAG:-budgeted_backbones_${PRESET}}"
GPT_RUN_ID="${GPT_RUN_ID:-gpt_${RUN_TAG}}"
GRIFFIN_RUN_ID="${GRIFFIN_RUN_ID:-griffin_${RUN_TAG}}"
GPT_CHECKPOINT="${GPT_CHECKPOINT:-checkpoints/gpt_${PRESET}_{step}.pt}"
GRIFFIN_CHECKPOINT="${GRIFFIN_CHECKPOINT:-checkpoints/griffin_${PRESET}_{step}.pt}"
GRIFFIN_LOCAL_WINDOW="${GRIFFIN_LOCAL_WINDOW:-32}"

echo "preset:${PRESET} qat_bits:${QAT_BITS} block_size:${BLOCK_SIZE} train_batch_tokens:${TRAIN_BATCH_TOKENS} val_batch_tokens:${VAL_BATCH_TOKENS} steps:${STEPS}"
echo "gpt_model:${GPT_LAYERS}x${GPT_EMBD}h${GPT_HEADS} griffin:${GRIFFIN_LAYERS}x${GRIFFIN_EMBD}h${GRIFFIN_HEADS}"

COMMON_ARGS=(
  --mode backbone
  --device "${DEVICE}"
  --train-pattern "${TRAIN_PATTERN}"
  --val-pattern "${VAL_PATTERN}"
  --tokenizer-path "${TOKENIZER}"
  --report-bpb
  --qat-bits "${QAT_BITS}"
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

should_run() {
  local log_path="$1"
  if [[ "${FORCE_RERUN}" == "1" ]]; then
    return 0
  fi
  [[ ! -f "${log_path}" ]]
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
  local model_params_line
  model_params_line="$(grep '^model_params ' "${log_path}" | tail -n 1 || true)"
  if [[ -n "${model_params_line}" ]]; then
    extract_metric "${model_params_line}" "backbone"
    return
  fi
  local trainable_line
  trainable_line="$(grep '^mode:backbone trainable_params:' "${log_path}" | tail -n 1 || true)"
  if [[ -n "${trainable_line}" ]]; then
    extract_metric "${trainable_line}" "trainable_params"
    return
  fi
  echo "n/a"
}

extract_last_loss() {
  local log_path="$1"
  if [[ ! -f "${log_path}" ]]; then
    echo "n/a"
    return
  fi
  local step_line
  step_line="$(grep '^step:' "${log_path}" | tail -n 1 || true)"
  if [[ -z "${step_line}" ]]; then
    echo "n/a"
    return
  fi
  extract_metric "${step_line}" "loss"
}

estimate_quant_mb() {
  local params="$1"
  if [[ -z "${params}" || "${params}" == "n/a" ]]; then
    echo "n/a"
    return
  fi
  awk -v params="${params}" -v bits="${QAT_BITS}" 'BEGIN { printf "%.2f", (params * bits / 8.0) / (1024.0 * 1024.0) }'
}

run_gpt() {
  local log_path="logs/${GPT_RUN_ID}.txt"
  if should_run "${log_path}"; then
    echo "== Running Transformer budgeted baseline =="
    python3 train_gpt.py \
      "${COMMON_ARGS[@]}" \
      --n-layer "${GPT_LAYERS}" \
      --n-head "${GPT_HEADS}" \
      --n-embd "${GPT_EMBD}" \
      --run-id "${GPT_RUN_ID}" \
      --checkpoint-path "${GPT_CHECKPOINT}"
  else
    echo "== Reusing Transformer run: ${log_path} =="
  fi
}

run_griffin() {
  local log_path="logs/${GRIFFIN_RUN_ID}.txt"
  if should_run "${log_path}"; then
    echo "== Running Griffin budgeted baseline =="
    python3 train_griffin.py \
      "${COMMON_ARGS[@]}" \
      --n-layer "${GRIFFIN_LAYERS}" \
      --n-head "${GRIFFIN_HEADS}" \
      --n-embd "${GRIFFIN_EMBD}" \
      --local-window "${GRIFFIN_LOCAL_WINDOW}" \
      --run-id "${GRIFFIN_RUN_ID}" \
      --checkpoint-path "${GRIFFIN_CHECKPOINT}"
  else
    echo "== Reusing Griffin run: ${log_path} =="
  fi
}

print_summary() {
  local gpt_log="logs/${GPT_RUN_ID}.txt"
  local griffin_log="logs/${GRIFFIN_RUN_ID}.txt"
  local gpt_eval griffin_eval
  local gpt_params griffin_params
  local gpt_loss griffin_loss
  local gpt_val_loss griffin_val_loss
  local gpt_val_bpb griffin_val_bpb
  local gpt_q_mb griffin_q_mb

  gpt_eval="$(last_eval_line "${gpt_log}")"
  griffin_eval="$(last_eval_line "${griffin_log}")"

  echo
  echo "== Transformer final eval =="
  echo "${gpt_eval}"
  echo
  echo "== Griffin final eval =="
  echo "${griffin_eval}"

  gpt_params="$(extract_backbone_params "${gpt_log}")"
  griffin_params="$(extract_backbone_params "${griffin_log}")"
  gpt_loss="$(extract_last_loss "${gpt_log}")"
  griffin_loss="$(extract_last_loss "${griffin_log}")"
  gpt_val_loss="$(extract_metric "${gpt_eval}" "val_loss")"
  griffin_val_loss="$(extract_metric "${griffin_eval}" "val_loss")"
  gpt_val_bpb="$(extract_metric "${gpt_eval}" "val_bpb")"
  griffin_val_bpb="$(extract_metric "${griffin_eval}" "val_bpb")"
  gpt_q_mb="$(estimate_quant_mb "${gpt_params}")"
  griffin_q_mb="$(estimate_quant_mb "${griffin_params}")"

  echo
  echo "== Side-by-side summary =="
  printf "%-16s %-14s %-12s %-12s %-12s %-12s %-40s\n" "model" "backbone_params" "est_q_mb" "train_loss" "val_loss" "val_bpb" "log"
  printf "%-16s %-14s %-12s %-12s %-12s %-12s %-40s\n" "transformer" "${gpt_params:-n/a}" "${gpt_q_mb:-n/a}" "${gpt_loss:-n/a}" "${gpt_val_loss:-n/a}" "${gpt_val_bpb:-n/a}" "${gpt_log}"
  printf "%-16s %-14s %-12s %-12s %-12s %-12s %-40s\n" "griffin" "${griffin_params:-n/a}" "${griffin_q_mb:-n/a}" "${griffin_loss:-n/a}" "${griffin_val_loss:-n/a}" "${griffin_val_bpb:-n/a}" "${griffin_log}"
}

case "${MODE}" in
  gpt)
    run_gpt
    ;;
  griffin)
    run_griffin
    ;;
  all)
    run_gpt
    run_griffin
    ;;
  *)
    echo "Usage: bash compare_budgeted_backbones.sh [all|gpt|griffin]" >&2
    exit 1
    ;;
esac

print_summary
