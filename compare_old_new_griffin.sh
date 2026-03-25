#!/usr/bin/env bash
set -euo pipefail

# Benchmark the legacy simplified Griffin once, then compare one or more
# upgraded Griffin experiments against that cached baseline.
#
# Examples:
#   bash compare_old_new_griffin.sh baseline
#   bash compare_old_new_griffin.sh experiments
#   bash compare_old_new_griffin.sh default
#
# Default experiment matrix:
#   rglru|--recurrent-variant rglru --mix-strategy learned
#   rglru_sum|--recurrent-variant rglru --mix-strategy sum
#   relaxed|--recurrent-variant rglru_relaxed --mix-strategy sum
#   legacy_like|--recurrent-variant legacy_like --mix-strategy sum
#
# Override defaults, for example:
#   STEPS=2000 LR=3e-4 bash compare_old_new_griffin.sh default
#   EXPERIMENT_SPECS='attn_only|--mix-strategy attention_only;rec_only|--mix-strategy recurrent_only' \
#     bash compare_old_new_griffin.sh experiments

MODE="${1:-default}"

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
OLD_GRIFFIN_LAYERS="${OLD_GRIFFIN_LAYERS:-2}"
OLD_GRIFFIN_HEADS="${OLD_GRIFFIN_HEADS:-4}"
OLD_GRIFFIN_EMBD="${OLD_GRIFFIN_EMBD:-128}"
NEW_GRIFFIN_LAYERS="${NEW_GRIFFIN_LAYERS:-2}"
NEW_GRIFFIN_HEADS="${NEW_GRIFFIN_HEADS:-4}"
NEW_GRIFFIN_EMBD="${NEW_GRIFFIN_EMBD:-128}"
DROPOUT="${DROPOUT:-0.0}"
LR="${LR:-1e-3}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-2}"
OLD_GRIFFIN_CHECKPOINT="${OLD_GRIFFIN_CHECKPOINT:-checkpoints/griffin_old_compare_{step}.pt}"
NEW_GRIFFIN_CHECKPOINT_TEMPLATE="${NEW_GRIFFIN_CHECKPOINT_TEMPLATE:-checkpoints/griffin_new_{experiment}_{step}.pt}"
OLD_GRIFFIN_LOCAL_WINDOW="${OLD_GRIFFIN_LOCAL_WINDOW:-32}"
NEW_GRIFFIN_LOCAL_WINDOW="${NEW_GRIFFIN_LOCAL_WINDOW:-32}"
NEW_GRIFFIN_BACKEND="${NEW_GRIFFIN_BACKEND:-sdpa}"
NEW_GRIFFIN_CHUNK_SIZE="${NEW_GRIFFIN_CHUNK_SIZE:-128}"
RUN_TAG="${RUN_TAG:-griffin_old_new_compare}"
OLD_GRIFFIN_RUN_ID="${OLD_GRIFFIN_RUN_ID:-griffin_old_${RUN_TAG}}"
FORCE_OLD_BASELINE="${FORCE_OLD_BASELINE:-0}"
EXPERIMENT_SPECS="${EXPERIMENT_SPECS:-rglru|--recurrent-variant rglru --mix-strategy learned;rglru_sum|--recurrent-variant rglru --mix-strategy sum;relaxed|--recurrent-variant rglru_relaxed --mix-strategy sum;legacy_like|--recurrent-variant legacy_like --mix-strategy sum}"

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
  local params_line
  params_line="$(grep '^model_params ' "${log_path}" | tail -n 1 || true)"
  if [[ -n "${params_line}" ]]; then
    extract_metric "${params_line}" "backbone"
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

run_old_griffin() {
  local old_log="logs/${OLD_GRIFFIN_RUN_ID}.txt"
  if [[ "${FORCE_OLD_BASELINE}" != "1" && -f "${old_log}" ]]; then
    echo "== Reusing legacy Griffin baseline: ${old_log} =="
    return
  fi
  echo "== Running legacy Griffin baseline =="
  python3 train_griffin_old.py \
    "${COMMON_ARGS[@]}" \
    --n-layer "${OLD_GRIFFIN_LAYERS}" \
    --n-head "${OLD_GRIFFIN_HEADS}" \
    --n-embd "${OLD_GRIFFIN_EMBD}" \
    --local-window "${OLD_GRIFFIN_LOCAL_WINDOW}" \
    --run-id "${OLD_GRIFFIN_RUN_ID}" \
    --checkpoint-path "${OLD_GRIFFIN_CHECKPOINT}"
}

run_new_experiment() {
  local exp_name="$1"
  local exp_args="$2"
  local run_id="griffin_new_${exp_name}_${RUN_TAG}"
  local checkpoint_path="${NEW_GRIFFIN_CHECKPOINT_TEMPLATE//\{experiment\}/${exp_name}}"
  echo "== Running upgraded Griffin experiment: ${exp_name} =="
  # shellcheck disable=SC2206
  local old_ifs="${IFS}"
  IFS=$' \t\n'
  local extra_args=( ${exp_args} )
  IFS="${old_ifs}"
  python3 train_griffin.py \
    "${COMMON_ARGS[@]}" \
    --n-layer "${NEW_GRIFFIN_LAYERS}" \
    --n-head "${NEW_GRIFFIN_HEADS}" \
    --n-embd "${NEW_GRIFFIN_EMBD}" \
    --local-window "${NEW_GRIFFIN_LOCAL_WINDOW}" \
    --local-attn-backend "${NEW_GRIFFIN_BACKEND}" \
    --recurrent-chunk-size "${NEW_GRIFFIN_CHUNK_SIZE}" \
    --run-id "${run_id}" \
    --checkpoint-path "${checkpoint_path}" \
    "${extra_args[@]}"
}

print_baseline_summary() {
  local old_log="logs/${OLD_GRIFFIN_RUN_ID}.txt"
  local old_eval
  old_eval="$(last_eval_line "${old_log}")"
  echo
  echo "== Legacy Griffin final eval =="
  echo "${old_eval}"
}

print_experiment_table() {
  local old_log="logs/${OLD_GRIFFIN_RUN_ID}.txt"
  local old_eval old_params old_train_loss old_val_loss old_val_bpb
  old_eval="$(last_eval_line "${old_log}")"
  old_params="$(extract_backbone_params "${old_log}")"
  old_train_loss="$(extract_last_loss "${old_log}")"
  old_val_loss="$(extract_metric "${old_eval}" "val_loss")"
  old_val_bpb="$(extract_metric "${old_eval}" "val_bpb")"

  echo
  echo "== Side-by-side summary =="
  printf "%-16s %-14s %-12s %-12s %-12s %-40s\n" "model" "backbone_params" "train_loss" "val_loss" "val_bpb" "log"
  printf "%-16s %-14s %-12s %-12s %-12s %-40s\n" "griffin_old" "${old_params:-n/a}" "${old_train_loss:-n/a}" "${old_val_loss:-n/a}" "${old_val_bpb:-n/a}" "${old_log}"

  local IFS=';'
  for spec in ${EXPERIMENT_SPECS}; do
    [[ -z "${spec}" ]] && continue
    local exp_name="${spec%%|*}"
    local run_id="griffin_new_${exp_name}_${RUN_TAG}"
    local log_path="logs/${run_id}.txt"
    local eval_line val_loss val_bpb params train_loss
    eval_line="$(last_eval_line "${log_path}")"
    val_loss="$(extract_metric "${eval_line}" "val_loss")"
    val_bpb="$(extract_metric "${eval_line}" "val_bpb")"
    params="$(extract_backbone_params "${log_path}")"
    train_loss="$(extract_last_loss "${log_path}")"
    printf "%-16s %-14s %-12s %-12s %-12s %-40s\n" "${exp_name}" "${params:-n/a}" "${train_loss:-n/a}" "${val_loss:-n/a}" "${val_bpb:-n/a}" "${log_path}"
  done
}

run_experiments() {
  local IFS=';'
  for spec in ${EXPERIMENT_SPECS}; do
    [[ -z "${spec}" ]] && continue
    local exp_name="${spec%%|*}"
    local exp_args="${spec#*|}"
    run_new_experiment "${exp_name}" "${exp_args}"
  done
}

case "${MODE}" in
  baseline)
    run_old_griffin
    print_baseline_summary
    ;;
  experiments)
    run_old_griffin
    run_experiments
    print_baseline_summary
    print_experiment_table
    ;;
  default)
    run_old_griffin
    run_experiments
    print_baseline_summary
    print_experiment_table
    ;;
  *)
    echo "Usage: bash compare_old_new_griffin.sh [baseline|experiments|default]" >&2
    exit 1
    ;;
esac
