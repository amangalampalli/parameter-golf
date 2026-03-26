#!/usr/bin/env bash
set -euo pipefail

# Long-run legacy Griffin backbone training script.
#
# Goals:
# - 50k-step default run for serious convergence testing
# - checkpoint every 10k steps
# - late QAT during the last 15% of training
# - quantized int8+zlib export at the end so we can inspect compressed size
#
# Override any setting with env vars, for example:
#   TOTAL_STEPS=60000 TRAIN_BATCH_TOKENS=65536 BLOCK_SIZE=512 bash run_griffin_long.sh

DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
TRAIN_PATTERN="${TRAIN_PATTERN:-${DATA_PATH}/fineweb_train_*.bin}"
VAL_PATTERN="${VAL_PATTERN:-${DATA_PATH}/fineweb_val_*.bin}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
DEVICE="${DEVICE:-auto}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
COMPILE_MODEL="${COMPILE_MODEL:-0}"
RUN_ID="${RUN_ID:-griffin_long_$(date +%Y%m%d_%H%M%S)}"
TOTAL_STEPS="${TOTAL_STEPS:-50000}"
QAT_TAIL_PERCENT="${QAT_TAIL_PERCENT:-15}"
MATRIX_QAT_START_PERCENT="${MATRIX_QAT_START_PERCENT:-60}"
QAT_START_STEP=$(( TOTAL_STEPS - (TOTAL_STEPS * QAT_TAIL_PERCENT / 100) ))
if (( QAT_START_STEP < 0 )); then
  QAT_START_STEP=0
fi
MATRIX_QAT_START_STEP=$(( TOTAL_STEPS * MATRIX_QAT_START_PERCENT / 100 ))
if (( MATRIX_QAT_START_STEP < 0 )); then
  MATRIX_QAT_START_STEP=0
fi
if (( MATRIX_QAT_START_STEP > QAT_START_STEP && QAT_START_STEP > 0 )); then
  MATRIX_QAT_START_STEP=${QAT_START_STEP}
fi

TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-8192}"
VAL_BATCH_TOKENS="${VAL_BATCH_TOKENS:-8192}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
N_LAYER="${N_LAYER:-18}"
N_HEAD="${N_HEAD:-12}"
N_EMBD="${N_EMBD:-768}"
LOCAL_WINDOW="${LOCAL_WINDOW:-64}"
RECURRENT_CHUNK_SIZE="${RECURRENT_CHUNK_SIZE:-64}"
RECURRENT_MULTISCALE="${RECURRENT_MULTISCALE:-1}"
RECURRENT_SLOW_DECAY_BIAS="${RECURRENT_SLOW_DECAY_BIAS:-2.5}"
MLP_MULT="${MLP_MULT:-4.0}"
MIX_STRATEGY="${MIX_STRATEGY:-recurrent_delta}"
MIX_TEMPERATURE="${MIX_TEMPERATURE:-0.8}"
RECURRENT_BRANCH_INIT="${RECURRENT_BRANCH_INIT:-1.0}"
ATTENTION_BRANCH_INIT="${ATTENTION_BRANCH_INIT:-0.35}"
LR="${LR:-3e-4}"
BACKBONE_OPTIMIZER="${BACKBONE_OPTIMIZER:-muon_adamw}"
BACKBONE_EMBED_LR_SCALE="${BACKBONE_EMBED_LR_SCALE:-0.5}"
BACKBONE_SCALAR_LR_SCALE="${BACKBONE_SCALAR_LR_SCALE:-0.25}"
MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
MUON_BACKEND_STEPS="${MUON_BACKEND_STEPS:-5}"
MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.85}"
MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-500}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-16}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
QAT_BITS="${QAT_BITS:-6}"
QAT_EMBED_BITS="${QAT_EMBED_BITS:-6}"
QAT_SENSITIVE_BITS="${QAT_SENSITIVE_BITS:-6}"
USE_SELECTIVE_QAT="${USE_SELECTIVE_QAT:-1}"
EXPORT_PROBE_EVERY="${EXPORT_PROBE_EVERY:-5000}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/${RUN_ID}_{step}.pt}"
EXPORT_PATH="${EXPORT_PATH:-exports/${RUN_ID}.int8.ptz}"
METRICS_CSV_PATH="${METRICS_CSV_PATH:-metrics/${RUN_ID}.csv}"

echo "run_id:${RUN_ID} total_steps:${TOTAL_STEPS} matrix_qat_start_step:${MATRIX_QAT_START_STEP} qat_start_step:${QAT_START_STEP} save_every:${SAVE_EVERY}"
echo "model:${N_LAYER}x${N_EMBD}h${N_HEAD} block_size:${BLOCK_SIZE} recurrent_chunk_size:${RECURRENT_CHUNK_SIZE} train_batch_tokens:${TRAIN_BATCH_TOKENS} val_batch_tokens:${VAL_BATCH_TOKENS}"
echo "mix:strategy:${MIX_STRATEGY} temperature:${MIX_TEMPERATURE} recurrent_branch_init:${RECURRENT_BRANCH_INIT} attention_branch_init:${ATTENTION_BRANCH_INIT}"
echo "recurrent:multiscale:${RECURRENT_MULTISCALE} slow_decay_bias:${RECURRENT_SLOW_DECAY_BIAS}"
echo "optimizer:${BACKBONE_OPTIMIZER} lr:${LR} embed_lr_scale:${BACKBONE_EMBED_LR_SCALE} scalar_lr_scale:${BACKBONE_SCALAR_LR_SCALE} muon_momentum:${MUON_MOMENTUM} muon_backend_steps:${MUON_BACKEND_STEPS}"
echo "distributed:nproc_per_node:${NPROC_PER_NODE} compile_model:${COMPILE_MODEL}"
echo "qat:bits:${QAT_BITS} embed_bits:${QAT_EMBED_BITS} sensitive_bits:${QAT_SENSITIVE_BITS} selective:${USE_SELECTIVE_QAT}"
echo "export_probe_every:${EXPORT_PROBE_EVERY}"
echo "checkpoint_path:${CHECKPOINT_PATH} export_path:${EXPORT_PATH} metrics_csv_path:${METRICS_CSV_PATH}"

RUNNER=(python3)
if [[ "${NPROC_PER_NODE}" != "1" ]]; then
  RUNNER=(torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}")
fi
EXTRA_ARGS=()
if [[ "${COMPILE_MODEL}" == "1" ]]; then
  EXTRA_ARGS+=(--compile-model)
fi
if [[ "${USE_SELECTIVE_QAT}" == "1" ]]; then
  EXTRA_ARGS+=(--use-selective-qat)
else
  EXTRA_ARGS+=(--no-use-selective-qat)
fi
if [[ "${RECURRENT_MULTISCALE}" == "1" ]]; then
  EXTRA_ARGS+=(--recurrent-multiscale)
else
  EXTRA_ARGS+=(--no-recurrent-multiscale)
fi

"${RUNNER[@]}" train_griffin.py \
  --mode backbone \
  --device "${DEVICE}" \
  --train-pattern "${TRAIN_PATTERN}" \
  --val-pattern "${VAL_PATTERN}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --report-bpb \
  --block-size "${BLOCK_SIZE}" \
  --n-layer "${N_LAYER}" \
  --n-head "${N_HEAD}" \
  --n-embd "${N_EMBD}" \
  --local-window "${LOCAL_WINDOW}" \
  --recurrent-chunk-size "${RECURRENT_CHUNK_SIZE}" \
  --recurrent-slow-decay-bias "${RECURRENT_SLOW_DECAY_BIAS}" \
  --mlp-mult "${MLP_MULT}" \
  --mix-strategy "${MIX_STRATEGY}" \
  --mix-temperature "${MIX_TEMPERATURE}" \
  --recurrent-branch-init "${RECURRENT_BRANCH_INIT}" \
  --attention-branch-init "${ATTENTION_BRANCH_INIT}" \
  --dropout 0.0 \
  --train-batch-tokens "${TRAIN_BATCH_TOKENS}" \
  --val-batch-tokens "${VAL_BATCH_TOKENS}" \
  --max-steps "${TOTAL_STEPS}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --eval-interval "${EVAL_INTERVAL}" \
  --log-interval "${LOG_INTERVAL}" \
  --save-every "${SAVE_EVERY}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --backbone-lr "${LR}" \
  --backbone-optimizer "${BACKBONE_OPTIMIZER}" \
  --backbone-embed-lr-scale "${BACKBONE_EMBED_LR_SCALE}" \
  --backbone-scalar-lr-scale "${BACKBONE_SCALAR_LR_SCALE}" \
  --muon-momentum "${MUON_MOMENTUM}" \
  --muon-backend-steps "${MUON_BACKEND_STEPS}" \
  --muon-momentum-warmup-start "${MUON_MOMENTUM_WARMUP_START}" \
  --muon-momentum-warmup-steps "${MUON_MOMENTUM_WARMUP_STEPS}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --legacy-export-path "${EXPORT_PATH}" \
  --metrics-csv-path "${METRICS_CSV_PATH}" \
  --export-probe-every "${EXPORT_PROBE_EVERY}" \
  --qat \
  --qat-bits "${QAT_BITS}" \
  --qat-embed-bits "${QAT_EMBED_BITS}" \
  --qat-sensitive-bits "${QAT_SENSITIVE_BITS}" \
  --matrix-qat-start-step "${MATRIX_QAT_START_STEP}" \
  --qat-start-step "${QAT_START_STEP}" \
  "${EXTRA_ARGS[@]}" \
  --run-id "${RUN_ID}"
