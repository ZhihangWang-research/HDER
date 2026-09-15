#!/usr/bin/env bash
set -euo pipefail

# HDER launcher.
# Usage:
#   bash train.sh full
#   bash train.sh no_hae
#   bash train.sh no_cross_layer
#   bash train.sh no_srr
#   bash train.sh no_position
#   bash train.sh fixed_mean
#   bash train.sh bert_backbone
#
# Re-DocRED example:
#   DATA_DIR=./dataset/redocred TRAIN_FILE=train_revised.json DEV_FILE=dev_revised.json bash train.sh full
#
# Benchmark datasets are not redistributed. Place DocRED/Re-DocRED JSON files
# under DATA_DIR and rel2id.json under META_DIR before running.

VARIANT="${1:-full}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

DATA_DIR="${DATA_DIR:-./dataset/docred}"
META_DIR="${META_DIR:-./dataset/meta}"
SAVE_ROOT="${SAVE_ROOT:-./output}"
TRAIN_FILE="${TRAIN_FILE:-train_annotated.json}"
DEV_FILE="${DEV_FILE:-dev.json}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-roberta-large}"
TRANSFORMER_TYPE="${TRANSFORMER_TYPE:-roberta}"
GLOBAL_MEAN_PATH="${GLOBAL_MEAN_PATH:-./artifacts/fixed_global_mean.pt}"
AUX_GRAPH_PATH="${AUX_GRAPH_PATH:-./auxiliary_graph_data/graph_statistics.json}"

mkdir -p "$DATA_DIR" "$META_DIR" "$SAVE_ROOT"

if [[ ! -f "$DATA_DIR/$TRAIN_FILE" || ! -f "$DATA_DIR/$DEV_FILE" || ! -f "$META_DIR/rel2id.json" ]]; then
  echo "[error] Missing dataset files. Required:"
  echo "  $DATA_DIR/$TRAIN_FILE"
  echo "  $DATA_DIR/$DEV_FILE"
  echo "  $META_DIR/rel2id.json"
  exit 1
fi

USE_HAE=true
USE_CROSS=true
USE_POS=true
USE_SRR=true
GLOBAL_MODE=dynamic
EXPERIMENT_NAME="hder_full"
case "$VARIANT" in
  full) ;;
  no_hae) USE_HAE=false; USE_CROSS=false; USE_POS=false; EXPERIMENT_NAME="no_hae" ;;
  no_cross_layer) USE_CROSS=false; EXPERIMENT_NAME="no_cross_layer" ;;
  no_srr) USE_SRR=false; EXPERIMENT_NAME="no_srr" ;;
  no_position) USE_POS=false; EXPERIMENT_NAME="no_position" ;;
  fixed_mean) TRANSFORMER_TYPE=roberta; MODEL_NAME_OR_PATH="${FIXED_MEAN_MODEL_NAME_OR_PATH:-roberta-large}"; GLOBAL_MODE=fixed_mean; EXPERIMENT_NAME="fixed_global_mean" ;;
  bert_backbone) TRANSFORMER_TYPE=bert; MODEL_NAME_OR_PATH="${BERT_MODEL_NAME_OR_PATH:-bert-large-cased}"; EXPERIMENT_NAME="bert_backbone" ;;
  *) echo "[error] Unknown variant: $VARIANT"; exit 2 ;;
esac

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  python -m pip install -q -r requirements.txt
fi

if [[ "$VARIANT" == "fixed_mean" ]]; then
  if [[ "$TRAIN_FILE" != "train_annotated.json" ]]; then
    echo "[error] fixed_mean ablation is defined for DocRED train_annotated.json."
    exit 3
  fi
  mkdir -p "$(dirname "$GLOBAL_MEAN_PATH")"
  if [[ ! -f "$GLOBAL_MEAN_PATH" ]]; then
    echo "[info] Generating fixed global mean: $GLOBAL_MEAN_PATH"
    python compute_global_mean.py \
      --data_dir "$DATA_DIR" \
      --train_file "$TRAIN_FILE" \
      --transformer_type "$TRANSFORMER_TYPE" \
      --model_name_or_path "$MODEL_NAME_OR_PATH" \
      --max_seq_length "${MAX_SEQ_LENGTH:-1024}" \
      --batch_size "${MEAN_BATCH_SIZE:-1}" \
      --output "$GLOBAL_MEAN_PATH"
  fi
  python validate_fixed_mean.py --path "$GLOBAL_MEAN_PATH" --hidden_size 1024 --max_seq_length "${MAX_SEQ_LENGTH:-1024}"
fi

ARGS=(
  --do_train
  --experiment_name "$EXPERIMENT_NAME"
  --run_mode formal
  --data_dir "$DATA_DIR"
  --transformer_type "$TRANSFORMER_TYPE"
  --model_name_or_path "$MODEL_NAME_OR_PATH"
  --train_file "$TRAIN_FILE"
  --dev_file "$DEV_FILE"
  --save_path "$SAVE_ROOT"
  --train_batch_size "${TRAIN_BATCH_SIZE:-8}"
  --test_batch_size "${TEST_BATCH_SIZE:-8}"
  --gradient_accumulation_steps "${GRAD_ACCUM:-1}"
  --num_train_epochs "${NUM_EPOCHS:-20}"
  --lr_transformer "${LR_TRANSFORMER:-3e-5}"
  --lr_added "${LR_ADDED:-1e-4}"
  --evi_lambda "${EVI_LAMBDA:-0.03}"
  --warmup_ratio "${WARMUP_RATIO:-0.06}"
  --max_seq_length "${MAX_SEQ_LENGTH:-1024}"
  --max_sent_num "${MAX_SENT_NUM:-25}"
  --num_class 97
  --seed "${SEED:-66}"
  --use_hae "$USE_HAE"
  --use_cross_layer_attention "$USE_CROSS"
  --use_sentence_position_encoding "$USE_POS"
  --global_context_mode "$GLOBAL_MODE"
  --global_mean_path "$GLOBAL_MEAN_PATH"
  --use_srr "$USE_SRR"
  --srr_steps "${SRR_STEPS:-2}"
  --srr_token_attention_scale "${SRR_TOKEN_ATTENTION_SCALE:-0.05}"
  --srr_init learned
  --use_span_boundary_head "${USE_SPAN_BOUNDARY_HEAD:-true}"
  --span_aux_weight "${SPAN_AUX_WEIGHT:-0.01}"
  --span_detach_backbone "${SPAN_DETACH_BACKBONE:-true}"
  --aux_graph_path "$AUX_GRAPH_PATH"
  --entity_type_conflict_policy "${ENTITY_TYPE_CONFLICT_POLICY:-majority_first}"
  --paragraph_strategy "${PARAGRAPH_STRATEGY:-sentence_window}"
  --sentences_per_paragraph "${SENTENCES_PER_PARAGRAPH:-4}"
)

if [[ -n "${MAX_TRAIN_BATCHES:-}" ]]; then
  ARGS+=(--max_train_batches "$MAX_TRAIN_BATCHES")
fi

python run.py "${ARGS[@]}"
