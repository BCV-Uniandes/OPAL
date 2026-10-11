#!/usr/bin/env bash
set -euo pipefail

GPUS="0,1"
NUM_WORKERS=40
NUM_GPUS=2
FAST_DEV_RUN=False
DEVICE="cuda"

DATASET="carbon_24"

EPOCHS=1000
BATCH=64
LR=1e-3
SCHEDULER_FACTOR=0.6
SCHEDULER_PATIENCE=30
ALPHA=10
BETA=1
MAX_RADIUS=5.0
MAX_NEIGHBORS=20
NUM_LAYERS=6
USE_PBC=True
LOSS_TYPE="manual_w"
WEIGHT_DECAY=0.0
SCHEDULER_MONITOR="val_normalized_loss"
CKPT_MONITOR="val_normalized_loss"
JUST_SAMPLE=False

WANDB_ENTITY="BIOS-BCV2.0"
WANDB_MODE="online"
SEED=42

CKPT_NAME="opalcsp_carbon24"

flag_bool () {
  local var_value="$1"; shift
  local flag="$1"
  if [[ "${var_value}" == "True" || "${var_value}" == "true" || "${var_value}" == "1" ]]; then
    echo "${flag}"
  else
    echo "--no-${flag#--}"
  fi
}

USE_PBC_FLAG=$(flag_bool "$USE_PBC" --use_pbc)
FAST_DEV_RUN_FLAG=$(flag_bool "$FAST_DEV_RUN" --fast_dev_run)
JUST_SAMPLE_FLAG=$(flag_bool "$JUST_SAMPLE" --just_sample)

CUDA_VISIBLE_DEVICES="${GPUS}" python py_files/scripts/run.py \
--dataset_name "${DATASET}" \
--num_workers ${NUM_WORKERS} \
--device "${DEVICE}" \
--epochs ${EPOCHS} \
--batch_size ${BATCH} \
--lr ${LR} \
--scheduler_factor ${SCHEDULER_FACTOR} \
--scheduler_patience ${SCHEDULER_PATIENCE} \
--alpha ${ALPHA} \
--beta ${BETA} \
--max_radius ${MAX_RADIUS} \
--max_neighbors ${MAX_NEIGHBORS} \
--num_layers ${NUM_LAYERS} \
${USE_PBC_FLAG} \
--wandb_mode "${WANDB_MODE}" \
--num_gpus ${NUM_GPUS} \
--ckpt_name "${CKPT_NAME}" \
--wandb_entity "${WANDB_ENTITY}" \
--seed ${SEED} \
${FAST_DEV_RUN_FLAG} \
${JUST_SAMPLE_FLAG} \
--loss_type "${LOSS_TYPE}" \
--weight_decay "${WEIGHT_DECAY}" \
--scheduler_monitor "${SCHEDULER_MONITOR}" \
--ckpt_monitor "${CKPT_MONITOR}" \
"$@"
