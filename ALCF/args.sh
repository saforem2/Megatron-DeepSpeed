#!/bin/bash --login

USER=$(whoami)
HOST=$(hostname)

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
PARENT=$(dirname ${DIR})

DDP_IMPL="FSDP"
USE_FLASH_ATTN=1

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Model / Architecture settings                       ┃
# ┃ --------------------------------------------------- ┃
# ┃ GPT-3 models use 2K sequence length/context window  ┃
# ┃ The "GPT-3 XXX" below are configs from GPT-3 paper  ┃
# ┃ https://arxiv.org/abs/2005.14165, choose based on   ┃
# ┃ your desired model size or build your own configs   ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
SEQ_LEN=2048

# ┏━━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 2.7B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="2.7B"
# NLAYERS=32
# HIDDEN=2560
# ATEN_HEADS=32
# GLOBAL_BATCH=512

# ┏━━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 6.7B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━┛
MODEL_SIZE="6.7B"
NLAYERS=32
HIDDEN=4096
ATEN_HEADS=32
GLOBAL_BATCH=1024

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Model Parallel / Pipeline Parallel ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ----------
# Originals
# MPSIZE=8
# PPSIZE=16
# ----------
MPSIZE=1
PPSIZE=1
MICRO_BATCH=1
ZERO_STAGE=1

# ┏━━━━━━━━━━━━┓
# ┃ Data paths ┃
# ┗━━━━━━━━━━━━┛
DATA_PATH="${PARENT}/dataset/BookCorpusDataset_text_document"
VOCAB_FILE="${PARENT}/dataset/gpt2-vocab.json"
MERGE_FILE="${PARENT}/dataset/gpt2-merges.txt"

# ┏━━━━━━━━━━━━━━━━━━━┓
# ┃ FILE I/O SETTINGS ┃
# ┗━━━━━━━━━━━━━━━━━━━┛
RUN_STR="gb${GLOBAL_BATCH}_mb${MICRO_BATCH}"
RUN_STR="nl${NLAYERS}_hs${HIDDEN}_${RUN_STR}"
RUN_STR="mp${MPSIZE}_pp${PPSIZE}_${RUN_STR}"
RUN_STR="z${ZERO_STAGE}_seqlen${SEQ_LEN}_${RUN_STR}"
RUN_STR="${MODEL_SIZE}_${RUN_STR}"

if [[ $USE_FLASH_ATTN == 1 ]] ; then
  RUN_STR="flashAttn_${RUN_STR}"
fi
if [[ $DDP_IMPL == 'FSDP' ]]; then
  RUN_STR="FSDP_${RUN_STR}"
fi

RUN_STR="GPT3_${RUN_STR}"
# else
#   RUN_STR="${RUN_STR}"
# fi

OUTPUT_DIR="${PARENT}/outputs/${RUN_STR}"
CHECKPOINT_DIR="${PARENT}/checkpoints/$RUN_STR"
TENSORBOARD_DIR="${PARENT}/outputs/${RUN_STR}/tensorboard"

export TENSORBOARD_DIR=$TENSORBOARD_DIR
export OUTPUT_DIR=$OUTPUT_DIR
mkdir -p "$OUTPUT_DIR/tensorboard/wandb"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$TENSORBOARD_DIR"
mkdir -p "${OUTPUT_DIR}"
echo "OUTPUT TO: ${OUTPUT_DIR}"

# if [[ -z "${NVME_PATH}" ]]; then
#   echo "NVME_PATH: $NVME_PATH"
# else
#   if [[ $(hostname) == x* ]]; then
#     export NVME_PATH="/local/scratch/"
#   elif [[ $(hostname) == theta* ]]; then
#     export NVME_PATH="/raid/scratch/"
#   else
#     export NVME_PATH="/tmp/"
#   fi
# fi

# echo "NVME_PATH: ${NVME_PATH}"

# ┏━━━━━━━━━━━━━━━━━━┓
# ┃ DeepSpeed Config ┃
# ┗━━━━━━━━━━━━━━━━━━┛
DS_CONFIG=${PARENT}/ds_config-gpt.json
cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "allgather_partitions": true,
    "reduce_scatter": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "offload_optimizer": {
      "device": "nvme",
      "nvme_path": "/raid/scratch/"
    }
  },
  "fp16": {
    "enabled": true,
    "initial_scale_power": 12
  },
  "wall_clock_breakdown" : true,
  "flops_profiler": {
    "enabled": true,
    "profile_step": -1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": null
  },
  "wandb": {
    "enabled": true,
    "project": "megatron-LM"
  }
}
EOT

# ┏━━━━━━━━━━━━━━━━━━━━━┓
# ┃ DeepSpeed Arguments ┃
# ┗━━━━━━━━━━━━━━━━━━━━━┛
ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_mpi ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
# ds_args=" --pipeline-model-parallel-size ${PPSIZE} ${ds_args}"

if [[ "${PPSIZE}" == 1 ]]; then
  ds_args="${ds_args} --no-pipeline-parallel"
fi

# ┏━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ MEGATRON-LM SETTINGS ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━┛
export gpt_args="\
  --pipeline-model-parallel-size $PPSIZE \
  --tensor-model-parallel-size $MPSIZE \
  --num-layers $NLAYERS \
  --hidden-size $HIDDEN \
  --num-attention-heads ${ATEN_HEADS} \
  --micro-batch-size ${MICRO_BATCH} \
  --global-batch-size ${GLOBAL_BATCH} \
  --seq-length ${SEQ_LEN} \
  --max-position-embeddings ${SEQ_LEN} \
  --train-iters 5 \
  --lr-decay-iters 320000 \
  --data-path $DATA_PATH \
  --vocab-file $VOCAB_FILE \
  --merge-file $MERGE_FILE \
  --data-impl mmap \
  --split 949,50,1 \
  --distributed-backend nccl \
  --lr 0.00015 \
  --lr-decay-style cosine \
  --min-lr 1.0e-5 \
  --weight-decay 1e-2 \
  --clip-grad 1.0 \
  --lr-warmup-fraction .01 \
  --checkpoint-activations \
  --log-interval 1 \
  --save-interval 1000 \
  --eval-interval 1000 \
  --eval-iters 1 \
  --tensorboard-dir ${TENSORBOARD_DIR} \
  --log-timers-to-tensorboard \
  --tensorboard-log-interval 1 \
  --fp16"

if [[ "$USE_FLASH_ATTN" == 1 ]] ; then
  gpt_args="\
    --use-flash-attn \
    ${gpt_args}"
fi

if [[ "$DDP_IMPL" == "FSDP" ]] ; then
  gpt_args="\
    --DDP-impl FSDP \
    ${gpt_args}"
fi
