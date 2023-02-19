#!/bin/bash -l

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)

echo "Job started at: ${TSTAMP} on $(hostname)"
echo "Job running in: ${DIR}"

export NCCL_DEBUG=warn
export WANDB_CACHE_DIR="./cache/wandb"

# ┏━━━━━━━━━━┓
# ┃ ThetaGPU ┃
# ┗━━━━━━━━━━┛
setupThetaGPU() {
  if [[ $(hostname) == theta* ]]; then
    export MACHINE="ThetaGPU"
    HOSTFILE="${COBALT_NODEFILE}"

    # -- Python / Conda setup -------------------------------------------------
    module load conda/2022-07-01 ; conda activate base
    conda activate \
      /lus/grand/projects/datascience/foremans/locations/thetaGPU/miniconda3/envs/2022-07-01
    source ./venvs/thetaGPU/2022-07-01-deepspeed/bin/activate
    # -------------------------------------------------------------------------

    # -- MPI / Comms Setup ----------------------------------
    NRANKS=$(wc -l < "${HOSTFILE}")
    NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
    NGPUS=$((${NRANKS}*${NGPU_PER_RANK}))
    MPI_COMMAND=$(which mpirun)
    MPI_DEFAULTS="\
      --hostfile ${HOSTFILE} \
      -x CFLAGS \
      -x LDFLAGS \
      -x PYTHONUSERBASE \
      -x http_proxy \
      -x https_proxy \
      -x PATH \
      -x LD_LIBRARY_PATH"
    MPI_ELASTIC="\
      -n ${NGPUS} \
      -npernode ${NGPU_PER_RANK}"
    # -------------------------------------------------------
  else
    echo "Unexpected hostname: $(hostname)"
  fi
}

# ┏━━━━━━━━━┓
# ┃ Polaris ┃
# ┗━━━━━━━━━┛
setupPolaris()  {
  if [[ $(hostname) == x* ]]; then
    export MACHINE="Polaris"
    HOSTFILE="${PBS_NODEFILE}"
    module load conda/2023-01-10
    conda activate base
    conda activate \
      /lus/grand/projects/datascience/foremans/locations/polaris/miniconda3/envs/2023-01-10

    # export IBV_FORK_SAFE=1
    # -----------------------------------------------
    NRANKS=$(wc -l < "${HOSTFILE}")
    NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
    NGPUS=$((${NRANKS}*${NGPU_PER_RANK}))
    MPI_COMMAND=$(which mpiexec)
    # -----------------------------------------------
    MPI_DEFAULTS="\
      --envall \
      --verbose \
      --hostfile ${HOSTFILE}"
    MPI_ELASTIC="\
      -n ${NGPUS} \
      --ppn ${NGPU_PER_RANK}"
  else
    echo "Unexpected hostname: $(hostname)"
  fi
}


export CFLAGS="-I${CONDA_PREFIX}/include/"
export LDFLAGS="-L${CONDA_PREFIX}/lib/"

TP=8
PP=16
NLAYERS=24
HIDDEN=1024
GLOBAL_BATCH=64
MICRO_BATCH=4
ZERO_STAGE=1
RUN_STR=ds_z${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}
DATA_PATH="./dataset/BookCorpusDataset_text_document"
VOCAB_FILE="./dataset/gpt2-vocab.json"
MERGE_FILE="./dataset/gpt2-merges.txt"
OUTPUT_DIR="outputs/${RUN_STR}"
CHECKPOINT_DIR="./checkpoints/$RUN_STR"
TENSORBOARD_DIR="./outputs/${RUN_STR}/tensorboard"
export TENSORBOARD_DIR=$TENSORBOARD_DIR
mkdir -p $OUTPUT_DIR/tensorboard/wandb
mkdir -p $CHECKPOINT_DIR
mkdir -p $TENSORBOARD_DIR

DS_CONFIG=./ds_config-gpt.json
cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": $ZERO_STAGE
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

# "csv_monitor": {
#   "enabled": true,
#   "output_path": "$OUTPUT_DIR/ds_logs/",
#   "job_name": "pretrain-gpt",
#   "group": "polaris"
# }

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_mpi ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
# ds_args=" --no-pipeline-parallel ${ds_args}"

  # --save $CHECKPOINT_DIR \
  # --load $CHECKPOINT_DIR \
gpt_args="\
  --tensor-model-parallel-size $TP \
  --pipeline-model-parallel-size $PP \
  --num-layers $NLAYERS \
  --hidden-size $HIDDEN \
  --num-attention-heads 16 \
  --micro-batch-size 4 \
  --global-batch-size 16 \
  --seq-length 1024 \
  --max-position-embeddings 1024 \
  --train-iters 200 \
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
  --eval-interval 50 \
  --eval-iters 100 \
  --tensorboard-dir ${TENSORBOARD_DIR} \
  --log-timers-to-tensorboard \
  --tensorboard-log-interval 5 \
  --fp16"


setup() {
  if [[ $(hostname) == theta* ]]; then
    echo "Setting up ThetaGPU from $(hostname)"
    setupThetaGPU
  elif [[ $(hostname) == x* ]]; then
    echo "Setting up Polaris from $(hostname)"
    setupPolaris
  else
    echo "Unexpected hostname $(hostname)"
  fi

  export NODE_RANK=0
  export NNODES=$NRANKS
  export GPUS_PER_NODE=$NGPU_PER_RANK
  export WORLD_SIZE=$NGPUS
}

singleGPU() {
  echo "\
    Running on 1 ranks \
    with 1 GPUs each \
    for a total of 1 GPUs"
  EXEC="\
    $(which python3) \
    ./pretrain_gpt.py \
    ${gpt_args} \
    ${ds_args}"
  echo "EXEC: $EXEC" "$@"
  ${EXEC} "$@"
}

fullNode() {
  NRANKS=1
  NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
  NGPUS=$((${NRANKS}*${NGPU_PER_RANK}))
  echo "\
    Running on $NRANKS ranks \
    with $NGPU_PER_RANK GPUs each \
    for a total of $NGPUS GPUs"
  EXEC="\
    ${MPI_COMMAND} \
    ${MPI_DEFAULTS} \
    -n ${NGPUS}
    $(which python3) \
    ./pretrain_gpt.py \
    ${gpt_args} \
    ${ds_args}"

  echo "EXEC: $EXEC" "$@"
  ${EXEC} "$@"
}

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Use all available GPUs on all available nodes ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
elasticDistributed() {
  NRANKS=$(wc -l < "${HOSTFILE}")
  NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
  NGPUS=$((${NRANKS}*${NGPU_PER_RANK}))
  echo "\
    Running on ${NRANKS} ranks \
    with ${NGPU_PER_RANK} GPUs each \
    for a total of ${NGPUS} GPUs"
  EXEC="\
    ${MPI_COMMAND} \
    ${MPI_DEFAULTS} \
    ${MPI_ELASTIC} \
    $(which python3) \
    ./pretrain_gpt.py \
    ${gpt_args} \
    ${ds_args}"
  ${EXEC} "$@"
}

setup
# singleGPU "$@"
elasticDistributed "$@"
