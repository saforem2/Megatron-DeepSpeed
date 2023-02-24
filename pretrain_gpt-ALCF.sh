#!/bin/bash -l

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)

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
    # -- MPI / Comms Setup ----------------------------------------------------
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
    # -- Python / Conda setup -------------------------------------------------
    module load conda/2023-01-10
    conda activate base
    conda activate \
      /lus/grand/projects/datascience/foremans/locations/polaris/miniconda3/envs/2023-01-10
    # -- MPI / Comms Setup ----------------------------------------------------
    # export IBV_FORK_SAFE=1
    NRANKS=$(wc -l < "${HOSTFILE}")
    NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
    NGPUS=$((${NRANKS}*${NGPU_PER_RANK}))
    MPI_COMMAND=$(which mpiexec)
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

export NCCL_DEBUG=warn
export WANDB_CACHE_DIR="./cache/wandb"
export CFLAGS="-I${CONDA_PREFIX}/include/"
export LDFLAGS="-L${CONDA_PREFIX}/lib/"


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Model / Architecture settings                       ┃
# ┃ --------------------------------------------------- ┃
# ┃ GPT-3 models use 2K sequence length/context window  ┃
# ┃ The "GPT-3 XXX" below are configs from GPT-3 paper  ┃
# ┃ https://arxiv.org/abs/2005.14165, choose based on   ┃
# ┃ your desired model size or build your own configs   ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
SEQ_LEN=4096

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
MPSIZE=8
PPSIZE=16
MICRO_BATCH=1
ZERO_STAGE=1

# ┏━━━━━━━━━━━━┓
# ┃ Data paths ┃
# ┗━━━━━━━━━━━━┛
DATA_PATH="./dataset/BookCorpusDataset_text_document"
VOCAB_FILE="./dataset/gpt2-vocab.json"
MERGE_FILE="./dataset/gpt2-merges.txt"

# ┏━━━━━━━━━━━━━━━━━━━┓
# ┃ FILE I/O SETTINGS ┃
# ┗━━━━━━━━━━━━━━━━━━━┛
RUN_STR="gb${GLOBAL_BATCH}_mb${MICRO_BATCH}"
RUN_STR="nl${NLAYERS}_hs${HIDDEN}_${RUN_STR}"
RUN_STR="mp${MPSIZE}_pp${PPSIZE}_${RUN_STR}"
RUN_STR="z${ZERO_STAGE}_seqlen${SEQ_LEN}_${RUN_STR}"
RUN_STR="GPT3_${MODEL_SIZE}_${RUN_STR}"
OUTPUT_DIR="./outputs/${RUN_STR}"
OUTPUT_LOG="./outputs/${RUN_STR}/logs/$TSTAMP.log"
CHECKPOINT_DIR="./checkpoints/$RUN_STR"
TENSORBOARD_DIR="./outputs/${RUN_STR}/tensorboard"
export TENSORBOARD_DIR=$TENSORBOARD_DIR
mkdir -p $OUTPUT_DIR/tensorboard/wandb
mkdir -p $CHECKPOINT_DIR
mkdir -p $TENSORBOARD_DIR
mkdir -p "$(dirname "${OUTPUT_LOG}")"
echo "${OUTPUT_LOG}" >> logfiles

# ┏━━━━━━━━━━━━━━━━━━┓
# ┃ DeepSpeed Config ┃
# ┗━━━━━━━━━━━━━━━━━━┛
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

# ┏━━━━━━━━━━━━━━━━━━━━━┓
# ┃ DeepSpeed Arguments ┃
# ┗━━━━━━━━━━━━━━━━━━━━━┛
ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_mpi ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"

# ┏━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ MEGATRON-LM SETTINGS ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━┛
gpt_args="\
  --tensor-model-parallel-size $MPSIZE \
  --pipeline-model-parallel-size $PPSIZE \
  --num-layers $NLAYERS \
  --hidden-size $HIDDEN \
  --num-attention-heads ${ATEN_HEADS} \
  --micro-batch-size ${MICRO_BATCH} \
  --global-batch-size ${GLOBAL_BATCH} \
  --seq-length ${SEQ_LEN} \
  --max-position-embeddings ${SEQ_LEN} \
  --train-iters 10 \
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
  --eval-interval 5 \
  --eval-iters 10 \
  --tensorboard-dir ${TENSORBOARD_DIR} \
  --log-timers-to-tensorboard \
  --tensorboard-log-interval 1 \
  --fp16"

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ SETUP CONDA + MPI ENVIRONMENT @ ALCF ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
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

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Use all available GPUs a single nodes ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
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

printJobInfo() {
  echo "Job started at: ${TSTAMP} on $(hostname)" 
  echo "Job running in: ${DIR}" 
  echo "Training GPT-3 with ${MODEL_SIZE} parameters" 
  echo "Writing logs to: ${OUTPUT_LOG}" 
  echo "to view output: 'tail -f $(tail -1 logfiles)'"
}

setup
printJobInfo | tee -a ${OUTPUT_LOG}
# singleGPU "$@" >> ${OUTPUT_LOG} 2>&1 &
# fullNode  "$@" >> ${OUTPUT_LOG} 2>&1 &
elasticDistributed "$@"  >> ${OUTPUT_LOG} 2>&1 &
