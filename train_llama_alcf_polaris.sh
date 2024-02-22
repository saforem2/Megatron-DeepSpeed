#!/bin/bash --login
#PBS -l walltime=06:00:00
#PBS -A argonne_tpc
#PBS -q prod
#PBS -l select=48
#PBS -l filesystems=eagle:home

ezpz() {
    if [[ ! -d ezpz ]]; then
        git clone https://github.com/saforem2/ezpz
    else
        echo "Found ezpz!"
    fi
    echo "Using $(which python3) to install \`ezpz\`:"
    mkdir -p logs
    python3 -m pip install -e ezpz > ezpz-install.log 2>&1
    source ezpz/src/ezpz/bin/savejobenv > /tmp/savejobenv.log 2>&1 || exit
    source ezpz/src/ezpz/bin/getjobenv || exit
}

makeHostfiles() {
    GPUS_PER_NODE=$(python3 -Wignore -c 'import ezpz; print(ezpz.get_gpus_per_node())')
    export GPUS_PER_NODE="${GPUS_PER_NODE}"
    # ---- Make MPICH hostfile ----------------
    export hostfile_mpich=hostfile_mpich
    cat "$PBS_NODEFILE" > "${hostfile_mpich}"
    # ---- Make DeepSpeed hostfile -------------------
    export hostfile_deepspeed=hostfile_deepspeed
    cat "$PBS_NODEFILE" > "${hostfile_deepspeed}"
    sed -e "s/$/ slots=${GPUS_PER_NODE}/" -i "${hostfile_deepspeed}"
    {
        echo "PATH=${PATH}" ;
        echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" ;
        echo "http_proxy=${http_proxy}" ;
        echo "https_pro]xy=${https_proxy}" ;
        echo "CFLAGS=${CFLAGS}" ;
        echo "PYTHONUSERBASE=$PYTHONUSERBASE" ;
    } > .deepspeed_env
    # echo "PATH=${PATH}" > .deepspeed_env
    # echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> .deepspeed_env
    # echo "http_proxy=${http_proxy}" >> .deepspeed_env
    # echo "https_proxy=${https_proxy}" >> .deepspeed_env
    # echo "CFLAGS=${CFLAGS}" >> .d eepspeed_env
    # echo "PYTHONUSERBASE=$PYTHONUSERBASE" >> .deepspeed_env
    # -------------------------------------------------
}

setupData() {
    local cidx=$1
    # export DOLMA_CHUNK_IDX="${DOLMA_CHUNK_IDX:-0}"
    # HERE=$(python3 -c 'import os; print(os.getcwd())')
    export DATA_FILE_LIST="/eagle/datasets/dolma/chunks/4/data_file_list_chunk_${cidx}_of_4.txt"
    NDOCS=$(wc -l < "${DATA_FILE_LIST}")
    export NDOCS="${NDOCS}"
    if [[ -z "${DATA_CACHE_PATH}" ]]; then
        data_file_list_stem=$(echo "$DATA_FILE_LIST" | tr "\/" "\t" | awk '{print $NF}' | sed "s/\.txt//g")
        export DATA_CACHE_PATH=".cache/${data_file_list_stem}-index-cache"
    else
        export DATA_CACHE_PATH="${DATA_CACHE_PATH}"
        echo "CAUGHT DATA_CACHE_PATH: ${DATA_CACHE_PATH} from env !!"
    fi
    export DOLMA_CHUNK_IDX="${cidx}"
    mkdir -p "${DATA_CACHE_PATH}"
}


# ==== SCRIPT START ========================================================
cd "${PBS_O_WORKDIR}" || exit
module load conda/2023-10-04; conda activate base
ezpz
makeHostfiles
setupData "${DOLMA_CHUNK_IDX:-0}"
echo "Using DOLMA CHUNK ${DOLMA_CHUNK_IDX} from ${DATA_FILE_LIST} with ${NDOCS} documents..."

# ---- Parallelism Settings ----
export PP=${PP:-1}
export TP=${TP:-2}
# ------------------------------

HERE=$(python3 -c 'import os; print(os.getcwd())')
HOSTFILE="${HOSTFILE:-${PBS_NODEFILE}}"
export WORLD_SIZE=${WORLD_SIZE:-$(wc -l < "${HOSTFILE}")}

# ---- Llama2 7B Config -----------------------
export HEADS=${HEADS:-32}
export NLAYERS=${NLAYERS:-32}
export HIDDEN=${HIDDEN:-4096}
export NUM_KV_HEAD=${NUM_KV_HEAD:-8}
# ---------------------------------------------

# ---- Run Settings ------------------------------------------
export LR=${LR:-0.00015}
export SEQ=${SEQ:-4096}                       # SEQ_LEN: 4096
export DTYPE=${DTYPE:-fp16}                   # DTYPE: FP16
export ZERO_STAGE=${ZERO_STAGE:-1}
export MICRO_BATCH=${MICRO_BATCH:-8}
export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}
export GLOBAL_BATCH=$(( $WORLD_SIZE * $MICRO_BATCH * $GRAD_ACC_STEPS / $TP / $PP ))
# ---------------------------------------------

if [[ "${DOLMA_CHUNK_IDX}" == 0 ]]; then
    TRAIN_ITER=78739
elif [[ "${DOLMA_CHUNK_IDX}" == 1 ]]; then
    TRAIN_ITER=81008
elif [[ "${DOLMA_CHUNK_IDX}" == 2 ]]; then
    TRAIN_ITER=79591
elif [[ "${DOLMA_CHUNK_IDX}" == 3 ]]; then
    TRAIN_ITER=78552
else
    echo "Unknown DOLMA_CHUNK_IDX: ${DOLMA_CHUNK_IDX}"
    exit
fi
# export TRAIN_ITER=${TRAIN_ITER:-317892}

export SAVE_INTERVAL=${SAVE_INTERVAL:-200}
export USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING:-1}
# export USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING:-0}

export MODEL_TYPE="llama-seq${SEQ}-pp${PP}-tp${TP}-${NLAYERS}layers-${HEADS}heads-${HIDDEN}hidden"
export LLAMA_ARGS="--no-query-key-layer-scaling --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear"
# export DATA_FILE_LIST="/lus/eagle/projects/datasets/dolma/chunks/40/data_file_list_chunk_0_of_40.txt"
# export DATA_FILE_LIST="/lus/eagle/projects/datasets/dolma/chunks/10/data_file_list_chunk_0_of_10.txt"
#

# export DOLMA_CHUNK_00_of_10="./chunks/10/data_file_list_chunk_00_of_10.txt"  # 762 documents (lines)
# export DOLMA_CHUNK_01_of_10="./chunks/10/data_file_list_chunk_01_of_10.txt"  # 722
# export DOLMA_CHUNK_02_of_10="./chunks/10/data_file_list_chunk_02_of_10.txt"  # 727
# export DOLMA_CHUNK_03_of_10="./chunks/10/data_file_list_chunk_03_of_10.txt"  # 707
# export DOLMA_CHUNK_04_of_10="./chunks/10/data_file_list_chunk_04_of_10.txt"  # 744
# export DOLMA_CHUNK_05_of_10="./chunks/10/data_file_list_chunk_05_of_10.txt"  # 766
# export DOLMA_CHUNK_06_of_10="./chunks/10/data_file_list_chunk_06_of_10.txt"  # 730
# export DOLMA_CHUNK_07_of_10="./chunks/10/data_file_list_chunk_07_of_10.txt"  # 759
# export DOLMA_CHUNK_08_of_10="./chunks/10/data_file_list_chunk_08_of_10.txt"  # 777
# export DOLMA_CHUNK_09_of_10="./chunks/10/data_file_list_chunk_09_of_10.txt"  # 752

#
# export DOLMA_CHUNK_00_of_04="./dolma_data_file_list-00-of-04.txt"  # 1860 documents (lines)
# export DOLMA_CHUNK_01_of_04="./dolma_data_file_list-01-of-04.txt"  # 1860 documents (lines)
# export DOLMA_CHUNK_02_of_04="./dolma_data_file_list-02-of-04.txt"  # 1860 documents (lines)
# export DOLMA_CHUNK_03_of_04="./dolma_data_file_list-03-of-04.txt"  # 1860 documents (lines)
# export DOLMA_CHUNK_04_of_04="./dolma_data_file_list-04-of-04.txt"  # 6 documents (lines)



# if [[ -n "$DEBUG_RUN" ]]; then
#     # echo "Using LAST DOLMA CHUNK {09 / 10} with ${NDOCS} documents..."
#     export DATA_FILE_LIST=${DATA_FILE_LIST:-${DOLMA_CHUNK_09_of_10}}
#     # export ndocs=$(wc -l < "${DATA_FILE_LIST}")
# else
#     # export fname="./chunks/10/data_file_list_chunk_${DOLMA_CHUNK_IDX}_of_10.txt"
#     # export fname="${DOLMA_CHUNK_!{DOLMA_CHUNK_IDX}_of_10}"
#     # export DATA_FILE_LIST="${DATA_FILE_LIST:-${DOLMA_CHUNK_00_of_10}}"
# fi
# export ndocs
# export DATA_FILE_LIST="${DATA_FILE_LIST}"
# export DATA_FILE_LIST="./dolma_data_file_list-00-of-04.txt"
# export DATA_FILE_LIST="./dolma-chunk-00-of-40.txt"
#
# bash "${HERE}/generate_config.sh" "${DS_CONFIG}" || exit

# export DATA_CACHE_PATH="${DATA_CACHE_PATH}"
# if [[ -z "$DATA_CACHE_PATH" ]]; then
#     echo "Not using DATA_CACHE_PATH !!"
# else
#     echo "Using DATA_CACHE_PATH: ${DATA_CACHE_PATH}"
# fi


export EXTRA_ARGS="--use-flash-attn-v2 --num-key-value-heads ${NUM_KV_HEAD}"

# export DATA_FILE_LIST="./chunks/10/data_file_list_chunk_${DOLMA_CHUNK_IDX}_of_10.txt"
# export DATA_FILE_LIST="/lus/eagle/projects/datasets/dolma/chunks/data_file_list_chunk_${DOLMA_CHUNK_IDX}_of_20.txt"
# export DATA_FILE_LIST="./dolma_data_file_list-${DOLMA_CHUNK_IDX}-of-04.txt"

echo "++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "- WORLD_SIZE:${WORLD_SIZE}"
echo "- NCCL: ${NCCL:-nccl}"
echo "- MODEL_TYPE: ${MODEL_TYPE}"
echo "- Using DATA_FILE_LIST: ${DATA_FILE_LIST}"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++"

# bash $LLM_DK_DIR/intel-extension-for-deepspeed/examples/gpt.sh $@

DS_CONFIG="ds_stage${ZERO_STAGE}_mb${MICRO_BATCH}_gb${GLOBAL_BATCH}_pp${PP}_${DTYPE}.json"
bash ./generate_config.sh "${DS_CONFIG}" || exit 1

OUTPUT_PREFIX="logs/ds_stage${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_mb${MICRO_BATCH}_seq${SEQ}_gb${GLOBAL_BATCH}_pp${PP}_tp${TP}_${DTYPE}"
# OUTPUT_DIR=logs/ds_stage${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_mb${MICRO_BATCH}_seq${SEQ}_gb${GLOBAL_BATCH}_pp${PP}_tp${TP}_${DTYPE}_`date +%m%d%H%M%S`_${HOSTNAME}
OUTPUT_DIR="${OUTPUT_PREFIX}/$(date +%m%d%H%M%S)_${HOSTNAME}"
mkdir -p "${OUTPUT_DIR}"
echo "!!!Please see logs at ${OUTPUT_DIR}"

# Hostfile path
hostfile_deepspeed=./hostfile_deepspeed
hostfile_mpich=./hostfile_mpich
cat "$PBS_NODEFILE" > hostfile_mpich
cat "$PBS_NODEFILE" > hostfile_deepspeed ; sed -e 's/$/ slots=4/' -i hostfile_deepspeed

ds_args=" "
ds_args=" --deepspeed ${ds_args}"
if [ "$PP" == 1 ]; then
   ds_args=" --no-pipeline-parallel ${ds_args}" 
fi
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"

if [[ "$USE_ACTIVATION_CHECKPOINTING" == 1 ]]; then
    echo "!! Caught USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING} !!"
    ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
    # --checkpoint-activations \
    # --deepspeed-activation-checkpointing
fi

gpt_args=()

if [[ "$USE_ACTIVATION_CHECKPOINTING" == 1 ]]; then
    echo "!! Caught USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING} !!"
    gpt_args+=(
        "--checkpoint-activations"
        "--checkpoint-num-layers 1"
    )
fi
# we are now using activation checkpoint provided by megatron, see below.
# ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
# NUM_KV_HEADS="${NUM_KV_HEADS:-0}"
# if [[ $NUM_KV_HEADS -]]

# take custom args
custom_args=" $@"

# launcher setting
LAUNCHER=${LAUNCHER:-MPICH}
if [[ $LAUNCHER == "deepspeed" ]]; then
    launcher=""
else
    launcher="--force_multi --hostfile $hostfile_deepspeed --launcher=${LAUNCHER} --launcher_args='-hostfile ${hostfile_mpich}'"
fi

NCCL=${NCCL:-nccl}
EXEC="./pretrain_gpt_alcf.py"

# MODEL=LLAMA_7B
# OUTPUT_PREFIX=${MODEL}_z${ZERO_STAGE}_seqlen_tp${TP}_pp${PP}_sp${SP}_nl${NUM_LAYERS}_hs${HIDDEN_SIZE}_gb${BS}_mb${MBS}

# --vocab-file $VOCAB_FILE \
# --merge-file $MERGE_FILE \
# --lr-decay-iters 320000 \
    # --num-workers 0 \
    # --eval-iters ${EVAL_ITERS} \
    # --eval-interval ${EVAL_INTERVAL} \
    # --lr-warmup-iters 5000 \
    # --lr-decay-iters 10000 \
run_cmd="
    deepspeed $launcher ${EXEC} \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --ffn-hidden-size 11008 \
    --num-attention-heads $HEADS \
    --seq-length $SEQ \
    --max-position-embeddings $SEQ \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --train-iters $TRAIN_ITER \
    --lr ${LR} \
    --lr-decay-style cosine \
    --data-impl mmap \
    --log-interval 1 \
    --save-interval ${SAVE_INTERVAL} \
    --split 90,5,5 \
    --$DTYPE \
    $ds_args \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-gradient-accumulation-fusion \
    --distributed-backend $NCCL \
    --tokenizer-type Llama2Tokenizer \
    --save checkpoints/${OUTPUT_PREFIX} \
    --load checkpoints/${OUTPUT_PREFIX} \
    --use-checkpoint-opt_param-scheduler \
    --accumulate-allreduce-grads-in-fp32 \
    --tokenizer-model /eagle/datasets/dolma/utils/tokenizer.model \
    --data-file-list ${DATA_FILE_LIST} \
    --data-cache-path ${DATA_CACHE_PATH} \
    --num-workers 4 \
    ${LLAMA_ARGS} \
    ${EXTRA_ARGS} \
    ${gpt_args[*]} \
    $custom_args \
    |& tee $OUTPUT_DIR/output.log
    "

echo "Using $(which deepspeed)"
ds_report

echo ${run_cmd}
eval ${run_cmd}
set +x
