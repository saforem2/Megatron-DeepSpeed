#!/bin/bash
#SBATCH -A m3957_g
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -N 16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none


export SLURM_CPU_BIND="cores"
# module load pytorch/2.0.1
module load pytorch/1.13.1
cd ~/Project2-ClimRR/foremans/projects/saforem2/Megatron-DeepSpeed

echo "which python: $(which python3)"
VENV_DIR="./venvs/perlmutter/1.13.1"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating new venv"
  mkdir -p "${VENV_DIR}"
  python3 -m venv ${VENV_DIR} --system-site-packages
fi
source "${VENV_DIR}/bin/activate"
echo "which python: $(which python3)"
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install deepspeed ninja regex apex einops flash_attn wandb
python3 -m pip install -e "."

# source ~/Project2-ClimRR/foremans/projects/saforem2/Megatron-DeepSpeed/venvs/perlmutter/torch2.0.1/bin/activate

ds_report

# printenv

export WORLD_SIZE="${SLURM_NTASKS}"

source ./ALCF/model.sh
source ./ALCF/args.sh

echo "gpt_args: ${gpt_args}"
echo "ds_args: ${ds_args}"
echo "which python: $(which python3)"
ARGS="${gpt_args} ${ds_args}"
MACHINE="perlmutter" MASTER_ADDR="127.0.0.1" MASTER_PORT=5432 srun -l -u python3 pretrain_gpt.py ${ARGS}

# MACHINE="perlmutter" MASTER_ADDR="127.0.0.1" MASTER_PORT=5432 \
#   srun -l -u python3 pretrain_gpt.py "${gpt_args} ${ds_args}"
    # --checkpoint-activations \
    # --seed 26149 \
    # --DDP-impl local \
    # --pipeline-model-parallel-size 1 \
    # --tensor-model-parallel-size 1 \
    # --num-layers 48 \
    # --hidden-size 1600 \
    # --num-attention-heads 25  \
    # --micro-batch-size 1   \
    # --seq-length 2048  \
    # --max-position-embeddings 2048  \
    # --train-iters 5   \
    # --lr-decay-iters 320000  \
    # --data-path ./dataset/BookCorpusDataset_text_document \
    # --vocab-file ./dataset/gpt2-vocab.json \
    # --merge-file ./dataset/gpt2-merges.txt  \
    # --data-impl mmap \
    # --split 949,50,1 \
    # --distributed-backend nccl \
    # --lr 0.00015  \
    # --lr-decay-style cosine  \
    # --min-lr 1.0e-5  \
    # --weight-decay 1e-2   \
    # --clip-grad 1.0   \
    # --lr-warmup-fraction .01  \
    # --log-interval 1   \
    # --save-interval 1000   \
    # --eval-interval 1000   \
    # --eval-iters 1  \
    # --tensorboard-dir ./tensorboard/ \
    # --log-timers-to-tensorboard  \
    # --tensorboard-log-interval 1 \
    # --fp16  \
    # --deepspeed-activation-checkpointing \
    # --no-pipeline-parallel \
    # --zero-stage=2 \
    # --deepspeed_config ./ds_config-gpt.json \
    # --deepspeed_mpi \
    # --deepspeed 
