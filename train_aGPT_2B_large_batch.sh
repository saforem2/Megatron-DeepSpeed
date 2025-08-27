#!/bin/bash --login
#PBS -q prod
#PBS -A AuroraGPT
#PBS -j oe
#PBS -l walltime=06:00:00,filesystems=flare:home
#PBS -l select=256

cd /flare/datascience/foremans/projects/argonne-lcf/Megatron-DeepSpeed || {
  echo "Failed to change directory to /flare/datascience/foremans/projects/argonne-lcf/Megatron-DeepSpeed"
  exit 1
}

PBS_O_WORKDIR="$(pwd)"
export PBS_O_WORKDIR

export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"


# shellcheck disable=SC1090
source <(curl -L https://bit.ly/ezpz-utils)
ezpz_setup_env

# eval "$(micromamba shell hook --shell "$(basename "${SHELL}")")"
# ezpz_load_new_pt_modules_aurora
# micromamba activate /flare/datascience_collab/foremans/micromamba/envs/pt29-2025-07
#
# NODES=$(cat "${PBS_NODEFILE}" | uniq | wc -l)
# mpiexec -np "${NODES}" --ppn 1 python3 \
#   /flare/datascience/foremans/projects/argonne-lcf/scalable_conda_env/cache_soft.py \
#   --src=/flare/datascience/foremans/pt29-ezpz-2025-07.tar.gz \
#   --dst=/tmp/pt29-ezpz-2025-07.tar.gz \
#   --d
#
# micromamba deactivate
# micromamba activate /tmp/pt29-ezpz-2025-07
# which python3

MODEL_ARCH=smollm3-3B \
  OPT=ipex.fusedlamb \
  NLAYERS=12 \
  GRAD_ACC_STEPS=2 \
  MICRO_BATCH=1 \
  USE_ACTIVATION_CHECKPOINTING=0 \
  ZERO_STAGE=0 \
  OPT=adamw \
  LR_DECAY_STYLE=constant \
  TOKENIZER_TYPE=HFTokenizer \
  TOKENIZER_MODEL=google/gemma-7b \
  DATA_FILE_LIST=ALCF/data-lists/aurora/olmo-mix-1124.txt \
  bash "${PBS_O_WORKDIR}/train_alcf.sh"
