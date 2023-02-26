#!/bin/bash --login

# TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
# DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
#
condaThetaGPU() {
  module load conda/2022-07-01 ; conda activate base
  conda activate \
    /lus/grand/projects/datascience/foremans/locations/thetaGPU/miniconda3/envs/2022-07-01
  source ./venvs/thetaGPU/2022-07-01-deepspeed/bin/activate
}

# ┏━━━━━━━━━━┓
# ┃ ThetaGPU ┃
# ┗━━━━━━━━━━┛
setupThetaGPU() {
  if [[ $(hostname) == theta* ]]; then
    export MACHINE="ThetaGPU"
    HOSTFILE="${COBALT_NODEFILE}"
    # -- Python / Conda setup -------------------------------------------------
    condaThetaGPU
    # -- MPI / Comms Setup ----------------------------------------------------
    NRANKS=$(wc -l < "${HOSTFILE}")
    NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
    NGPUS=$((${NRANKS}*${NGPU_PER_RANK}))
    NVME_PATH="/raid/scratch/"
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


condaPolaris() {
  # -- Python / Conda setup -------------------------------------------------
  module load conda/2023-01-10
  conda activate base
  conda activate \
    /lus/grand/projects/datascience/foremans/locations/polaris/miniconda3/envs/2023-01-10
}

# ┏━━━━━━━━━┓
# ┃ Polaris ┃
# ┗━━━━━━━━━┛
setupPolaris()  {
  if [[ $(hostname) == x* ]]; then
    export MACHINE="Polaris"
    HOSTFILE="${PBS_NODEFILE}"
    # -- MPI / Comms Setup ----------------------------------------------------
    condaPolaris
    # export IBV_FORK_SAFE=1
    NRANKS=$(wc -l < "${HOSTFILE}")
    NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
    NGPUS=$((${NRANKS}*${NGPU_PER_RANK}))
    MPI_COMMAND=$(which mpiexec)
    NVME_PATH="/local/scratch/"
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

export NVME_PATH="${NVME_PATH}"
export MPI_DEFAULTS="${MPI_DEFAULTS}"
export MPI_ELASTIC="${MPI_ELASTIC}"
export MPI_COMMAND="${MPI_COMMAND}"

PYTHON_EXECUTABLE="$(which python3)"
export PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}"
# source "${DIR}/args.sh"

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
