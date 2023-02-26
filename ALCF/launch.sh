#!/bin/bash --login
#

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
PARENT=$(dirname ${DIR})

source "${DIR}/setup.sh"
source "${DIR}/args.sh"

MAIN="${PARENT}/pretrain_gpt.py"

singleGPU() {
  echo "\
    Running on 1 ranks \
    with 1 GPUs each \
    for a total of 1 GPUs"
  EXEC="\
    $(which python3) \
    ${MAIN} \
    ${gpt_args} \
    ${ds_args}"
  echo EXEC="${EXEC}"
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
    ${MAIN} \
    ${gpt_args} \
    ${ds_args}"
  echo EXEC="${EXEC}"
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
      ${MAIN} \
      ${gpt_args} \
      ${ds_args}"
  echo EXEC="${EXEC}"
  ${EXEC} "$@"
}
