#!/bin/bash --login

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
PARENT=$(dirname ${DIR})
MAIN="${PARENT}/pretrain_gpt.py"

ARGS_FILE="${DIR}/args.sh"
if [[ -f "${ARGS_FILE}" ]]; then
  echo "source-ing ${ARGS_FILE}"
  source "${ARGS_FILE}"
else
  echo "ERROR: UNABLE TO SOURCE ${ARGS_FILE}"
fi

SETUP_FILE="${DIR}/setup.sh"
if [[ -f "${SETUP_FILE}" ]]; then
  echo "source-ing ${SETUP_FILE}"
  source "${SETUP_FILE}"
else
  echo "ERROR: UNABLE TO SOURCE ${SETUP_FILE}"
fi

printJobInfo() {
  echo "Job started at: ${TSTAMP} on $(hostname)"
  echo "Job running in: ${DIR}"
  echo "Training GPT-3 with ${MODEL_SIZE} parameters"
  echo "Writing logs to: ${OUTPUT_DIR}"
  echo 'to view output: tail -f $(tail -1 ${PARENT}/logfiles)'
  echo "i.e. tail -f $(tail -1 $PARENT/logfiles)"
}

launchJob() {
  echo "using: $(which python3)" | tee -a "${OUTPUT_LOG}"
  printJobInfo | tee -a "${OUTPUT_LOG}"
  echo EXEC="${EXEC}" | tee -a "${OUTPUT_LOG}"
  echo "Writing logs to: ${OUTPUT_LOG}" | tee -a "${OUTPUT_LOG}"
  ${EXEC} "$@" # >> "${OUTPUT_LOG}" 2>&1 &
}

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
  OUTPUT_LOG="${OUTPUT_DIR}/logs/$USER-$HOST-nranks1-ngpu1-$TSTAMP.log"
  mkdir -p "$(dirname "${OUTPUT_LOG}")"
  echo "${OUTPUT_LOG}" >> "${PARENT}/logfiles"
  printJobInfo | tee -a "${OUTPUT_LOG}"
  launchJob "$@" >> "${OUTPUT_LOG}" 2>&1 &
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
  OUTPUT_LOG="${OUTPUT_DIR}/logs/$USER-$HOST-nranks1-ngpu${NGPUS}-$TSTAMP.log"
  mkdir -p "$(dirname "${OUTPUT_LOG}")"
  echo "${OUTPUT_LOG}" >> "${PARENT}/logfiles"
  printJobInfo | tee -a "${OUTPUT_LOG}"
  launchJob "$@" >> "${OUTPUT_LOG}" 2>&1 &
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
  EXEC_STR=(
    "${MPI_COMMAND}"
    "${MPI_DEFAULTS}"
    "${MPI_ELASTIC}"
    "$(which python3)"
    "${MAIN}"
    "${gpt_args}"
    "${ds_args}"
  )
  EXEC="${EXEC_STR[*]}"
  OUTPUT_LOG="${OUTPUT_DIR}/logs/$USER-$HOST-nranks${NRANKS}-ngpu${NGPUS}-$TSTAMP.log"
  mkdir -p "$(dirname "${OUTPUT_LOG}")"
  echo "${OUTPUT_LOG}" >> "${PARENT}/logfiles"
  printJobInfo | tee -a "${OUTPUT_LOG}"
  launchJob "$@" >> "${OUTPUT_LOG}" 2>&1 &
}
