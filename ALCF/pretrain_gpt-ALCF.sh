#!/bin/bash -l


TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
PARENT=$(dirname $DIR)

source "${DIR}/launch.sh"

printJobInfo() {
  echo "Job started at: ${TSTAMP} on $(hostname)" 
  echo "Job running in: ${DIR}" 
  echo "Training GPT-3 with ${MODEL_SIZE} parameters" 
  echo "Writing logs to: ${OUTPUT_LOG}" 
  echo "to view output: 'tail -f $(tail -1 ${PARENT}/logfiles)'"
  echo "EXEC: ${EXEC}"  #  $@
}


setup
printJobInfo | tee -a ${OUTPUT_LOG}
# singleGPU "$@" >> ${OUTPUT_LOG} 2>&1 &
fullNode  "$@" >> "${OUTPUT_LOG}" 2>&1 &
# elasticDistributed "$@"  >> ${OUTPUT_LOG} 2>&1 &
