#!/bin/bash -l

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
# PARENT=$(dirname "${DIR}")

LAUNCH_FILE="${DIR}/launch.sh"
if [[ -f "${LAUNCH_FILE}" ]]; then
  echo "source-ing ${LAUNCH_FILE}"
  source "${LAUNCH_FILE}"
else
  echo "ERROR: UNABLE TO SOURCE ${LAUNCH_FILE}"
fi
# source "${DIR}/launch.sh"


setup
singleGPU "$@" 2>&1 &
# fullNode  "$@" 2>&1 &
# elasticDistributed "$@" 2>&1 &
