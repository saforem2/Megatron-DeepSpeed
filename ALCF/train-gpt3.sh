#!/bin/bash -l

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
# PARENT=$(dirname "${DIR}")
#
PIDS=$(ps aux | grep pretrain_gpt.py | grep -v grep | awk '{print $2}')
if [ -n "${PIDS}" ]; then
  echo "Already running! Exiting!"
  exit 1
fi


LAUNCH_FILE="${DIR}/launch.sh"
if [[ -f "${LAUNCH_FILE}" ]]; then
  echo "source-ing ${LAUNCH_FILE}"
  # shellcheck source=./launch.sh
  source "${LAUNCH_FILE}"
else
  echo "ERROR: UNABLE TO SOURCE ${LAUNCH_FILE}"
fi
# source "${DIR}/launch.sh"


setup
# singleGPU "$@" 2>&1 &
fullNode "$@" 2>&1 &
# elasticDistributed "$@" 2>&1 &
