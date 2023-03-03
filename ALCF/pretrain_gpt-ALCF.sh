#!/bin/bash -l

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
PARENT=$(dirname "${DIR}")

source "${DIR}/launch.sh"


setup
# fullNode  "$@" 2>&1 &
elasticDistributed "$@" 2>&1 &
