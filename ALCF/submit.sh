#!/bin/bash --login
#
cd "${PBS_O_WORKDIR}" || exit

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
export TSTAMP="$TSTAMP"
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

#┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#┃ Make sure we're not already running; if so, exit here ┃
#┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
PIDS=$(ps aux | grep pretrain_gpt.py | grep -v grep | awk '{print $2}')
if [ -n "${PIDS}" ]; then
  echo "Already running! Exiting!"
  exit 1
fi

#┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#┃ source ./launch.sh                       ┃
#┃ which then sources ./{args.sh,setup.sh}  ┃
#┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
LAUNCH_FILE="/lus/grand/projects/datascience/foremans/locations/polaris/projects/saforem2/Megatron-DeepSpeed/ALCF/launch.sh"
if [[ -f "${LAUNCH_FILE}" ]]; then
  echo "source-ing ${LAUNCH_FILE}"
  # shellcheck source=./launch.sh
  source "${LAUNCH_FILE}"
else
  echo "ERROR: UNABLE TO SOURCE ${LAUNCH_FILE}"
fi


setup
elasticDistributed "$@"
wait $!
