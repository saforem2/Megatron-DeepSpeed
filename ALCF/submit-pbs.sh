#!/bin/bash --login

SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

TSTAMP=$(tstamp)

getValFromFile() {
  FILE=$1
  KEY=$2
  echo "getting ${KEY} from ${FILE}"
  if [[ -f "${FILE}" ]]; then
    VAL="$(cat "${FILE}" | grep -E "^${KEY}=" | sed "s/${KEY}=//g" | sed 's/\"//g')"
    echo "setting ${KEY}: ${VAL}"
    export "${KEY}"="${VAL}"
  fi
}


getValFromFile "${DIR}/model.sh" MODEL_SIZE

getValFromFile "${DIR}/args.sh" PPSIZE
getValFromFile "${DIR}/args.sh" MPSIZE
getValFromFile "${DIR}/args.sh" MICRO_BATCH
getValFromFile "${DIR}/args.sh" GRADIENT_ACCUMULATION_STEPS

QUEUE=$1
NUM_NODES=$2
DURATION=$3
PROJECT=$4


export QUEUE="${QUEUE}"
export DURATION="${DURATION}"
export TSTAMP="${TSTAMP}"
export MICRO_BATCH="${MICRO_BATCH}"
export NUM_NODES="${NUM_NODES}"
export MODEL_SIZE="${MODEL_SIZE}"
export PROJECT="${PROJECT}"
export GAS="${GRADIENT_ACCUMULATION_STEPS}"

RUN_NAME="N${NUM_NODES}-${TSTAMP}"
RUN_NAME="mb${MICRO_BATCH}-gas${GAS}-${RUN_NAME}"
RUN_NAME="GPT3-${MODEL_SIZE}-${RUN_NAME}"
export RUN_NAME="${RUN_NAME}"

echo "QUEUE=$QUEUE"
echo "PROJECT=$PROJECT"
echo "DURATION=$DURATION"
echo "TSTAMP=$TSTAMP"
echo "NUM_NODES=$NUM_NODES"
echo "MODEL_SIZE=$MODEL_SIZE"
echo "GAS=$GRADIENT_ACCUMULATION_STEPS"
echo "RUN_NAME: ${RUN_NAME}"

qsub \
  -q "${QUEUE}" \
  -A "${PROJECT}" \
  -N "${RUN_NAME}" \
  -l select="$NUM_NODES" \
  -l walltime="${DURATION}" \
  -l filesystems=eagle:home:grand \
  "${DIR}/submit.sh"
