#!/bin/bash
RUN_DIR=$(dirname $(readlink -f $0))
XSOCK="/tmp/.X11-unix"
XAUTH="/tmp/.docker.xauth"

DEVICE_ID=0

DATASET_DIR=""

function usage_exit {
  cat <<_EOS_ 1>&2
  Usage: run.sh [OPTIONS...]
  OPTIONS:
    -h, --help          Show this help
    -i, --gpu-id ID     Specify the ID of the GPU
    -d, --dataset-dir   Specify the directory where the data set is stored
_EOS_
  exit 1
}

while (( $# > 0 )); do
  if [[ $1 == "-h" ]] || [[ $1 == "--help" ]]; then
    usage_exit
  elif [[ $1 == "-i" ]] || [[ $1 == "--gpu-id" ]]; then
    if [[ $2 == -* ]]; then
      echo "Invalid parameter"
      usage_exit
    fi
    DEVICE_ID=$2
    shift 2
  elif [[ $1 == "-d" ]] || [[ $1 == "--dataset-dir" ]]; then
    if [[ $2 == -* ]]; then
      echo "Invalid parameter"
      usage_exit
    fi
    DATASET_DIR=$2
    shift 2
  else
    echo "Invalid parameter: $1"
    usage_exit
  fi
done

DOCKER_VOLUME="${DOCKER_VOLUME} -v $(dirname ${RUN_DIR}):/workspace/pmod:rw"
DOCKER_VOLUME="${DOCKER_VOLUME} -v ${XSOCK}:${XSOCK}:rw"
DOCKER_VOLUME="${DOCKER_VOLUME} -v ${XAUTH}:${XAUTH}:rw"
if [[ ${DATASET_DIR} != "" ]]; then
  DOCKER_VOLUME="${DOCKER_VOLUME} -v ${DATASET_DIR}:/workspace/pmod/datasets:rw"
fi

DOCKER_ENV="${DOCKER_ENV} -e XAUTHORITY=${XAUTH}"
DOCKER_ENV="${DOCKER_ENV} -e DISPLAY=$DISPLAY"
DOCKER_ENV="${DOCKER_ENV} -e TERM=xterm-256color"
DOCKER_ENV="${DOCKER_ENV} -e QT_X11_NO_MITSHM=1"
DOCKER_ENV="${DOCKER_ENV} -e HOST_NAME=$(hostname)"

docker run \
    -it \
    --rm \
    --gpus '"device='${DEVICE_ID}'"' \
    -p $(($DEVICE_ID + 6006)):6006 \
    -w /workspace/pmod \
    ${DOCKER_VOLUME} \
    ${DOCKER_ENV} \
    --name pmod-${DEVICE_ID} \
    shikishimatasakilab/pmod
