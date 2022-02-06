#!/bin/bash
RUN_DIR=$(dirname $(readlink -f $0))
PKG_DIR=$(dirname ${RUN_DIR})

SRC_IMAGE=""
DST_IMAGE="pmod"

function usage_exit {
  cat <<_EOS_ 1>&2
  Usage: build-docker.sh [OPTIONS...]
  OPTIONS:
    -h, --help                          Show this help
    -i, --base-image DOCKER_IMAGE[:TAG] Specify the base Docker image
_EOS_
  exit 1
}

while (( $# > 0 )); do
  if [[ $1 == "-h" ]] || [[ $1 == "--help" ]]; then
    usage_exit
  elif [[ $1 == "-i" ]] || [[ $1 == "--base-image" ]]; then
    if [[ $2 == -* ]]; then
      echo "Invalid parameter"
      usage_exit
    else
      SRC_IMAGE=$2
    fi
    shift 2
  else
    echo "Invalid parameter: $1"
    usage_exit
  fi
done

if [[ -z ${SRC_IMAGE} ]]; then
  echo "Specify the base Docker image."
  usage_exit
fi

SRC_IMAGE_ARR=(${SRC_IMAGE//:/ })

IMAGE_EXIST=$(docker images ${SRC_IMAGE} | grep ${SRC_IMAGE_ARR[0]})
if [[ -z ${IMAGE_EXIST} ]]; then
  echo "The specified Docker image does not exist."
  usage_exit
fi

docker build \
  -t "${DST_IMAGE}" \
  --build-arg SRC_IMAGE="${SRC_IMAGE}" \
  ${RUN_DIR}

if [[ $? != 0 ]]; then
    echo "Aborted due to an error."
    exit 1
fi
