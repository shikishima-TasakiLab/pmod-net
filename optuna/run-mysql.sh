#!/bin/bash

RUN_DIR=$(dirname $(readlink -f $0))

DOCKER_VOLUME="${DOCKER_VOLUME} -v ${RUN_DIR}/conf:/etc/mysql/conf.d:rw"
DOCKER_VOLUME="${DOCKER_VOLUME} -v ${RUN_DIR}/init:/docker-entrypoint-initdb.d:rw"
DOCKER_VOLUME="${DOCKER_VOLUME} -v ${RUN_DIR}/db:/var/lib/mysql:rw"

DOCKER_ENV="${DOCKER_ENV} -e MYSQL_RANDOM_ROOT_PASSWORD=yes"

docker run \
    -it \
    --rm \
    ${DOCKER_VOLUME} \
    ${DOCKER_ENV} \
    -p 13306:3306 \
    --name pmod-optuna-mysql \
    mysql
