#!/bin/bash

RUN_DIR=$(dirname $(readlink -f $0))
rm -rf ${RUN_DIR}/db/*
