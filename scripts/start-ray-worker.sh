#!/bin/bash
set -e

source .venv/bin/activate

eval "$(direnv export bash)"  # makes .envrc variables available

# Make sure Makefile-exported env vars are present
: "${HEAD_HOST:?Need to set HEAD_HOST}"
: "${SERVER_RAY_PORT:?Need to set SERVER_RAY_PORT}"
: "${ALLOC_CPUS:?Need to set ALLOC_CPUS}"
: "${ALLOC_GPUS:?Need to set ALLOC_GPUS}"
: "${ALLOC_RAMS:?Need to set ALLOC_RAMS}"

# Start Ray worker
POLARS_SKIP_CPU_CHECK=1 ray start \
    --address=${HEAD_HOST}:${SERVER_RAY_PORT} \
    --node-ip-address=$(hostname -I | awk '{print $1}') \
    --num-cpus=${ALLOC_CPUS} \
    --num-gpus=${ALLOC_GPUS} \
    --memory=${ALLOC_RAMS} \
    --object-spilling-directory=/tmp/spill

make logs-ray