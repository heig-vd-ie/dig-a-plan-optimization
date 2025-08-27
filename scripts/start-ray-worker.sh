#!/bin/bash
set -e

source .venv/bin/activate

HEAD_HOST=${1}
SERVER_RAY_PORT=${2}
ALLOC_CPUS=${3}
ALLOC_GPUS=${4}
ALLOC_RAMS=${5}

# Start Ray worker
POLARS_SKIP_CPU_CHECK=1 ray start \
    --address=${HEAD_HOST}:${SERVER_RAY_PORT} \
    --node-ip-address=$(hostname -I | awk '{print $1}') \
    --num-cpus=${ALLOC_CPUS} \
    --num-gpus=${ALLOC_GPUS} \
    --memory=${ALLOC_RAMS} \
    --object-spilling-directory=/tmp/spill
