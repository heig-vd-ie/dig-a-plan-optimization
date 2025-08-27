#!/bin/bash
set -e

SERVER_RAY_PORT=${1}
SERVER_RAY_DASHBOARD_PORT=${2}
SERVER_RAY_METRICS_EXPORT_PORT=${3}
ALLOC_CPUS=${4}
ALLOC_GPUS=${5}
ALLOC_RAMS=${6}
GRAFANA_PORT=${7}

# Start Ray head
ray start --head \
    --port=${SERVER_RAY_PORT} \
    --num-cpus=${ALLOC_CPUS} \
    --num-gpus=${ALLOC_GPUS} \
    --memory=${ALLOC_RAMS} \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=${SERVER_RAY_DASHBOARD_PORT} \
    --metrics-export-port=${SERVER_RAY_METRICS_EXPORT_PORT} \
    --disable-usage-stats \
    --object-spilling-directory=/tmp/spill

make logs-ray