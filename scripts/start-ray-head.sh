#!/bin/bash
set -e

SERVER_RAY_PORT=${1}
SERVER_RAY_DASHBOARD_PORT=${2}
SERVER_RAY_METRICS_EXPORT_PORT=${3}
ALLOC_CPUS=${4}
ALLOC_GPUS=${5}
ALLOC_RAMS=${6}
GRAFANA_PORT=${7}
RUN_NATIVE=${8}

# echo -n "Run Server (Wait for Python server to be UP)?..."
# read dummy

# Launch Prometheus metrics for Ray
ray metrics launch-prometheus

# Set environment variables for Grafana/Prometheus
export RAY_GRAFANA_HOST="http://${LOCAL_HOST}:${GRAFANA_PORT}"
export RAY_GRAFANA_IFRAME_HOST="http://${LOCAL_HOST}:${GRAFANA_PORT}"
export RAY_GRAFANA_ORG_ID=1
export RAY_PROMETHEUS_HOST="http://${LOCAL_HOST}:9090"
export RAY_PROMETHEUS_NAME="Prometheus"

mkdir -p ray
mkdir -p spill
sudo chown -R $(id -u):$(id -g) ray
chmod -R 777 ray
sudo chown -R $(id -u):$(id -g) spill
chmod -R 777 spill

# Start Ray head
ray stop
ray start --head \
    --port=${SERVER_RAY_PORT} \
    --num-cpus=${ALLOC_CPUS} \
    --num-gpus=${ALLOC_GPUS} \
    --memory=${ALLOC_RAMS} \
    --dashboard-host=${LOCAL_HOST} \
    --dashboard-port=${SERVER_RAY_DASHBOARD_PORT} \
    --metrics-export-port=${SERVER_RAY_METRICS_EXPORT_PORT} \
    --disable-usage-stats \
    --object-spilling-directory=spill
watch -n 5 ray status
