#!/bin/bash
set -e

echo "Starting Ray Head node..."

# Launch Prometheus metrics for Ray
ray metrics launch-prometheus

# Ensure directories exist with proper permissions
mkdir -p /tmp/ray
mkdir -p /tmp/spill
chmod -R 777 /tmp/ray
chmod -R 777 /tmp/spill

# Stop any existing Ray processes
ray stop || true

# Start Ray head with environment variables
ray start --head \
    --port=${SERVER_RAY_PORT} \
    --num-cpus=${ALLOC_CPUS:-4} \
    --num-gpus=${ALLOC_GPUS:-0} \
    --memory=${ALLOC_RAMS:-8000000000} \
    --dashboard-host=${LOCAL_HOST:-0.0.0.0} \
    --dashboard-port=${SERVER_RAY_DASHBOARD_PORT} \
    --metrics-export-port=${SERVER_RAY_METRICS_EXPORT_PORT} \
    --disable-usage-stats \
    --object-spilling-directory=/tmp/spill \
    --block

echo "Ray head started successfully!"