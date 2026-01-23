#!/bin/bash
set -euo pipefail

default_head_host_ip="100.66.5.3"
read -r -p "Enter head host IP (e.g., ${default_head_host_ip}): " HEAD_HOST
HEAD_HOST="${HEAD_HOST:-$default_head_host_ip}"

source .venv/bin/activate
eval "$(direnv export bash)"

: "${SERVER_RAY_PORT:?Need to set SERVER_RAY_PORT}"
: "${ALLOC_CPUS:?Need to set ALLOC_CPUS}"
: "${ALLOC_GPUS:?Need to set ALLOC_GPUS}"
: "${ALLOC_RAMS:?Need to set ALLOC_RAMS}"

mkdir -p spill

NODE_IP="$(hostname -I | awk '{print $1}')"

POLARS_SKIP_CPU_CHECK=1 ray start \
  --address="${HEAD_HOST}:${SERVER_RAY_PORT}" \
  --node-ip-address="$NODE_IP" \
  --num-cpus="${ALLOC_CPUS}" \
  --num-gpus="${ALLOC_GPUS}" \
  --memory="${ALLOC_RAMS}" \
  --object-spilling-directory=spill

exec watch -n 5 ray status
