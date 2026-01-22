#!/bin/bash
set -euo pipefail
default_head_host_ip="100.66.5.3"
read -r -p "Enter head host IP (e.g., ${default_head_host_ip}): " HEAD_HOST
HEAD_HOST="${HEAD_HOST:-$default_head_host_ip}"

: "${SERVER_RAY_PORT:?Need to set SERVER_RAY_PORT}"


POLARS_SKIP_CPU_CHECK=1 ray start \
  --address="${HEAD_HOST}:${SERVER_RAY_PORT}"

exec watch -n 5 ray status
