#!/bin/bash
set -e

default_worker_host="mohammad@100.82.212.62"
read -r -p "Enter ssh worker address (e.g., ${default_worker_host}): " WORKER_HOST
WORKER_HOST="${WORKER_HOST:-$default_worker_host}"

read -r -p "Press Enter to start deployment on ${WORKER_HOST}..."

ssh -tt "$WORKER_HOST" 'bash -s' < ./scripts/start-ray-worker.sh
