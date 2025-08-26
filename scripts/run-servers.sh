#!/usr/bin/env bash

SERVER_JL_PORT=${1}
SERVER_PY_PORT=${2}

source .venv/bin/activate
direnv allow

SESSION="optimization-servers"

# Always attempt to kill both ports first
if [[ -n "$SERVER_JL_PORT" ]]; then
  fuser -k ${SERVER_JL_PORT}/tcp 2>/dev/null
fi
if [[ -n "$SERVER_PY_PORT" ]]; then
  fuser -k ${SERVER_PY_PORT}/tcp 2>/dev/null
fi

tmux kill-session -t $SESSION 2>/dev/null

commands=(
  "make run-server-jl SERVER_JL_PORT=${SERVER_JL_PORT}"
  "echo PYTHONPATH: $PYTHONPATH && echo GRB_LICENCE_FILE: $GRB_LICENSE_FILE && make run-server-py SERVER_PY_PORT=${SERVER_PY_PORT}"
  "ray stop && make run-server-ray && ray status"
  "make run-server-grafana"
  "ssh -t mohammad@10.192.189.173"
)

# Create a detached session with first pane
tmux new-session -d -s $SESSION

# Split 4 more times → total 5 panes
for i in {1..4}; do
  tmux split-window -h -t $SESSION:0
done

# Arrange evenly
tmux select-layout -t $SESSION:0 even-vertical


sleep 1.0

# Send commands to each pane
for i in "${!commands[@]}"; do
  tmux send-keys -t $SESSION:0.$i "${commands[$i]}" C-m
done

# Attach once
tmux attach -t $SESSION
