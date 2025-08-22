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

tmux new-session -d -s $SESSION

tmux send-keys -t $SESSION "make run-server-jl SERVER_JL_PORT=${SERVER_JL_PORT}" C-m
tmux split-window -v -t $SESSION
tmux send-keys -t $SESSION "echo PYTHONPATH: $PYTHONPATH && echo GRB_LICENCE_FILE: $GRB_LICENSE_FILE && make run-server-py SERVER_PY_PORT=${SERVER_PY_PORT}" C-m
tmux split-window -v -t $SESSION
tmux send-keys -t $SESSION "ray stop && make run-server-ray && ray status" C-m

tmux select-pane -t 1
tmux attach -t $SESSION