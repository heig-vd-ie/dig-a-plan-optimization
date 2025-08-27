#!/usr/bin/env bash

SERVER_JL_PORT=${1}
SERVER_PY_PORT=${2}
BE_NATIVE=${3:-false}

source .venv/bin/activate
direnv allow

SESSION="optimization-servers"

if [[ -n "$SERVER_JL_PORT" ]]; then
  fuser -k ${SERVER_JL_PORT}/tcp 2>/dev/null
fi
if [[ -n "$SERVER_PY_PORT" ]]; then
  fuser -k ${SERVER_PY_PORT}/tcp 2>/dev/null
fi

tmux kill-session -t $SESSION 2>/dev/null

commands=(
  "make run-server-jl SERVER_JL_PORT=${SERVER_JL_PORT}"
)

# Conditionally append
if [[ "$BE_NATIVE" == "true" ]]; then
    commands+=("echo PYTHONPATH: $PYTHONPATH && echo GRB_LICENCE_FILE: $GRB_LICENSE_FILE && make run-server-py-native SERVER_PY_PORT=${SERVER_PY_PORT}")
else
    commands+=("echo PYTHONPATH: $PYTHONPATH && echo GRB_LICENCE_FILE: $GRB_LICENSE_FILE && make run-server-py SERVER_PY_PORT=${SERVER_PY_PORT}")
fi
commands+=("make run-server-ray")
commands+=("make run-server-grafana")
commands+=("ssh -t mohammad@${WORKER_HOST} 'read -p \"Press Enter to continue...\"; mkdir -p /tmp/spill; cd projects/dig-a-plan-monorepo/optimization; make run-ray-worker; bash'")
commands+=("./scripts/run-interactive.sh")


tmux new-session -d -s $SESSION

for i in {1..5}; do
  tmux split-window -h -t $SESSION:0
done

tmux select-layout -t $SESSION:0 even-vertical

sleep 1.0

for i in "${!commands[@]}"; do
  tmux send-keys -t $SESSION:0.$i "${commands[$i]}" C-m
done

titles=(
  "JL Server"
  "PY Server"
  "Ray Server"
  "Grafana"
  "Ray Worker"
  "Interactive"
)
for i in "${!titles[@]}"; do
    tmux select-pane -t $SESSION:0.$i -T "${titles[$i]}"
done

tmux select-pane -t $SESSION:0.4

tmux set-option -g pane-border-status top
tmux set-option -g pane-border-format "#{pane_title}"

tmux attach -t $SESSION
