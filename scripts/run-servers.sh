#!/usr/bin/env bash

SESSION="optimization-servers"
tmux kill-session -t $SESSION 2>/dev/null

SERVER_JL_PORT=${1}
SERVER_PY_PORT=${2}
BE_NATIVE=${3:-false}

source .venv/bin/activate
direnv allow

commands=(
  "make run-server-jl SERVER_JL_PORT=${SERVER_JL_PORT}; sleep 3"
)
if [[ "$BE_NATIVE" == "true" ]]; then
    commands+=("echo PYTHONPATH: $PYTHONPATH && echo GRB_LICENCE_FILE: $GRB_LICENSE_FILE && make run-server-py-native SERVER_PY_PORT=${SERVER_PY_PORT}; sleep 3")
else
    commands+=("echo PYTHONPATH: $PYTHONPATH && echo GRB_LICENCE_FILE: $GRB_LICENSE_FILE && make run-server-py SERVER_PY_PORT=${SERVER_PY_PORT}; sleep 3")
fi
commands+=("make run-server-grafana; sleep 3")
commands+=("make run-server-mongodb; sleep 3")
if [[ "$BE_NATIVE" == "true" ]]; then
    commands+=("make run-server-ray-native; sleep 3")
else
    commands+=("make run-server-ray; sleep 3")
fi
commands+=("ssh -t mohammad@${WORKER_HOST} 'read -p \"Press Enter to continue...\"; mkdir -p spill; cd projects/dig-a-plan-monorepo/optimization; make run-ray-worker; bash'")
commands+=("./scripts/run-interactive.sh; sleep 3")

tmux new-session -d -s $SESSION

for ((i=1; i<${#commands[@]}; i++)); do
    tmux split-window -v -t $SESSION:0
    tmux select-layout -t $SESSION:0 tiled
done

for i in "${!commands[@]}"; do
    tmux send-keys -t $SESSION:0.$i "${commands[$i]}" C-m
    sleep 0.5
done

# Optionally set titles
titles=(
  "JL Server"
  "PY Server"
  "Grafana"
  "MongoDB"
  "Ray Server"
  "Ray Worker"
  "Interactive"
)

for i in "${!titles[@]}"; do
    tmux select-pane -t $SESSION:0.$i -T "${titles[$i]}"
done

tmux select-pane -t $SESSION:0.4

tmux set-option -g pane-border-status top
tmux set-option -g pane-border-format "#{pane_title}"

tmux select-layout -t $SESSION:0 even-vertical

tmux attach -t $SESSION
