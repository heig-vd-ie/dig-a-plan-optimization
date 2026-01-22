#!/usr/bin/env bash

SESSION="optimization-servers"
tmux kill-session -t $SESSION 2>/dev/null

source .venv/bin/activate
direnv allow

commands=("source .envrc && cd dockerfiles && docker compose -p optimization up -d; sleep 3; docker compose -p optimization logs -f")
commands+=("make run-ray-worker; sleep 3")
commands+=("make venv-activate; sleep 3")

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
  "Servers"
  "Ray Worker"
  "Interactive"
)

for i in "${!titles[@]}"; do
    tmux select-pane -t $SESSION:0.$i -T "${titles[$i]}"
done

tmux select-pane -t $SESSION:0.2

tmux set-option -g pane-border-status top
tmux set-option -g pane-border-format "#{pane_title}"

tmux select-layout -t $SESSION:0 even-vertical

tmux attach -t $SESSION
