#!/bin/bash
set -e

options=(
    "Run Expansion (Quick Integration Test)"
    "Run Expansion for simple grid"
    "Run Expansion for Simplified Boisy grid (it takes time)"
    "Run tests"
    "Exit"
)

PS3="Please enter your choice: "

while true; do
    select opt in "${options[@]}"; do
        case "$REPLY" in
            1)
                make run-expansion PAYLOAD=data/payloads/simple_grid_quick_test.json
                break
                ;;
            2)
                make run-expansion PAYLOAD=data/payloads/simple_grid.json
                break
                ;;
            3)
                make run-expansion PAYLOAD=data/payloads/simplified_boisy_grid.json
                break
                ;;
            4)
                make run-tests
                break
                ;;
            5)
                echo "Exiting..."
                exit 0
                ;;
            *)
                echo "Invalid option, try again."
                break
                ;;
        esac
    done
done
