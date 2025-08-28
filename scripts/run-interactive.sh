#!/bin/bash
set -e

main_options=(
    "Run Expansion"
    "Run tests"
    "Sync MongoDB"
    "Delete MongoDB data"
    "Exit"
)

expansion_cases=(
    "Quick Integration Test"
    "Simple Grid"
    "Simplified Boisy Grid"
    "Boisy Grid"
    "Back"
)

simulations=(
    "Expectation"
    "Worst Case"
    "Wasserstein"
    "Back"
)

PS3="Please enter your choice: "

while true; do
    echo "Main Menu:"
    select opt in "${main_options[@]}"; do
        case "$REPLY" in
            1)
                # Expansion menu
                echo "Choose Expansion Case:"
                select case_opt in "${expansion_cases[@]}"; do
                    case "$REPLY" in
                        1) payload="data/payloads/simple_grid_quick_test.json";;
                        2) payload="data/payloads/simple_grid.json";;
                        3) payload="data/payloads/simplified_boisy_grid.json";;
                        4) payload="data/payloads/boisy_grid.json";;
                        5) break 2;;  # back to main menu
                        *) echo "Invalid option"; continue;;
                    esac

                    # Simulation menu
                    echo "Choose Simulation:"
                    select sim_opt in "${simulations[@]}"; do
                        case "$REPLY" in
                            1)
                                make run-expansion PAYLOAD="$payload"
                                break 2
                                ;;
                            2)
                                tmp_payload=$(mktemp)
                                jq '.sddp_params.risk_measure_type = "WorstCase"' "$payload" > "$tmp_payload"
                                make run-expansion PAYLOAD="$tmp_payload"
                                break 2
                                ;;
                            3)
                                tmp_payload=$(mktemp)
                                jq '.sddp_params.risk_measure_type = "Wasserstein"' "$payload" > "$tmp_payload"
                                make run-expansion PAYLOAD="$tmp_payload"
                                break 2
                                ;;
                            4) break;;  # back to expansion menu
                            *) echo "Invalid option";;
                        esac
                    done
                done
                break
                ;;
            2)
                make run-tests
                break
                ;;
            3)
                make sync-mongodb
                break
                ;;
            4)
                make clean-mongodb
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
