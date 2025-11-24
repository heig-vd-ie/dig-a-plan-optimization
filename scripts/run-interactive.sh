#!/bin/bash
set -e

main_options=(
    "Run Expansion"
    "Run Tests"
    "Manage MongoDB"
    "Exit"
)

expansion_cases=(
    "Quick Integration Test"
    "Simple Grid"
    "Simplified Boisy Grid"
    "Boisy Grid"
    "Simplified Estavayer Grid"
    "Back"
)

simulations=(
    "Expectation"
    "Worst Case"
    "Wasserstein"
    "All"
    "Back"
)

mongodb_options=(
    "Sync MongoDB"
    "Delete MongoDB"
    "Backup MongoDB"
    "Restore MongoDB"
    "Chunk Large Files"
    "Back"
)

PS3="Please enter your choice: "

while true; do
    echo "Main Menu:"
    select main_opt in "${main_options[@]}"; do
        case "$REPLY" in
            1)
                # Expansion menu
                while true; do
                    echo "Choose Expansion Case:"
                    select case_opt in "${expansion_cases[@]}"; do
                        case "$REPLY" in
                            1) payload="examples/payloads/simple_grid_quick_test.json";;
                            2) payload="examples/payloads/simple_grid.json";;
                            3) payload="examples/payloads/simplified_boisy_grid.json";;
                            4) payload="examples/payloads/boisy_grid.json";;
                            5) payload="examples/payloads/simplified_estavayer_grid.json";;
                            6) payload="examples/payloads/estavayer_grid.json";;
                            7) break 2;;  # back to main menu
                            *) echo "Invalid option"; continue;;
                        esac

                        # Simulation menu
                        while true; do
                            echo "Choose Simulation:"
                            select sim_opt in "${simulations[@]}"; do
                                case "$REPLY" in
                                    1)
                                        make run-expansion PAYLOAD="$payload"
                                        break 3
                                        ;;
                                    2)
                                        tmp_payload=$(mktemp)
                                        jq '.sddp_params.risk_measure_type = "WorstCase"' "$payload" > "$tmp_payload"
                                        make run-expansion PAYLOAD="$tmp_payload"
                                        rm "$tmp_payload"
                                        break 3
                                        ;;
                                    3)
                                        tmp_payload=$(mktemp)
                                        jq '.sddp_params.risk_measure_type = "Wasserstein"' "$payload" > "$tmp_payload"
                                        make run-expansion PAYLOAD="$tmp_payload"
                                        rm "$tmp_payload"
                                        break 3
                                        ;;
                                    4)
                                        # Run all simulations sequentially
                                        tmp_payload=$(mktemp)
                                        cp "$payload" "$tmp_payload"
                                        make run-expansion PAYLOAD="$tmp_payload"
                                        latest_folder=$(ls -dt .cache/algorithm/*/ | head -n 1)
                                        cut_file="${latest_folder%/}/bender_cuts.json"
                                        rm "$tmp_payload"

                                        tmp_payload=$(mktemp)
                                        jq --arg r "WorstCase" --arg i "0" --arg p "0.1" \
                                            '.sddp_params.risk_measure_type = $r | .iterations = ($i|tonumber) | .sddp_params.risk_measure_param = $p' \
                                            "$payload" > "$tmp_payload"
                                        make run-expansion-with-cut PAYLOAD="$tmp_payload" CUT_FILE="$cut_file"
                                        rm "$tmp_payload"

                                        for risk_param in "0.02" "0.05" "0.1" "0.2"; do
                                            tmp_payload=$(mktemp)
                                            jq --arg r "Wasserstein" --arg i "0" --arg p "$risk_param" \
                                                '.sddp_params.risk_measure_type = $r | .iterations = ($i|tonumber) | .sddp_params.risk_measure_param = $p' \
                                                "$payload" > "$tmp_payload"
                                            make run-expansion-with-cut PAYLOAD="$tmp_payload" CUT_FILE="$cut_file"
                                            rm "$tmp_payload"
                                        done
                                        break 3
                                        ;;
                                    5) break 3;;  # back to expansion case menu
                                    *) echo "Invalid option";;
                                esac
                            done
                        done
                    done
                done
                ;;
            2)
                make run-tests
                break
                ;;
            3)
                # MongoDB menu
                while true; do
                    echo "MongoDB Options:"
                    select mongo_opt in "${mongodb_options[@]}"; do
                        case "$REPLY" in
                            1) 
                                make sync-mongodb
                                break 2
                                ;;
                            2) 
                                make clean-mongodb
                                break 2
                                ;;
                            3) 
                                .venv/bin/python scripts/mongo-backup-tools.py --db optimization --backup
                                break 2
                                ;;
                            4) 
                                read -p "Enter backup path: " backup_path
                                .venv/bin/python scripts/mongo-backup-tools.py --db optimization --restore "$backup_path"
                                break 2
                                ;;
                            5) 
                                .venv/bin/python scripts/mongo-tools.py --chunk
                                break 2
                                ;;
                            6) break;;  # back to main menu
                            *) echo "Invalid option";;
                        esac
                    done
                    break
                done
                ;;
            4)
                echo "Exiting..."
                exit 0
                ;;
            *)
                echo "Invalid option, try again."
                ;;
        esac
        break
    done
done
