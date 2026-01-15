#!/bin/bash
set -e

main_options=(
    "Run Tests"
    "Manage MongoDB"
    "Exit"
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
                make run-tests
                break
                ;;
            2)
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
            3)
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
