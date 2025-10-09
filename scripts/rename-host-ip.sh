#!/bin/bash
set -e

HOSTNAME="${1}"
CONF_FILE="/etc/avahi/avahi-daemon.conf"

# Ensure Avahi is installed
if ! command -v avahi-daemon >/dev/null 2>&1; then
  echo "Installing avahi-daemon..."
  sudo apt update -qq
  sudo apt install -y avahi-daemon
fi

# Enable and start the service
sudo systemctl enable --now avahi-daemon

# Check if hostname already exists in config
if grep -q "^host-name=${HOSTNAME}$" "$CONF_FILE"; then
  echo "Hostname '${HOSTNAME}' already exists in ${CONF_FILE}, skipping modification."
else
  echo "Adding hostname '${HOSTNAME}' to ${CONF_FILE}..."
  sudo sed -i "/^\[server\]/a host-name=${HOSTNAME}" "$CONF_FILE"
  sudo systemctl restart avahi-daemon
  echo "Hostname '${HOSTNAME}' added and avahi-daemon restarted."
fi
