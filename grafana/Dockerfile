FROM grafana/grafana:latest

# Copy config and dashboards
COPY grafana.ini /etc/grafana/grafana.ini
COPY dashboards/ /var/lib/grafana/dashboards/
COPY provisioning/ /etc/grafana/provisioning/
