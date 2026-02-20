# Makefile for Dig-A-Plan setup (supports Linux and WSL)
include Makefile.common.mak


IMAGES := dap-py-api custom-grafana
CACHE_FOLDER := .cache

clean: ## Clean ignored files
	@echo "Cleaning up ignored files..."
	@echo "This will remove all ignored files except .cache. Are you sure? (y/n)"
	@bash -c 'read -r answer; if [ "$$answer" != "y" ]; then echo "Aborted."; exit 1; fi'
	@git clean -fdX -e .cache
	@echo "Cleaned."

install-all:  ## Install all dependencies and set up the environment
	@$(MAKE) install-basics
	@$(MAKE) install-grafana
	@$(MAKE) install-jl
	@$(MAKE) install-docker

install-grafana: ## Install Grafana
	@echo "Installing Grafana..."
	@sudo apt update
	@sudo apt install -y grafana

install-jl:  ## Install Julia
	@echo "Installing Julia..."
	@bash scripts/install-julia.sh

show-current-specs: ## Show current host IP value
	@echo "CURRENT_HOST: $(CURRENT_HOST)"
	@echo "CURRENT_CPUS: $(CURRENT_CPUS)"
	@echo "CURRENT_RAMS: $(CURRENT_RAMS)"
	@echo "CURRENT_GPUS: $(CURRENT_GPUS)"

install-docker: ## Install Docker
	@echo "Installing Docker..."
	@bash scripts/install-docker.sh

run-tests-jl:  ## Run tests of Julia
	@echo "Running Julia tests..."
	julia --project=src/model_expansion/. src/model_expansion/test/runtests.jl

run-tests: ## Run all tests
	@$(MAKE) run-tests-py
	@$(MAKE) run-tests-jl

format-jl:  ## Format Julia code in the src directory
	@echo "Formatting Julia code with JuliaFormatter..."
	julia --project=src/model_expansion/. -e 'using JuliaFormatter; format("src/")'

format-py: ## Format Python code using black
	@echo "Formatting Python code with black..."
	@poetry run black .

format: format-jl format-py ## Format all code (Julia and Python)

run-ray-worker:  ## Start Ray worker
	@direnv allow && \
	echo "Starting Ray worker natively connecting to HEAD_HOST_IP:$$SERVER_RAY_PORT" && \
	./scripts/start-ray-worker.sh

make connect-ray-worker:  ## Connect to Ray worker via SSH
	@direnv allow && \
	echo "Connecting to Ray worker via SSH..." && \
	./scripts/connect-ray-worker.sh

start: ## Start all servers
	@echo "Starting all servers..."
	@bash ./scripts/start-servers.sh

build: ## Build Docker images
	@echo "Building Docker images..."
	@export DOCKER_BUILDKIT=1 && export COMPOSE_DOCKER_CLI_BUILD=1 && cd dockerfiles && docker compose -p optimization build

stop:  ## Stop Docker containers from specific images
	@echo "Stopping Docker containers for: $(IMAGES)"
	@cd dockerfiles && docker compose -p optimization down || true

remove:  ## Remove Docker containers from specific images
	@echo "Removing Docker containers for: $(IMAGES)"
	@cd dockerfiles && docker compose -p optimization rm -f || true

kill-port: ## Kill process running on specified port (PORT)
	@echo "Killing process on port $(PORT)..."
	@PID=$$(lsof -t -i :$(PORT)) || true; \
	if [ -n "$$PID" ]; then \
		echo "Found process $$PID on port $(PORT), killing it..."; \
		kill -9 $$PID; \
	else \
		echo "No process found on port $(PORT)"; \
	fi

kill-ports-all: ## Kill all processes running on specified ports
	for port in $(SERVER_JL_PORT) $(SERVER_PY_PORT) $(SERVER_RAY_PORT) $(SERVER_RAY_DASHBOARD_PORT) $(GRAFANA_PORT) $(SERVER_MONGODB_PORT); do \
		$(MAKE) kill-port PORT=$$port; \
	done

clean-ray:  ## Clean up Ray processes and files
	@echo "Stopping all Ray processes..."
	@ray stop --force || true
	@echo "Removing Ray session folders..."
	@rm -rf /tmp/ray/session_* || true
	@echo "Cleaned Ray."

shutdown-prometheus: ## Shutdown Prometheus and clean up files
	@echo "Shutting down Prometheus..."
	@ray metrics shutdown-prometheus || true
	@sudo rm -rf prometheus-* || true

fix-cache-permissions: ## Fix permissions of the CACHE_FOLDER folder
	@echo "Fixing permissions for a folder ..."
	@sudo chown -R $(USER):$(USER) $(CACHE_FOLDER) || true
	@sudo chmod -R 775 $(CACHE_FOLDER) || true
	@echo "Done."

permit-remote-ray-port: ## Permit remote access to Ray server
	@echo "Permitting remote access to Ray server on port $(SERVER_RAY_PORT)..."
	sudo ufw allow $(SERVER_RAY_PORT)

EXPERIMENT := all

sync-pg: ## Sync data from PostgreSQL
	@echo "Syncing data from PostgreSQL..."
	$(MAKE) fix-cache-permissions
	@.venv/bin/python ./experiments/expansion_planning_result_sync.py --sync --experiment geolocations
	@.venv/bin/python ./experiments/expansion_planning_result_sync.py --reset --sync --force --experiment $(EXPERIMENT)
