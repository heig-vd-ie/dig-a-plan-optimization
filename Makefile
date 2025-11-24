# Makefile for Dig-A-Plan setup (supports Linux and WSL)
include Makefile.common.mak

DATA_EXPORTER_REPO := data-exporter
DATA_EXPORTER_BRANCH := main
DATA_EXPORTER_VERSION := 0.1.0
TWINDIGRID_REPO := digrid-schema
TWINDIGRID_BRANCH := main
TWINDIGRID_VERSION := 0.5.0


IMAGES := dap-py-api custom-grafana mongo:latest

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
	@$(MAKE) install-mongodb-gui

install-grafana: ## Install Grafana
	@echo "Installing Grafana..."
	@sudo apt update
	@sudo apt install -y grafana

install-mongodb-gui: ## Install MongoDB GUI (MongoDB Compass)
	@echo "Installing MongoDB Compass..."
	@wget https://downloads.mongodb.com/compass/mongodb-compass_1.46.8_amd64.deb
	@sudo dpkg -i mongodb-compass_1.46.8_amd64.deb
	@sudo apt-get install -f
	@rm mongodb-compass_1.46.8_amd64.deb

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
	@echo -n "Run Worker?..."
	@read dummy
	@direnv allow && \
	echo "Starting Ray worker natively connecting to $$HEAD_HOST:$$SERVER_RAY_PORT" && \
	./scripts/start-ray-worker.sh

start: ## Start all servers
	@echo "Starting all servers..."
	@bash ./scripts/run-servers.sh

docker-build: ## Build Docker images
	@echo "Building Docker images..."
	@export DOCKER_BUILDKIT=1 && export COMPOSE_DOCKER_CLI_BUILD=1 && docker compose build --ssh default

fetch-all:  ## Fetch all dependencies
	@$(MAKE) fetch-wheel \
		REPO=$(DATA_EXPORTER_REPO) \
		BRANCH=$(DATA_EXPORTER_BRANCH) \
		VERSION=$(DATA_EXPORTER_VERSION)
	@$(MAKE) fetch-wheel \
		REPO=$(TWINDIGRID_REPO) \
		BRANCH=$(TWINDIGRID_BRANCH) \
		VERSION=$(TWINDIGRID_VERSION)

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

docker-stop:  ## Stop Docker containers from specific images
	@echo "Stopping Docker containers for: $(IMAGES)"
	@docker compose down || true

docker-remove:  ## Remove Docker containers from specific images
	@echo "Removing Docker containers for: $(IMAGES)"
	@docker compose rm -f || true

fix-cache-permissions: ## Fix permissions of the .cache/algorithm folder
	@echo "Fixing permissions for .cache/algorithm..."
	@sudo chown -R $(USER):$(USER) .cache/algorithm || true
	@sudo chmod -R 775 .cache/algorithm || true
	@echo "Done."

permit-remote-ray-port: ## Permit remote access to Ray server
	@echo "Permitting remote access to Ray server on port $(SERVER_RAY_PORT)..."
	sudo ufw allow $(SERVER_RAY_PORT)

run-expansion: ## Curl expansion for a payload
	@echo "Triggering expansion..."
	@curl -X PATCH \
		-H "Content-Type: application/json" \
		-d @$(PAYLOAD) \
		"http://$(LOCAL_HOST):${SERVER_PY_PORT}/expansion?with_ray=$(USE_RAY)"

run-expansion-with-cut: ## Curl expansion for a payload with cut_file
	@echo "Triggering expansion..."
	@curl -X PATCH \
		-H "Content-Type: application/json" \
		--data-binary @$(PAYLOAD) \
		"http://$(LOCAL_HOST):${SERVER_PY_PORT}/expansion?with_ray=$(USE_RAY)&cut_file=$(CUT_FILE)"

# Run with: make sync-mongodb FORCE=true
sync-mongodb: ## Sync data from MongoDB
	@echo "Syncing data from MongoDB..."
	@.venv/bin/python ./scripts/mongo-tools.py $(if $(FORCE),--force)

clean-mongodb:  ## Clean up MongoDB data
	@echo "Cleaning MongoDB data..."
	@.venv/bin/python ./scripts/mongo-tools.py --delete

chunk-mongodb: ## Chunk large files for MongoDB
	@echo "Chunking large files..."
	@.venv/bin/python ./scripts/mongo-tools.py --chunk
