# Makefile for Dig-A-Plan setup (supports Linux and WSL)
include Makefile.common.mak

DATA_EXPORTER_REPO := data-exporter
DATA_EXPORTER_BRANCH := main
DATA_EXPORTER_VERSION := 0.1.0
TWINDIGRID_REPO := digrid-schema
TWINDIGRID_BRANCH := main
TWINDIGRID_VERSION := 0.5.0
UTILITY_FUNCTIONS_REPO := utility-functions
UTILITY_FUNCTIONS_BRANCH := main
UTILITY_FUNCTIONS_VERSION := 0.1.0

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

run-server-grafana: ## Build & start Grafana server (use GRAFANA_PORT=xxxx to specify port)
	@echo "Building custom Grafana image..."
	@docker buildx build -t custom-grafana -f grafana/Dockerfile grafana
	@echo "Starting Grafana server on port $${GRAFANA_PORT:-4000}..."
	@docker rm -f ray-grafana >/dev/null 2>&1 || true
	@docker run -d --network host \
	    -e GF_SERVER_HTTP_PORT=$${GRAFANA_PORT:-4000} \
		--name ray-grafana \
		custom-grafana
	@echo "Grafana running → http://localhost:$${GRAFANA_PORT:-4000}"
	@docker logs -f ray-grafana

install-jl:  ## Install Julia
	@echo "Installing Julia..."
	@bash scripts/install-julia.sh

show-host-ip: ## Show current IP value
	@echo "HEAD_HOST: $(HEAD_HOST)"

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
	julia -e 'using JuliaFormatter; format("src/")'

format: format-jl format-py ## Format all code (Julia and Python)

run-server-jl: ## Start Julia API server (use SERVER_PORT=xxxx to specify port)
	@echo "Starting Julia API server on localhost:$(SERVER_JL_PORT)..."
	julia --project=src/model_expansion/. src/model_expansion/src/Server.jl $(SERVER_JL_PORT)

run-server-py-native: ## Start Python API server natively (use SERVER_PORT=xxxx to specify port)
	@echo "Starting Python API server on localhost:$(SERVER_PY_PORT)..."
	PYTHONPATH=src uvicorn main:app --host 0.0.0.0 --port $(SERVER_PY_PORT) --reload

run-server-py: ## Start Python API server in Docker (use SERVER_PORT=xxxx to specify port)
	@echo "Building Docker image..."
	@docker rm -f dap-py-api >/dev/null 2>&1 ||	true
	@docker build -t dap-py-api -f Dockerfile .
	@docker run -it \
		--network host \
	  -e SERVER_RAY_ADDRESS=$(SERVER_RAY_ADDRESS) \
	  -p $(SERVER_PY_PORT):$(SERVER_PY_PORT) \
	  -v $(GRB_LICENSE_FILE):/licenses/GRB_LICENCE_FILE:ro \
	  -v /tmp/spill:/tmp/spill \
	  -v /tmp/ray:/tmp/ray \
	  -v .cache:/app/.cache:rw \
	  --name dap-py-api \
	  dap-py-api
	@docker logs -f dap-py-api

run-server-ray-native: ## Start Ray server natively
	@echo "Starting Ray head node natively on $(SERVER_RAY_ADDRESS)"
	@./scripts/start-ray-head.sh \
		$(SERVER_RAY_PORT) \
		$(SERVER_RAY_DASHBOARD_PORT) \
		$(SERVER_RAY_METRICS_EXPORT_PORT) \
		$(ALLOC_CPUS) \
		$(ALLOC_GPUS) \
		$(ALLOC_RAMS) \
		$(GRAFANA_PORT) \
		true
	@ray status

run-server-ray: ## Start Ray server
	@echo "Starting Ray head node on $(SERVER_RAY_ADDRESS)"
	@./scripts/start-ray-head.sh \
		$(SERVER_RAY_PORT) \
		$(SERVER_RAY_DASHBOARD_PORT) \
		$(SERVER_RAY_METRICS_EXPORT_PORT) \
		$(ALLOC_CPUS) \
		$(ALLOC_GPUS) \
		$(ALLOC_RAMS) \
		$(GRAFANA_PORT) \
		false
	@docker exec -it dap-py-api ray status

run-ray-worker:  ## Start Ray worker
	@echo -n "Run Worker?..."
	@read dummy
	@direnv allow && \
	echo "Starting Ray worker natively connecting to $$HEAD_HOST:$$SERVER_RAY_PORT" && \
	./scripts/start-ray-worker.sh

run-server-mongodb: ## Start MongoDB server
	@echo "Starting MongoDB server..."
	@docker pull mongo:latest
	@docker run -d \
		--name mongo-db \
		-p $(SERVER_MONGODB_PORT):$(SERVER_MONGODB_PORT) \
		-v /path/on/host/mongo-data:/data/db \
		mongo:latest
	@echo "MongoDB running → http://localhost:$(SERVER_MONGODB_PORT)"
	@docker logs -f mongo-db

run-all-native: ## Start all servers
	@echo "Starting Julia, Python API, and Ray servers..."
	@$(MAKE) stop
	@bash ./scripts/run-servers.sh $(SERVER_JL_PORT) $(SERVER_PY_PORT) true

run-all: ## Start all servers
	@echo "Starting all servers..."
	@$(MAKE) stop
	@bash ./scripts/run-servers.sh $(SERVER_JL_PORT) $(SERVER_PY_PORT) false

fetch-all:  ## Fetch all dependencies
	@$(MAKE) fetch-wheel \
		REPO=$(DATA_EXPORTER_REPO) \
		BRANCH=$(DATA_EXPORTER_BRANCH) \
		VERSION=$(DATA_EXPORTER_VERSION)
	@$(MAKE) fetch-wheel \
		REPO=$(TWINDIGRID_REPO) \
		BRANCH=$(TWINDIGRID_BRANCH) \
		VERSION=$(TWINDIGRID_VERSION)
	@$(MAKE) fetch-wheel \
		REPO=$(UTILITY_FUNCTIONS_REPO) \
		BRANCH=$(UTILITY_FUNCTIONS_BRANCH) \
		VERSION=$(UTILITY_FUNCTIONS_VERSION)

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
	@$(MAKE) kill-port PORT=$(SERVER_JL_PORT)
	@$(MAKE) kill-port PORT=$(SERVER_PY_PORT)
	@$(MAKE) kill-port PORT=$(SERVER_RAY_PORT)
	@$(MAKE) kill-port PORT=$(SERVER_RAY_DASHBOARD_PORT)
	@$(MAKE) kill-port PORT=$(GRAFANA_PORT)
	@$(MAKE) kill-port PORT=$(SERVER_MONGODB_PORT)

clean-ray:  ## Clean up Ray processes and files
	@echo "Stopping all Ray processes..."
	@ray stop --force || true
	@echo "Removing Ray session folders..."
	@rm -rf /tmp/ray/session_* || true
	@echo "Cleaned Ray."

stop: ## Kill all Ray processes and clean Docker
	@echo "Killing all Ray processes..."
	@$(MAKE) clean-ray  || true
	@echo "Stopping all Docker containers..."
	@if [ -n "$$(docker ps -aq)" ]; then docker stop $$(docker ps -aq); else echo "No Docker containers to stop."; fi
	@echo "Removing all Docker containers..."
	@if [ -n "$$(docker ps -aq)" ]; then docker rm $$(docker ps -aq); else echo "No Docker containers to remove."; fi
	@$(MAKE) kill-ports-all
	@echo "Shutting down Prometheus metrics..."
	@ray metrics shutdown-prometheus || true
	@echo "Cleaning up local Prometheus files..."
	@sudo rm -rf prometheus-* || true
	@$(MAKE) fix-cache-permissions

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
		"http://localhost:$(SERVER_PY_PORT)/expansion?with_ray=$(USE_RAY)"
	@$(MAKE) sync-mongodb

run-expansion-with-cut: ## Curl expansion for a payload with cut_file
	@echo "Triggering expansion..."
	@curl -X PATCH \
		-H "Content-Type: application/json" \
		--data-binary @$(PAYLOAD) \
		"http://localhost:$(SERVER_PY_PORT)/expansion?with_ray=$(USE_RAY)&cut_file=$(CUT_FILE)"
	@$(MAKE) sync-mongodb

# Run with: make sync-mongodb FORCE=true
sync-mongodb: ## Sync data from MongoDB
	@echo "Syncing data from MongoDB..."
	@python ./scripts/mongo-tools.py $(if $(FORCE:-false),--force)

clean-mongodb:  ## Clean up MongoDB data
	@echo "Cleaning MongoDB data..."
	@python ./scripts/mongo-tools.py --delete
