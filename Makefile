# Makefile for Dig-A-Plan setup (supports Linux and WSL)
include Makefile.common.mak

CURRENT_HOST ?= $(shell hostname -I | awk '{print $$1}')
CURRENT_CPUS ?= $(shell nproc)
CURRENT_RAMS ?= $(shell free -m | awk '/^Mem:/{print $$2}')
CURRENT_GPUS ?= $(shell nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

DATA_EXPORTER_REPO := data-exporter
DATA_EXPORTER_BRANCH := main
DATA_EXPORTER_VERSION := 0.1.0
TWINDIGRID_REPO := digrid-schema
TWINDIGRID_BRANCH := main
TWINDIGRID_VERSION := 0.5.0
UTILITY_FUNCTIONS_REPO := utility-functions
UTILITY_FUNCTIONS_BRANCH := main
UTILITY_FUNCTIONS_VERSION := 0.1.0


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

run-server-py: ## Start Python API server (use SERVER_PORT=xxxx to specify port)
	@echo "Starting Python API server on localhost:$(SERVER_PY_PORT)..."
	PYTHONPATH=src uvicorn main:app --host 0.0.0.0 --port $(SERVER_PY_PORT) --reload

run-server-ray: ## Start Ray server
	@echo "Starting Ray server..."
	@ray start --head --port=$(SERVER_RAY_PORT) --num-cpus=$(SERVER_RAY_CPUS) --num-gpus=$(SERVER_RAY_GPUS)  --dashboard-host=$(CURRENT_HOST) --dashboard-port=$(SERVER_RAY_DASHBOARD_PORT) --disable-usage-stats

run-ray-worker: ## Remote Ray worker
	@echo "Starting remote Ray worker..."
	ray start --address=$(HEAD_HOST):$(SERVER_RAY_PORT)  --node-ip-address=$(CURRENT_HOST) --num-cpus=$(CURRENT_CPUS) --num-gpus=$(CURRENT_GPUS)

run-servers: ## Start both Julia and Python API servers
	@echo "Starting Julia, Python API, and Ray servers..."
	@$(MAKE) stop
	@bash ./scripts/run-servers.sh $(SERVER_JL_PORT) $(SERVER_PY_PORT)

fetch-all:  ## Fetch all dependencies
	@$(MAKE) fetch-wheel REPO=$(DATA_EXPORTER_REPO) BRANCH=$(DATA_EXPORTER_BRANCH) VERSION=$(DATA_EXPORTER_VERSION)
	@$(MAKE) fetch-wheel REPO=$(TWINDIGRID_REPO) BRANCH=$(TWINDIGRID_BRANCH) VERSION=$(TWINDIGRID_VERSION)
	@$(MAKE) fetch-wheel REPO=$(UTILITY_FUNCTIONS_REPO) BRANCH=$(UTILITY_FUNCTIONS_BRANCH) VERSION=$(UTILITY_FUNCTIONS_VERSION)

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

stop: ## Kill all Ray processes
	@echo "Killing all Ray processes..."
	@ray stop || true
	@$(MAKE) kill-ports-all


permit-remote-ray-port: ## Permit remote access to Ray server
	@echo "Permitting remote access to Ray server on port $(SERVER_RAY_PORT)..."
	sudo ufw allow $(SERVER_RAY_PORT)

