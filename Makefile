# Makefile for Dig-A-Plan setup (supports Linux and WSL)
include Makefile.common.mak

SERVER_JL_PORT ?= 8080 # Julia server targets
SERVER_PY_PORT ?= 8000 # Python server targets
SERVER_RAY_PORT ?= 6379 # Ray server targets
NUM_CPUS ?= 10
NUM_GPUS ?= 1
DATA_EXPORTER_REPO := data-exporter
DATA_EXPORTER_BRANCH := main
DATA_EXPORTER_VERSION := 0.1.0
TWINDIGRID_REPO := digrid-schema
TWINDIGRID_BRANCH := main
TWINDIGRID_VERSION := 0.5.0
UTILITY_FUNCTIONS_REPO := utility-functions
UTILITY_FUNCTIONS_BRANCH := main
UTILITY_FUNCTIONS_VERSION := 0.1.0

install-julia:  ## Install Julia
	@echo "Installing Julia..."
	@bash scripts/install-julia.sh

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
	ray start --head --port=$(SERVER_RAY_PORT) --num-cpus=$(NUM_CPUS) --num-gpus=$(NUM_GPUS)

run-server-worker: ## Start Ray worker
	@echo "Starting Ray worker..."
	ray start --address=localhost:$(SERVER_RAY_PORT) --num-cpus=$(NUM_CPUS) --num-gpus=$(NUM_GPUS)

run-servers: ## Start both Julia and Python API servers
	@echo "Starting Julia, Python API, and Ray servers..."
	@bash ./scripts/run-servers.sh $(SERVER_JL_PORT) $(SERVER_PY_PORT)

fetch-all:  ## Fetch all dependencies
	@$(MAKE) fetch-wheel REPO=$(DATA_EXPORTER_REPO) BRANCH=$(DATA_EXPORTER_BRANCH) VERSION=$(DATA_EXPORTER_VERSION)
	@$(MAKE) fetch-wheel REPO=$(TWINDIGRID_REPO) BRANCH=$(TWINDIGRID_BRANCH) VERSION=$(TWINDIGRID_VERSION)
	@$(MAKE) fetch-wheel REPO=$(UTILITY_FUNCTIONS_REPO) BRANCH=$(UTILITY_FUNCTIONS_BRANCH) VERSION=$(UTILITY_FUNCTIONS_VERSION)