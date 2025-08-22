# Makefile for Dig-A-Plan setup (supports Linux and WSL)
include Makefile.common.mak

SERVER_PORT ?= 8080 # Julia server targets
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
	@echo "Starting Julia API server on localhost:$(SERVER_PORT)..."
	julia --project=src/model_expansion/. src/model_expansion/src/Server.jl $(SERVER_PORT)

fetch-all:  ## Fetch all dependencies
	@$(MAKE) fetch-wheel REPO=$(DATA_EXPORTER_REPO) BRANCH=$(DATA_EXPORTER_BRANCH) VERSION=$(DATA_EXPORTER_VERSION)
	@$(MAKE) fetch-wheel REPO=$(TWINDIGRID_REPO) BRANCH=$(TWINDIGRID_BRANCH) VERSION=$(TWINDIGRID_VERSION)
	@$(MAKE) fetch-wheel REPO=$(UTILITY_FUNCTIONS_REPO) BRANCH=$(UTILITY_FUNCTIONS_BRANCH) VERSION=$(UTILITY_FUNCTIONS_VERSION)