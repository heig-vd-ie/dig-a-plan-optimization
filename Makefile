# Makefile for Dig-A-Plan setup (supports Linux and WSL)
include Makefile.common.mak

SERVER_PORT ?= 8080 # Julia server targets

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
