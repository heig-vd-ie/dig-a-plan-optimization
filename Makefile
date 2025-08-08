# Makefile for Dig-A-Plan setup (supports Linux and WSL)

PYTHON_VERSION := 3.12
VENV_DIR := .venv

# Default target: help
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

detect-env: ## Detect whether running in WSL or native Linux
	@if grep -qEi "(Microsoft|WSL)" /proc/version; then \
		echo "Detected: WSL environment"; \
	else \
		echo "Detected: Native Linux environment"; \
	fi

install-pipx: ## Install pipx (Python packaging tool)
	@echo "Installing pipx..."
	sudo apt update
	sudo apt install -y pipx
	pipx ensurepath --force

install-python-wsl: ## Install Python $(PYTHON_VERSION) and venv support on WSL
	@echo "Checking if Python $(PYTHON_VERSION) is installed..."
	@if ! command -v python$(PYTHON_VERSION) >/dev/null 2>&1; then \
		echo "Installing Python $(PYTHON_VERSION)..."; \
		echo "# Reference: Tutorial is the following link, https://www.linuxtuto.com/how-to-install-python-3-12-on-ubuntu-22-04/"; \
		sudo add-apt-repository -y ppa:deadsnakes/ppa; \
		sudo apt update; \
		sudo apt install -y python$(PYTHON_VERSION) python$(PYTHON_VERSION)-venv; \
	else \
		echo "Python $(PYTHON_VERSION) already installed"; \
	fi

install-poetry: ## Install Poetry using pipx
	@echo "Installing Poetry..."
	pipx install poetry

install-deps: ## Install system dependencies
	@echo "Installing system dependencies..."
	sudo apt update
	sudo apt install -y libpq-dev gcc python3-dev build-essential direnv

_venv: ## Create a virtual environment if it doesn't exist
	@echo "Creating virtual environment with Python $(PYTHON_VERSION)..."
	python$(PYTHON_VERSION) -m venv .venv

venv-activate: SHELL:=/bin/bash
venv-activate: ## enter venv in a subshell
	@test -d .venv || make _venv
	@bash --rcfile <(echo '. ~/.bashrc; . .venv/bin/activate; echo "You are now in a subshell with venv activated."; . scripts/enable-direnv.sh') -i

poetry-use: ## Install Python packages using Poetry
	@echo "Installing Python packages using Poetry..."
	poetry env use .venv/bin/python$(PYTHON_VERSION)

poetry-update: ## Update Python packages using Poetry
	@echo "Updating Python packages using Poetry..."
	@poetry update || ( \
		echo "⚠️ If psycopg-c installation fails, see:"; \
		echo "https://stackoverflow.com/questions/77727508/problem-installing-psycopg2-for-python-venv-through-poetry"; \
		echo "Error hint: _psycopg-c may not support PEP 517 builds or may be missing system dependencies."; \
		exit 1 \
	)

venv-activate-and-poetry-use-update: SHELL:=/bin/bash
venv-activate-and-poetry-use-update: ## Activate venv and install packages
	@echo "Activating virtual environment and installing packages..."
	@test -d .venv || make _venv
	@bash --rcfile <(echo '. ~/.bashrc; . .venv/bin/activate; echo "You are now in a subshell with venv activated."; make poetry-use; make poetry-update; . scripts/enable-direnv.sh') -i

install-all:  ## Install all dependencies and set up the environment
	@$(MAKE) install-pipx
	@$(MAKE) install-python-wsl
	@$(MAKE) install-poetry
	@$(MAKE) install-deps
	@$(MAKE) _venv
	@$(MAKE) venv-activate-and-poetry-use-update
	@echo "All dependencies installed successfully!"

install-julia:  ## Install Julia
	@echo "Installing Julia..."
	@bash scripts/install-julia.sh

uninstall-venv: ## Uninstall the virtual environment
	@echo "Uninstalling virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Virtual environment uninstalled."

run-pytest: ## Run tests using pytest (check venv is activated otherwise activated)
	@echo "Running Python tests..."
	@if [ -n "$(t)" ]; then \
		poetry run pytest -v "$(t)"; \
	else \
		poetry run pytest; \
	fi

run-jltest:  ## Run tests of Julia
	@echo "Running Julia tests..."
	julia --project=src/expansion_model/. src/expansion_model/test/runtests.jl

run-tests: ## Run all tests
	@$(MAKE) run-pytest
	@$(MAKE) run-jltest

format-julia:  ## Format Julia code in the src directory
	@echo "Formatting Julia code with JuliaFormatter..."
	julia -e 'using JuliaFormatter; format("src/")'

format-python: ## Format Python code using black
	@echo "Formatting Python code with black..."
	@poetry run black .

format: format-julia format-python ## Format all code (Julia and Python)

server-julia:  ## Start Julia server
	@echo "Starting Julia server..."
	julia --project=src/expansion_model/. src/expansion_model/api/server.jl
