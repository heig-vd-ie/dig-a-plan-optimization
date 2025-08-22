# Common Makefile

PYTHON_VERSION := 3.12
VENV_DIR := .venv
ORG := heig-vd-ie

# Default target: help
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available targets:"
	@grep -hE '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-42s\033[0m %s\n", $$1, $$2}'

setup-project: ## Setup the project
	@echo "Setting up the project..."
	@bash scripts/setup.sh
	@$(MAKE) install-all

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

poetry-install: ## Update Python packages using Poetry
	@echo "Updating Python packages using Poetry..."
	@poetry install --extras "internal" || ( \
		echo "⚠️ If psycopg-c installation fails, see:"; \
		echo "https://stackoverflow.com/questions/77727508/problem-installing-psycopg2-for-python-venv-through-poetry"; \
		echo "Error hint: _psycopg-c may not support PEP 517 builds or may be missing system dependencies."; \
		exit 1 \
	)

venv-activate-and-poetry-use-install: SHELL:=/bin/bash
venv-activate-and-poetry-use-install: ## Activate venv and install packages
	@echo "Activating virtual environment and installing packages..."
	@test -d .venv || make _venv
	@bash --rcfile <(echo '. ~/.bashrc; . .venv/bin/activate; echo "You are now in a subshell with venv activated."; make poetry-use; make poetry-install; . scripts/enable-direnv.sh') -i

install-vscode-extensions: ## Install Visual Studio Code extensions
	@echo "Installing Visual Studio Code extensions..."
	@xargs -n 1 code --install-extension < .vscode/extensions.txt 

freeze-vscode-extensions: ## Update Visual Studio Code extensions
	@echo "Freezing Visual Studio Code extensions..."
	@code --list-extensions > .vscode/extensions.txt

install-all:  ## Install all dependencies and set up the environment
	@$(MAKE) install-pipx
	@$(MAKE) install-python-wsl
	@$(MAKE) install-poetry
	@$(MAKE) install-deps
	@$(MAKE) install-vscode-extensions
	@$(MAKE) _venv
	@$(MAKE) venv-activate-and-poetry-use-install
	@echo "All dependencies installed successfully!"

uninstall-venv: ## Uninstall the virtual environment
	@echo "Uninstalling virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Virtual environment uninstalled."

run-tests-py: ## [file] Run tests using pytest (check venv is activated otherwise activated)
	@echo "Running Python tests..."
	@if [ -n "$(file)" ]; then \
		PYTHONWARNINGS=ignore $(VENV_DIR)/bin/python -m pytest "$(file)" -v; \
	else \
		PYTHONWARNINGS=ignore $(VENV_DIR)/bin/python -m pytest tests/ -v;\
	fi

format-julia:  ## Format Julia code in the src directory
	@echo "Formatting Julia code with JuliaFormatter..."
	julia -e 'using JuliaFormatter; format("src/")'

format-py: ## Format Python code using black
	@echo "Formatting Python code with black..."
	@poetry run black .

build-wheel: # Build the Python wheel for this project based on pyproject.toml version
	@echo "Building Python wheel..."
	@poetry build -f wheel

fetch-wheel: ## Fetch the Python wheel from a remote URL [<organization> <repo> <name-of-wheel> <version> <dest_dir>]
	@echo "Fetching Python wheel..."
	@bash scripts/fetch-wheel.sh $(ORG) $(REPO) $(BRANCH) $(VERSION) $(DEST_DIR)
