# Dig-A-Plan Optimization

Common commands are available in the `Makefile`. To view available options, simply run `make` in your shell.

## Initial Setup

To install all dependencies on a new machine, run:
```sh
make install-all
```

You will need a Gurobi license to run this project. Visit [https://license.gurobi.com/](https://license.gurobi.com/), request a new WSL license for your machine, and save it to `~/gurobi_license/gurobi.lic`.

Code formatting is handled automatically with `black`. Please install the Black extension in VS Code and enable it for consistent formatting.

To install Julia on a new machine, run:
```sh
make install-julia
```

After that, you can open a server of julia for expansion model with the following command:
```sh
make run-server-jl
```

For more information regarding Julia, go to [src/model_expansion/README.md](src/model_expansion/README.md).


## Updating the Virtual Environment or Packages

If you need to update packages listed in `pyproject.toml`, use:
```sh
make poetry-update
```
or
```sh
poetry update
```

## Activating the Virtual Environment and Setting Environment Variables

Each time you start working on the project, activate the virtual environment by running:
```sh
make enable-venv
```

## Install Julia
Check following doc [here](docs/Julia/01-install-julia.md).

## Project structure:

```
.
├── .github/         # GitHub workflows and configuration files
├── .vscode/         # VS Code workspace settings and recommended extensions
├── data/            # Input datasets or sample data for optimization tasks
├── docs/            # Project documentation and additional resources
├── dry-run/         # Scripts or outputs for test runs and experimentation
├── examples/        # Example scripts or notebooks demonstrating usage
├── src/             # Core optimization code and modules
├── tests/           # Unit tests for the optimization logic
├── .envrc           # Environment variable configuration for direnv
├── .gitignore       # Specifies files and directories to be ignored by Git
├── Makefile         # Common commands for setup, development, and testing
├── pyproject.toml   # Python dependencies and project configuration
└── README.md        # Project overview and instructions (this file)
```

### Development
In a separate shell, run `make run-server-jl`.

- If you add a new feature in python part, make sure to test `run-tests-py` to verify that existing features continue to work correctly.

- For Julia part, test `make run-tests-jl`.

Or do the following for testsing both:

```sh
make venv-activate
make run-tests
```
