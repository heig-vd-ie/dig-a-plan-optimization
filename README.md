# Dig-A-Plan Optimization

Common commands are available in the `Makefile`. To view available options, simply run `make` in your shell.

## Initial Setup

To install all dependencies on a new machine, run:
```sh
make install-all
```

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
