# Dig-A-Plan Optimization

Common commands are available in the `Makefile`. To view available options, simply run `make` in your shell.

## Initial Setup

To install all dependencies on a new machine, run:
```sh
make install-all
```

You will need a Gurobi license to run this project. Visit [https://license.gurobi.com/](https://license.gurobi.com/), request a new WSL license for your machine, and save it to `~/gurobi_license/gurobi.lic`.

Code formatting is handled automatically with `black`. Please install the Black extension in VS Code and enable it for consistent formatting.


### Activating the Virtual Environment and Setting Environment Variables

Each time you start working on the project, activate the virtual environment by running:
```sh
make venv-activate
```

## Run application

You can run the following to run all existing servers (Julia, Python, RAY, GRAFANA, and Worker). 
```sh
make start
```

### Access to worker Machine

1. You need to update the following env variable (in `.envrc`) based on your IP so it detects the HEAD machine running Ray:
```sh
export HEAD_HOST=10.192.189.51
```
> Note: Run `make show-current-specs` to get the ip of current machine.

2. For access to the Worker machine, you need to add your ssh key in the worker machine:
```sh
ls ~/.ssh/
ssh-keygen -y -f ~/.ssh/id_rsa > ~/.ssh/id_rsa.pub
ssh-copy-id -i ~/.ssh/id_rsa.pub user@address
```
Contact Mohammad for getting `user@access`. 

### Run the expansion problem

You can run the following target to run the expansion problem:

```sh
make run-expansion PAYLOAD=data/payloads/simple_grid.json
```

The results will be saved in the folder of `.cache`.

### Run the case of Boisy & Estavayer

In the `dig-a-plan-monorepo` (one folder above this project), if you run the following target, you will get the data needed for boisy and Estavayer in `.cache` folder of this project.
```sh
cd .. && make run-extraction
```

### Dashboards

1. FastAPI endpoints: [http://localhost:8000/docs#](http://localhost:8000/docs#)
2. Ray Dashboard: [http://localhost:8265](http://localhost:8265)
3. Grafana Dashboard [http://localhost:4000](http://localhost:4000)
4. Prometheous Dashboard: [http://localhost:9090](http://localhost:9090)
5. MongoDB GUI: `mongodb-compass` in a separate terminal


## Development

### Unit tests

In a shell, run `make start`.

- If you add a new feature in python part, make sure to test `run-tests-py` to verify that existing features continue to work correctly.

- For Julia part, test `make run-tests-jl`.

Or do the following for testsing both:

```sh
make venv-activate
make run-tests
```

### Updating the Virtual Environment or Packages

If you need to update packages listed in `pyproject.toml`, use:
```sh
make poetry-update
```
or
```sh
poetry update
```

### Julia
For more information regarding Julia, go to [src/model_expansion/README.md](src/model_expansion/README.md) and check following doc [here](docs/Julia/01-install-julia.md).
