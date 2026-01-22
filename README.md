# Dig-A-Plan Optimization

Dig-A-Plan is a scalable optimization framework for distribution grid planning and operational reconfiguration under uncertainty.  
It combines:
- **multistage expansion model** (SDDP) for long-term reinforcement planning, and  
- **scenario-based ADMM solver** for operational feasibility checks, network reconfiguration (switching), and OLTC tap control across many load and PV scenarios.

The framework is designed to handle large real-world distribution networks (from 33-bus test systems up to 10'000+ nodes) and ensures that long-term planning decisions remain operationally feasible under realistic uncertainty.

## Requirements

- **Gurobi license**: Request a (WSL) license at https://license.gurobi.com/ and save it to:
  - `~/gurobi_license/gurobi.lic`



## Installation

1. **Clone the repository**
```sh
   git clone <REPO_URL>
   cd <REPO_NAME>
```
2. **Install make (Ubuntu/WSL only, if not already installed)**
```sh
    sudo apt update
    sudo apt install make
```
3. **Install project dependencies**
```sh
    make install-all
```
If this fails and you can’t resolve it quickly, please open an issue (or contact the maintainers)

## Run application

1. **Activating the Virtual Environment and Setting Environment Variables (each time you work on the project)**

```sh
    make venv-activate
```
2. **Build and start all services**

This starts the full stack (Julia, Python/FastAPI, Ray Head, Grafana, and worker processes). 
```sh
    make build  # if it fails because of poetry, use `poetry lock`
    make start
```
3. **Use the tmux session**

make start opens a tmux session with three panes:

### Use the tmux session

```text
+------------------------------+
| Services logs                |
| (Julia / Python / Grafana)   |
+------------------------------+
| Ray logs / worker processes  |
+------------------------------+
| Interactive shell            |
| (run commands & experiments) |
+------------------------------+


Navigation: Ctrl + B, then Down Arrow → move to the next pane

4. **Run an experiment (example: IEEE-33 reconfiguration)**
From the interactive pane: 
```sh
    python experiments/ieee_33/01-reconfiguration-admm.py
```
For details about reconfiguration runs (per Benders / combined / ADMM), see [here](docs/experiments/01-theory-reconfiguration.md) 






Code formatting is handled automatically with `black`. Please install the Black extension in VS Code and enable it for consistent formatting.

### Run the expansion problem

You can run the respected endpoint.

The results will be saved in the folder of `.cache`.

### Run the case of Boisy & Estavayer

In the `dig-a-plan-monorepo` (one folder above this project), if you run the following target, you will get the data needed for boisy and Estavayer in `.cache` folder of this project.
```sh
cd .. && make run-all
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


### Add local network
On linux:
```sh
# Add Tailscale repo
curl -fsSL https://tailscale.com/install.sh | sh
# Start Tailscale
sudo tailscale up
```

```sh
tailscale ip
```

3. **Configure Ray Head + SSH access (if using a Worker machine)**

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