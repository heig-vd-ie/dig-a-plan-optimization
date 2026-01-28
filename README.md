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

**make start** opens a tmux session with three panes:


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

```

4. **Run an experiment (example: IEEE-33 reconfiguration)**
From the interactive pane: 
```sh
    python experiments/ieee_33/01-reconfiguration-admm.py
```
For details about reconfiguration runs (per Benders / combined / ADMM), see [here](docs/experiments/01-theory-reconfiguration.md) 

5. **Run the expansion problem**

The **expansion** solves the *long-term planning* problem: it decides which grid assets (lines/transformers) should be reinforced/expanded over multiple stages under uncertainty (load/PV scenarios).  It uses **SDDP** (served by the Julia service) and runs repeated operational checks to keep plans feasible.
For more information regarding Julia, go to [src/model_expansion/README.md](src/model_expansion/README.md) and check following doc [here](docs/Julia/01-install-julia.md).

```sh
    python experiments/ieee_33/04-expansion.py
```
The results will be saved in the folder of `.cache`.

### Run the case of Boisy & Estavayer

In the `dig-a-plan-data-processing` , if you run the following target, you will get the data needed for boisy and Estavayer in `.cache` folder of this project.
```sh
cd .. && make run-all
```

## Ray worker

This project can run heavy tasks on a **remote Ray worker**. The default setup assumes machines are connected via **Tailscale** (virtual network).

1. Start the stack on the **head** machine:
```sh
   make start
```
2. Connect to the worker machine with ssh:
```sh
    ssh `user@worker-host` or `user@<worker-ip>`
```
Or use the helper comment:
```sh
    make connect-ray-worker
```
3. On the worker machine, start the Ray worker and connect it to the head:
```sh
    make run-ray-worker
```

## Development

1. Code formatting is handled automatically with `black`. Please install the **Black** extension in VS Code and enable **format on save** for consistent formatting.












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
