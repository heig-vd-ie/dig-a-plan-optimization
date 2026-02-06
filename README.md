# Dig-A-Plan Planning Tool

<p align="center">
  <img src="docs/images/grid.jpg" width="300" alt="Description">
  <br>
  <em>© HEIG-VD, 2025</em>
</p>

Dig-A-Plan is a scalable optimization tool designed for distribution grid planning and operational reconfiguration. It bridges the gap between long-term investment and short-term operational flexibility under uncertainty by combining two core modules:
- **Multistage Expansion Planning (Long-term).** This module optimizes the long-term expansion of physical infrastructure, such as cables and transformers. It utilizes Stochastic Dual Dynamic Programming (SDDP) to handle complex, multistage decision-making processes over long horizons while accounting for uncertainty.
- **Switching Reconfiguration Planning (Operational).** This module performs operational feasibility checks and manages network topology through switching and OLTC (On-Load Tap Changer) control. It evaluates numerous load and PV scenarios using a Scenario-based ADMM (Alternating Direction Method of Multipliers) approach to ensure the grid remains stable and efficient across varying conditions.

The proposed tool is designed to handle large real-world distribution networks (from 33-bus test systems up to 1'000+ nodes) and ensures that long-term planning decisions remain operationally feasible under realistic uncertainty.

For the theory behind the **SDDP ↔ ADMM** coupling (cuts, planning/operation loop), see: [here](docs/experiments/02-theory-expansion-sddp-admm.md). There is also a publication explaining the method behind using ADMM for switching reconfiguration planning, see [here](https://papers.ssrn.com/sol3/Delivery.cfm?abstractid=5968040). 

## Requirements

- Linux or WSL on windows.
- Common tools such as `make`, `docker`, `tmux`, `julia`, and `python 3.12`.
- **Gurobi license**: Request a (WSL) license at https://license.gurobi.com/ and save it to:
  - `~/gurobi_license/gurobi.lic`

## Installation

1. **Clone the repository**
```sh
   git clone https://github.com/heig-vd-ie/dig-a-plan-optimization
   cd dig-a-plan-optimization
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

The **expansion** computes a multi-stage reinforcement plan (lines/transformers) under uncertainty (load/PV scenarios).  
It is solved with **SDDP** (Julia service). After each SDDP iteration, the pipeline runs **ADMM** (Python) as an operational feasibility check and uses the results to generate cuts that guide the next planning iteration.

Run an example (IEEE-33):
```sh
python experiments/expansion_planning_script.py --kace ieee_33 --cachename run_ieee33
```
To see all available options:
```sh
python experiments/expansion_planning_script.py --help
```

- Results are saved under .cache/output_expansion`

- For more information regarding Julia, go to [src/model_expansion/README.md](src/model_expansion/README.md) and check following doc [here](docs/Julia/01-install-julia.md).


### Run the case of Boisy & Estavayer [It needs access to confidential data]

In the `dig-a-plan-data-processing`, if you run the following target, you will get the data needed for boisy and Estavayer in `.cache` folder of this project.
```sh
cd .. && make run-all
```

## Ray worker

This project can run heavy tasks on a **remote Ray worker**. The default setup assumes machines are connected via **Tailscale** (virtual network).

1. Start the stack on the **head** machine:
```sh
   make start
```
This typically:

starts Docker services (Python/FastAPI, Julia SDDP service, Grafana/Prometheus/Mongo),

opens a tmux session with panes for logs and an interactive shell.

2. Connect to the worker machine with ssh:
```sh
    ssh `user@worker-host` or `user@<worker-Tailscale IP>`
```
Or use the helper comment:
```sh
    make connect-ray-worker
```
3. On the worker machine, start the Ray worker and connect it to the head:
```sh
    make run-ray-worker
```

For full setup details (Tailscale, SSH keys, troubleshooting), see [here](docs/ops/ray-worker.md) 

## Development

1. Code formatting is handled automatically with `black`. Please install the **Black** extension in VS Code and enable **format on save** for consistent formatting.

2. Updating the Virtual Environment or Packages: If you need to update packages listed in `pyproject.toml`, use:
```sh
make poetry-update
```
or
```sh
poetry update
```

### Unit tests

In a shell, run `make start`.

- If you add a new feature in python part, make sure to test `run-tests-py` to verify that existing features continue to work correctly.

- For Julia part, test `make run-tests-jl`.

Or do the following for testsing both:

```sh
make venv-activate
make run-tests
```
