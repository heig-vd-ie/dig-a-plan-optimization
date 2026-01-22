# Reconfiguration Theory & Experiments (IEEE-33)

This document explains what the **reconfiguration** experiments do, how the inputs are structured, and how to run the three available variants:

- **ADMM** (`01-reconfiguration-admm.py`)
- **Benders** (`02-reconfiguration-bender.py`)
- **Combined** (`03-reconfiguration-combined.py`)

The goal is to solve an **operational feasibility + reconfiguration** problem under many load/PV scenarios, optionally including **switching decisions** and **OLTC tap control**, while respecting network limits (voltage, currents, power flow).

---

## 1) What “reconfiguration” means 

- selecting a valid **radial**  switching state,
- potentially adjusting **transformer taps (OLTC)**,
- ensuring **power flow feasibility** across scenarios,
- minimizing operational objectives such as:
  - infeasibility ,
  - losses / proxy costs,
  - switching penalties or control effort.

This project exposes reconfiguration solvers through FastAPI endpoints, and the experiment scripts send payloads to those endpoints.

---

## 2) Input payload (00-reconfiguration.json)

All three reconfiguration scripts start from the same payload file:

`experiments/ieee_33/00-reconfiguration.json`

It has three main blocks:

### 2.1 `grid`

```json
"grid": {
  "name": "test",
  "pp_file": "examples/ieee_33/simple_grid.p",
  "s_base": 1000000,
  "cosφ": 0.95,
  "egid_id_mapping_file": "examples/ieee_33/consumer_egid_idx_mapping.csv",
  "minimum_impedance": 0.001
}
```

- `pp_file`: path to the **pandapower** network snapshot.
- `s_base`: base power used for per-unit scaling (e.g., 1 MVA).
- `cosφ`: default power factor assumption.
- `egid_id_mapping_file`: mapping between consumer identifiers and internal indices.
- `minimum_impedance`: a floor to prevent numerical issues (e.g., zero-impedance branches).

### 2.2 `profiles`

```json
"profiles": {
  "load_profiles": ["examples/ieee_33/load_profiles"],
  "pv_profile": "examples/ieee_33/pv_profiles",
  "target_year": 2030,
  "quarter": 1,
  "scenario_name": "Basic"
}
```

- Load and PV profiles define **scenario time series**.
- `target_year`, `quarter`, `scenario_name` choose a slice of the scenario dataset.

### 2.3 `konfig`

```json
"konfig": {
  "verbose": false,
  "solver_name": "gurobi",
  "solver_non_convex": 2,
  "big_m": 1000,
  "ε": 1,
  "ρ": 2,
  "γ_admm_penalty": 1,
  "γ_infeasibility": 10,
  "γ_trafo_loss": 100,
  "time_limit": 10,
  "max_iters": 20,
  "μ": 10,
  "τ_incr": 2,
  "τ_decr": 2,
  "seed": 42,
  "groups": 5,
  "vm_max_pu": 1.05,
  "vm_min_pu": 0.95
}
```

This block contains solver and algorithm parameters. Not all fields are used by all methods.

Typical roles:
- `solver_name`: optimizer backend (e.g., `gurobi`).
- `solver_non_convex`: controls nonconvex handling in the solver.
- `big_m`: Big-M constant used in mixed-integer constraints (e.g., switching logic).
- `ε`, `ρ`: commonly used as ADMM tuning parameters / tolerances.
- `γ_infeasibility`: weight for infeasibility slack penalties (encourages feasibility).
- `time_limit`: runtime limit per solve / iteration (seconds).
- `max_iters`: maximum iterations (notably ADMM).
- `vm_min_pu`, `vm_max_pu`: voltage bounds.

Note: In the Benders and Combined scripts, `konfig` is **filtered** to the fields supported by their respective Pydantic configs, then some values are overridden for that method.

---

## 3) Three solver variants

### ADMM (`/reconfiguration/admm`)
Script: `experiments/ieee_33/01-reconfiguration-admm.py`  
Best for many scenarios / large networks (decomposition-based).

```text
PATCH http://{LOCALHOST}:{PY_PORT}/reconfiguration/admm
```

### Benders (`/reconfiguration/bender`)
Script: `experiments/ieee_33/02-reconfiguration-bender.py`  
Uses a master–subproblem approach; the script filters/overrides `konfig` for `BenderConfig`.

```text
PATCH http://{LOCALHOST}:{PY_PORT}/reconfiguration/bender
```

### Combined (`/reconfiguration/combined`)
Script: `experiments/ieee_33/03-reconfiguration-combined.py`  
Single integrated (baseline) approach; the script filters/overrides `konfig` for `CombinedConfig`.

```text
PATCH http://{LOCALHOST}:{PY_PORT}/reconfiguration/combined
```

---
