# Expansion theory: SDDP + ADMM coupling

Expansion planning in Dig-A-Plan is built on two layers:

- **SDDP** (Julia) for **multi-stage expansion planning**, and
- **ADMM** (Python) for **operational feasibility checks** (power-flow constraints under scenarios),

using **cut feedback** (Benders-style information) to refine planning decisions.

---

## 1) What expansion planning solves

Expansion planning decides **what to upgrade (lines/transformers)** and **when** so the grid remains feasible as load and PV evolve.  
Because the future is uncertain, the plan must work across **multiple scenarios** while respecting operational limits (voltage/current/device constraints).

To achieve this, Dig-A-Plan uses a **planning–operations loop** coordinated by the **API**, which exchanges information between the planning and operational models:

1. **Planning (SDDP / Julia)** proposes candidate investment decisions for each stage (e.g., which line/transformer capacities to reinforce and when).
2. **Operations (ADMM / Python)** checks whether the grid can operate under many scenarios given those investments (power-flow feasibility, voltage/current limits, and device actions such as taps/switching if enabled).
3. **Cuts / feedback** are extracted from the operational solves and sent back (via the API) to SDDP, so the next planning iteration improves (the planner learns which investments avoid infeasibility or high operational penalties).

### Planning ↔ Operations feedback loop

![Planning–operations feedback loop (SDDP ↔ ADMM via cuts)](docs/images/expansion-loop.png)

*The API sits between SDDP and ADMM: SDDP sends candidate expansion decisions to ADMM for feasibility checks, and ADMM returns cutting-plane feedback (cuts) that refines the next SDDP iteration.*

---

## 2) What SDDP is doing (planning layer)

SDDP solves the **long-term expansion planning** problem as a **multi-stage stochastic optimization**:

- **Stages**: planning periods (e.g., 5 stages representing 5-year blocks)
- **Uncertainty**: how load and PV evolve over time (represented by sampled scenarios)
- **Decisions**: how much to reinforce/expand **line and transformer capacities**, balancing investment cost against future operational risk

SDDP runs in two complementary phases:

- **Forward simulation**: it samples scenarios and simulates the current planning policy stage-by-stage to generate candidate states/decisions.
- **Backward recursion**: it solves stage subproblems and updates a piecewise-linear approximation of the **future cost-to-go** using **cuts**.

These cuts capture how future cost/feasibility changes with the planning state. In Dig-A-Plan, the approximation is strengthened over iterations using feedback from the operational layer (ADMM), so SDDP learns expansion decisions that remain feasible across scenarios.

---

## 3) What ADMM does (operations layer)

The expansion decisions proposed by SDDP are only useful if the grid can **actually operate** once those reinforcements are built.  
For each candidate plan (and for many scenarios), Dig-A-Plan solves an operational feasibility problem that enforces:

- power-flow constraints
- voltage bounds
- current/thermal limits
- device constraints (e.g., OLTC tap ranges)
- switching / reconfiguration decisions

Running these operational checks for many scenarios can be computationally expensive, especially on large networks.

To scale this step, Dig-A-Plan uses **ADMM** as a **scenario-decomposition** method:

- Each scenario (or scenario group) is solved as a subproblem.
- A consensus step coordinates shared variables/constraints across subproblems.

---

## 4) How the coupling works: “cuts” from operations to planning

After SDDP proposes a candidate plan (capacities / investments), ADMM solves the corresponding operational problem.

From the operational solution, we extract **sensitivity information** (typically dual variables / multipliers) and build a cut that tells the planner:

These cuts are accumulated and passed back to SDDP, improving the cost-to-go approximation and steering investments toward operational feasibility.


---

## 5) Minimal “how to run” (for context)

### Run expansion (IEEE-33 example)
```sh
python experiments/expansion_planning_script.py --kace ieee_33 --cachename run_ieee33
```

Useful options:

--withapi false
Run locally without calling the FastAPI server (useful for debugging).

--admmiter <n>
Override admm_config.max_iters to speed up tests (fewer ADMM iterations).

--riskmeasuretype <Expectation|Entropic|Wasserstein|CVaR|WorstCase>
Select the SDDP risk measure.

--riskmeasureparam <value>
Set the risk-measure parameter used by SDDP.

--fixedswitches true|false
If true, ADMM treats switches as fixed (no switching optimization).

Outputs are saved under .cache/.

---

## 6) Key concepts

- **Stage**: a planning time period (e.g., 5-year block). 
- **Scenario**: one realization of uncertainty (load/PV evolution, profiles). 
- **Planning decision**: reinforcement/expansion capacity decisions. 
- **Operational decision**: feasibility actions (flows, taps, switching). 
- **Cut**: linear approximation used by SDDP to represent future cost/feasibility impact.

---

