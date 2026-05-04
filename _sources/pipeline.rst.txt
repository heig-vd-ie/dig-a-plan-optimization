Pipeline 
=========================================================================================


This module implements a **two-stage optimization framework** using Benders decomposition for **distribution grid planning**. The two stages are:

1. **Master problem**: Determines the grid topology by selecting active lines (e.g., switch statuses) via binary decisions.
2. **Slave problem**: Solves the DistFlow optimization problem for the fixed topology to evaluate feasibility and cost, including slack penalties.

The framework uses **Pyomo** for mathematical modeling and **Polars + Patito** for structured data handling and validation.

Model Structure
---------------

Model Generation Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: generate_master_model()

   Returns a Pyomo AbstractModel.

   Builds the **master model**, which defines the topology (candidate branches) using binary variables :math:`d_{l~i~j}`.


.. function:: generate_slave_model()

   Returns a Pyomo AbstractModel.

   Builds the **slave model**, which solves the power flow problem for a fixed topology, with dual suffix enabled to extract marginal costs of constraints for Benders cuts.


Main Class: DigAPlan
--------------------

.. class:: DigAPlan

Central class that manages the **Benders decomposition pipeline**, model generation, and execution loop.

Constructor
~~~~~~~~~~~

.. code-block:: python

    def __init__(..., big_m=1e4, penalty_cost=1e3, current_factor=1e2, voltage_factor=1e1, power_factor=1e1, ...)

**Sets up:**

- :code:`big_m`: Big-M constant for deactivating constraints.
- :code:`penalty_cost`: Penalty coefficient for slack violations.
- :code:`*_factor`: Scaling coefficients to normalize units (current, voltage, power).
- :code:`slack_threshold`: Threshold below which slack violations are ignored.

Attributes and Properties
~~~~~~~~~~~~~~~~~~~~~~~~~

- **node_data**, **edge_data**: Typed and validated using Patito.
- **master_model**, **slave_model**: Abstract models generated via helper functions.
- **master_model_instance**, **slave_model_instance**: Concrete Pyomo models instantiated with real data.
- **slack_node**: Unique slack node of the system



Data Injection and Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. method:: add_grid_data(**grid_data)

Injects `node_data` and `edge_data` into the model:

- Validates structure and schema via Patito.
- Ensures exactly one slack node exists.
- Computes :math:`V^2_{\text{slack}}` and stores as parameter.
- Triggers instantiation of Pyomo models using these validated inputs.


Model Instantiation
-------------------

.. method:: __instantiate_model()

Constructs the `grid_data` dictionary and instantiates Pyomo concrete models. Mathematical parameters included:

- Node set: :math:`\mathcal{N}`

- Line set: :math:`\mathcal{L}`

- Switch set: :math:`\mathcal{S}`

- Power injections: :math:`p_i, q_i \ \forall i \in \mathcal{N}`

- Impedance: :math:`r_\ell, x_\ell, b_\ell \ \forall \ell \in \mathcal{L}`

- Slack voltage: :math:`V_{slack}^2`

- Voltage limits: :math:`V^{min}, V^{max}`

- Current limits: :math:`I^{max}`

- Decision variables: :math:`d_{l i j}` (discrete switch decision)


Master-Slave Iteration
----------------------

.. method:: solve_models_pipeline(max_iters)

Performs the main optimization loop using Benders decomposition:

1. **Initialize topology** using :code:`find_initial_state_of_switches()`:
   - Builds a graph of normally closed switches.
   - If it forms a **spanning tree**, a BFS tree is used to initialize :math:`d_{l~i~j}`.
   - Otherwise, a first call to the master problem generates a feasible topology.

2. **Solve the slave model** using the fixed topology from the current master solution:
3. **Check feasibility** using slack variable violations:

   - Overcurrent: :math:`i_{l~i~j}^{\text{slack}}`
   - Overvoltage: :math:`V_n^{+}`
   - Undervoltage: :math:`V_n^{-}`
   - Sets self.infeasible_slave = True if any slack violations are present.

4. **Generate a Benders cut** using dual values from slave constraints.

    - If the slave is infeasible → infeasibility cut

    - If the slave is feasible → optimality cut

5. **Solve the master model** with the updated cuts.
6. **Repeat** until convergence or max iteration count.

Progress and convergence are tracked using `tqdm`, based on the gap between master and slave objectives.





Benders Cut Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. method:: add_benders_cut()

Generates a Benders cut based on dual information from the slave problem. The procedure dynamically constructs either an **optimality cut** or an **infeasibility cut**, depending on the outcome of the slave solution.

Steps performed:

1. **Determine constraint weights**:

   - A `constraint_dict` is defined to associate each relevant constraint in the slave model with a scalar multiplier:
     - If the slave is **infeasible**, the constraints are scaled by `self.infeasibility_factor`.
     - If the slave is **feasible**, they are scaled by `self.optimality_factor`.
   - These multipliers define how each dual is weighted in the Benders cut.



2. **Extract duals from slave model**:

   - Dual values (`self.slave_model_instance.dual`) are extracted and converted into a Polars DataFrame, mapping constraint names to marginal costs (shadow prices).
   - Constraint names are parsed and cleaned (e.g., removing array-like syntax) to isolate:
     - The base constraint name (e.g., `voltage_limit`)
     - Its associated indices `(l, i, j)`.


3. **Rescale and aggregate duals**:

   - Each dual is scaled by the corresponding weight from `constraint_dict`.
   - The indexed duals are grouped by the connection key `LC = (l, i, j)` and summed to compute the total marginal cost per connection.


4. **Link slave and master variables**:

   - Extracts the value of :math:`d_{l~i~j}` from the slave solution (:math:`d`).
   - Retrieves the symbolic decision variables :math:`d_{variable}` from the master model.


5. **Construct the Benders cut**:

   - Initializes the cut expression with the current slave objective value :math:`\theta`.

   - For each candidate line, adds a linear term:

     
     .. math::

        \mu_{l~i~j} \cdot (d_{l~i~j} - \hat{d}_{l~i~j})

     where:

     - :math:`\mu_{l~i~j}` is the aggregated dual multiplier from the slave.
     - :math:`\hat{d}_{l~i~j}` is the evaluated value of the binary decision variable.


6. **Add cut to master model**:

   - If the slave is **infeasible**, the cut forces the master to exclude the current topology:

     .. math::

        0 \geq \text{obj}_{\text{slave}} + \sum_{l,i,j} \mu_{l~i~j} \cdot (d_{l~i~j} - \hat{d}_{l~i~j})

   - **If the slave is feasible**, the optimality cut ensures that the master’s cost estimate :math:`\theta` is no better than the slave’s solution:

     .. math::

        \theta \geq \text{obj}_{\text{slave}}  + \sum_{l,i,j} \mu_{l~i~j} \cdot (d_{l~i~j} - \hat{d}_{l~i~j})

   - These cuts are added to the appropriate constraint lists in the master model (`infeasibility_cut` or `optimality_cut`).

This dynamic Benders cut formulation guides the master model toward more promising or feasible network topologies in subsequent iterations.


Slack Verification
------------------

.. method:: check_slave_feasibility()

- Reads slack variables for current (i), over-voltage (+), and under-voltage (-).
- Declares the slave model **infeasible** if any slack exceeds the defined threshold.

Switch and Solution Extraction
------------------------------

.. method:: extract_switch_status()

Returns switch statuses (open/closed) based on delta values.

.. method:: extract_node_voltage()

Extracts node voltages in per-unit from the slave model.

.. method:: extract_edge_current()

Extracts branch currents (per-unit) from the slave solution.

Utility Methods
---------------

.. method:: find_initial_state_of_switches()

Attempts to initialize the system topology before Benders iterations:

1. Builds a graph of all normally closed switches.
2. Checks if the resulting graph is a **spanning tree**:
   - If so, constructs a **BFS tree** rooted at the slack node.
   - Extracts active branches as initial values for :math:`d_{l~i~j}`.
3. If the graph is **not a tree**, solves the master problem once to determine an initial feasible topology.
4. Applies the resulting topology to the slave model instance using the :code:`master_d` parameter.

.. method:: __adapt_penalty_cost()

Scales the penalty cost automatically based on the order of magnitude of line resistance values.


