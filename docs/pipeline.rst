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
   - Otherwise, the master problem is solved once to find a valid topology.

2. **Solve the slave model** using the current fixed topology decision :math:`d_{l~i~j}`.
3. **Check feasibility** using slack variable violations:

   - Overcurrent: :math:`i_{l~i~j}^{\text{slack}}`
   - Overvoltage: :math:`V_n^{+}`
   - Undervoltage: :math:`V_n^{-}`

4. **Generate a Benders cut** using dual values from slave constraints.
5. **Solve the master model** with the updated cuts.
6. **Repeat** until convergence or max iteration count.

Progress and convergence are tracked using `tqdm`, based on the gap between master and slave objectives.

.. math::

    \theta \geq \sum_{l,i,j} \mu_{l~i~j} \cdot (d_{l~i~j} - \hat{d}_{l~i~j})

Where:

- :math:`\mu_{l~i~j}` are dual values (marginal costs) from the slave problem.
- :math:`\hat{d}_{l~i~j}` is the active configuration from the previous iteration.




Benders Cut Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. method:: add_benders_cut()

Generates a Benders cut based on dual information from the slave problem. The procedure dynamically constructs either an **optimality cut** or an **infeasibility cut**, depending on the outcome of the slave solution.

Steps performed:

1. **Select dual multipliers**:

   - Dual variables (shadow prices) are extracted from the slave model using `self.slave_model_instance.dual`.

   - Each dual value is mapped to a specific constraint name and indexed by branch tuple :math:`(l,i,j)`.


2. **Scale duals by constraint type**:

   - A dictionary of weights is defined depending on the feasibility of the slave solution:

     - If the slave is **infeasible**, constraints like `node_active_power_balance` or `voltage_drop_lower` are weighted by `infeasibility_factor`.
     - If the slave is **feasible**, the same constraints are scaled by `optimality_factor`.


3. **Filter and reshape duals**:

   - The constraint names are parsed and cleaned to extract the associated line and node indices.
   - Duals are grouped and summed by candidate connection :math:`(l,i,j)`.


4. **Link to master decisions**:

   - Extracts current values of the binary variable :math:`d_{l~i~j}` (denoted `d`) from the slave instance.
   - Retrieves symbolic Pyomo variables (`d_variable`) from the master instance.


5. **Construct the Benders cut**:

   - Initializes the cut expression with the current slave objective value :math:`\theta`.

   - For each candidate line, adds a linear term:

     
     .. math::

        \hat{\mu}_{l~i~j} \cdot (d_{l~i~j} - \hat{d}_{l~i~j})

     where:

     - :math:`\hat{\mu}_{l~i~j}` is the aggregated dual multiplier from the slave.
     - :math:`\hat{d}_{l~i~j}` is the evaluated value of the binary decision variable.


6. **Add cut to master model**:

   - If the slave is **infeasible**, the resulting cut ensures:

     .. math::

        0 \geq \theta + \sum_{l,i,j} \hat{\mu}_{l~i~j} \cdot (d_{l~i~j} - \hat{d}_{l~i~j})

   - If the slave is **feasible**, it imposes the optimality cut:

     .. math::

        \theta \geq \theta + \sum_{l,i,j} \hat{\mu}_{l~i~j} \cdot (d_{l~i~j} - \hat{d}_{l~i~j})

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


