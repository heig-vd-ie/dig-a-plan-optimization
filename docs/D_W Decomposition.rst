Dantzig–Wolfe Decomposition 
=========================================================================================

This module implements a two-stage optimization framework for distribution grid planning using **Dantzig–Wolfe decomposition** (column generation).
The approach decomposes the problem into a **master problem** (selecting an optimal convex combination of network configurations) and
a **slave problem** (generating new feasible configurations by solving detailed power flow with fixed topology).
The framework uses `Pyomo` for optimization modeling and `Polars` for structured data handling.

Model Structure
------------------

Model Generation Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~


.. function:: D_W_model_sets(), D_W_model_parameters(), D_W_model_variables(), D_W_model_constraints()

   Assemble the Dantzig–Wolfe master problem as a Pyomo AbstractModel. The master optimizes the selection of network configuration columns (via :math:`\lambda_k` variables).

.. function:: slave_model_sets(), slave_model_parameters(), slave_model_variables(), slave_model_constraints()

   Assemble the DistFlow slave problem as a Pyomo AbstractModel. The slave (subproblem) solves for the detailed operational cost of a given fixed configuration.

Main Class: DigAPlan  
--------------------

.. class:: DigAPlan
    :noindex:

This class manages the entire master-slave column generation framework.

Constructor  
~~~~~~~~~~~

.. code-block:: python

    def __init__(self, solver_name="gurobi", big_m=1e4, v_penalty_cost=1e-3, i_penalty_cost=1e-3, slack_threshold=1e-5, verbose=False):

Sets up hyperparameters:

- `solver_name`: Optimization solver (default: Gurobi)
- `big_m`: Big-M constant for constraints
- `v_penalty_cost`, `i_penalty_cost`: Penalties for voltage/current violations (in the slave)
- `slack_threshold`: Numerical threshold for constraint violation detection
- `verbose`: Enables detailed output

Data Handling  
~~~~~~~~~~~~~

- `add_grid_data(**grid_data)`: Validates and injects node/edge data, sets slack node, instantiates Pyomo models.
- Internal properties: node/edge dataframes, instantiated Pyomo master/slave models, solver factories.

Model Instantiation  
-------------------

.. method:: _instantiate_model()

- Collects and formats all network data into a dictionary required by Pyomo.
- Builds all sets, parameters, and operational limits for both master and slave models.
- Instantiates Pyomo concrete models (`self._master_inst`, `self._slave_inst`) with the prepared data.

Column Generation Loop  
----------------------

.. method:: solve_column_generation(max_iters=20)

Performs the iterative Dantzig–Wolfe column generation procedure:

1. Generate an initial feasible tree (pattern).
2. Solve the **slave problem** for the current pattern.
3. Add the result as a new column to the master problem.
   which means storing both the pattern (d0) and its associated operational cost (f0) as a new column.
   In the code, d0 is a dictionary of the form {:math:`\{(l~i~j): 0 \text{ or } 1\}`} representing the network configuration, and f0 is the cost for this pattern. 
   Each column thus contains a feasible network pattern (d0) and its cost (f0), and is made available for selection in the master problem.
4. Update the master and resolve, extract duals.
5. Use duals to generate a new candidate column (pricing problem).
6. Stop if no improving columns remain, else repeat.

Key Methods  
~~~~~~~~~~~

- .. method:: _initial_tree_pattern()
  
     Constructs an initial spanning tree over the candidate network using NetworkX, setting up the first feasible pattern (d0,f0).

- .. method:: _solve_slave(pattern)
  
     Solves the slave (DistFlow) problem for a given network configuration (pattern). Returns the objective value for that configuration.

- .. method:: _price_and_solve(:math:`\pi`, :math:`\sigma`)
  
     Uses dual variables (:math:`\pi`, :math:`\sigma`) from the master problem to define weights for a new minimum spanning tree (pricing problem), generating a new candidate column and its cost.

Mathematical Model
==================

Initial Feasible Pattern
------------------------

Before the column generation loop can start, the master problem must have at least one feasible column (network pattern). 
The `initial_tree_pattern` method generates this starting solution by finding a feasible spanning tree over the candidate network.

- This tree corresponds to a valid configuration of active lines (:math:`d_0`), ensuring the network is connected and radial (a fundamental requirement for distribution grids).
- In code, this is implemented using NetworkX’s ``minimum_spanning_tree`` function, creating a pattern (:math:`d_0`, dictionary of :math:`\{(l~i~j): 0 \text{ or } 1\}`) indicating which lines are active.
- The resulting pattern is evaluated in the slave to obtain its operational cost and is added as the first column to the master.

Master model
--------------

Slave model (Subproblem)
------------------------

The **slave problem** is solved for a fixed network pattern and computes the real operational cost (objective), power flows, voltages, and current violations.

Fixed Pattern Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   d_{l~i~j} = \text{given} \in \{0,1\} \qquad \forall (l~i~j)

*The network topology is fixed by the current column pattern when solving the slave.*

Column Generation / Pricing Problem
-----------------------------------

At each iteration, a new column is generated by solving a **minimum reduced cost spanning tree** problem, with edge weights defined by the dual variables :math:`\pi` extracted from the master:

.. math::

   w_{l~i~j} = -\frac{1}{2} \left( \pi_{l~i~j} + \pi_{l~j~i} \right)

Where :math:`\pi_{l~i~j}` is the dual variable for the coupling constraint associated with edge :math:`(l~i~j)`.

The minimum spanning tree solution defines the new candidate pattern, which is then evaluated in the slave problem for its true cost.

Reduced cost is computed as:

.. math::

   rc = f^* - \sum_{(l~i~j)} \pi_{l~i~j} \cdot d^*_{l~i~j} - \sigma

Where:

* :math:`f^*`: Cost of new column from the slave
* :math:`\pi_{l~i~j}`: Dual variables from master
* :math:`d^*_{l~i~j}`: New pattern
* :math:`\sigma`: Dual variable for convexity constraint

Algorithm:
-----------------------------
.. _grid-schema:
.. figure:: /images/floachart.pdf
    :width: 300



