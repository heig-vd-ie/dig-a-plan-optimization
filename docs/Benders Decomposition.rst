Benders Decomposition 
=========================================================================================


This module implements a two-stage optimization framework using Benders decomposition for solving distribution grid planning problems. The first stage (master problem) determines the configuration of the grid (e.g., switch statuses), and the second stage (slave problem) checks feasibility of DistFlow optimization and computes penalties (voltage/current violations) for the given configuration. The framework uses `Pyomo` for optimization modeling and `Polars` + `Patito` for structured data handling.


Model Structure
---------------

Model Generation Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: generate_master_model()

   Returns a Pyomo AbstractModel.

   Constructs the master problem to select tree base structure of :math:`d_{(l~i~j)}` 

.. function:: generate_slave_model()

   Returns a Pyomo AbstractModel.

   Constructs the slave problem and enables dual suffix extraction for Benders cuts.

Main Class: DigAPlan
--------------------

.. class:: DigAPlan

This is the central class that manages the master-slave decomposition. It stores data, manages solver configuration, and handles optimization iteration.

Constructor
~~~~~~~~~~~

.. code-block:: python

    def __init__(self, verbose=False, big_m=1e4, v_penalty_cost=1e-3, i_penalty_cost=1e-3, slack_threshold=1e-5):

Sets up hyperparameters such as:

- `big_m`: Big-M value for constraints

- `v_penalty_cost`, `i_penalty_cost`: Penalty costs for voltage and current violations

- `slack_threshold`: Threshold below which slacks are considered negligible

Properties
~~~~~~~~~~

Read-only properties are defined for:

- `node_data`, `edge_data`: Network data

- `master_model`, `slave_model`: Pyomo abstract models

- `master_model_instance`, `slave_model_instance`: Concrete models (instantiated with data)

- `slack_node`: Unique slack node of the system

Data Injection
~~~~~~~~~~~~~~

.. method:: add_grid_data(**grid_data)

Validates and injects node and edge data. Ensures exactly one slack node is defined and triggers model instantiation.

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

- Voltage limits: :math:`V_{min}, V_{max}`

- Current limits: :math:`I_{max}`

- Decision variables: :math:`d_{ij}` (discrete switch decision)

Master-Slave Iteration
----------------------

.. method:: solve_models_pipeline(max_iters, bender_cut_factor=1.0)

Performs iterative solving:

1. Solve master problem

2. Extract configuration :math:`d`

3. Solve slave with fixed :math:`d`

4. Check feasibility (slacks)

5. If infeasible, generate Benders cut and repeat


.. math::

    \theta^{(k)} \geq \sum_{(l~i~j)} \mu_{lij}^{(k)} \cdot \text{factor}_{lij}^{(k)} \cdot (d_{lij} - \hat{d}_{lij})

Where:

- :math:`\mu_{lij}` is the dual variable for constraint (l~i~j)

- :math:`\text{factor}_{lij}` adjusts for activation/inactivation

- :math:`\hat{d}_{lij}` is the solution from master at iteration k


Add Benders Cut
~~~~~~~~~~~~~~~

.. method:: add_benders_cut(nb_iter, bender_cut_factor)

Generates a new variable :math:`\theta_k` and constraint to be added to the master problem:

.. math::

    \theta_k \geq \sum \text{dual} \cdot \text{factor} \cdot (d - d^{prev})

Updates objective:

.. math::

    \min(\text{original objective} + \text{bender_cut_factor} \cdot \theta_k)



