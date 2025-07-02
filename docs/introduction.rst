Introduction
=================================


Modern power distribution networks are facing increasing uncertainty due to the rapid growth of distributed energy resources (DERs) such as solar panels, electric vehicles, and heat pumps. These new elements introduce both variability and unpredictability in consumption and production, which makes long-term planning and grid operation significantly more complex.

One of the main challenges for distribution system operators is to determine how to configure the network topology—deciding which lines and switches should be active—so that the grid remains *radial*, losses are minimized, and operational constraints such as voltage limits and current ratings are satisfied. This becomes especially difficult when the future growth of DERs is not fully predictable, which creates what is known as *deep uncertainty*.

Traditional approaches often attempt to solve the full optimization problem in a single step, combining binary decisions (e.g., switch statuses) with nonlinear equations that represent power flow. However, this results in a very complex problem that becomes hard to solve for large networks. To reduce complexity, many methods use simplified models or heuristic algorithms, but this often comes at the cost of accuracy, feasibility, or scalability.

To overcome these limitations, we present a **two-stage optimization framework** based on Benders decomposition:

* In the **first stage** (the master problem), we determine the network topology by selecting which lines and switches should be active. This problem is solved using integer variables and includes constraints to ensure radiality and respect orientation rules.

* In the **second stage** (the slave problem), we evaluate the chosen topology using a realistic power flow model based on Second-Order Cone Programming (SOCP). This model captures important physical effects such as voltage drops, power losses, and current constraints more accurately than linear approximations.

A key innovation in our framework is the use of **slack variables** in the slave problem. These allow the model to handle small violations in voltage or current limits in a controlled way, enabling the generation of informative cuts that guide the master problem. As a result, our method can handle both **feasible and infeasible configurations** gracefully and does not rely on penalty tuning or trial-and-error.

The overall system is implemented in **Python**, using Pyomo for modeling, GUROBI for solving both master and slave problems, and Polars + Patito for efficient data processing. Our code is modular, scalable, and designed to support future extensions such as uncertainty modeling and flexible operation from prosumers.

 
