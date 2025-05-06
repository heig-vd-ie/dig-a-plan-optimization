r"""


.. math::
    :label: distflow-initialization
    :nowrap:
    
    \begin{align} 
        v_{\text{slack}}^{2} = V_{\text{ref}}^{2}
    \end{align}
"""

import pyomo.environ as pyo

def slave_model_variables(model: pyo.AbstractModel) -> pyo.AbstractModel:
    
    # Candidate-indexed branch variables.
    model.p_z_up = pyo.Var(model.LC, domain=pyo.Reals)
    model.q_z_up = pyo.Var(model.LC, domain=pyo.Reals)
    model.p_z_dn = pyo.Var(model.LC, domain=pyo.Reals)
    model.q_z_dn = pyo.Var(model.LC, domain=pyo.Reals)
    model.i_sq = pyo.Var(model.LC, domain=pyo.NonNegativeReals)
    model.v_sq = pyo.Var(model.N, domain=pyo.NonNegativeReals) # type: ignore
    # Slack variable for voltage drop constraints.
    model.slack_v_sq = pyo.Var(model.N, domain=pyo.NonNegativeReals)
    model.slack_i_sq = pyo.Var(model.LC, domain=pyo.NonNegativeReals)
    # “d” will hold the d‐values coming from the master
    model.d = pyo.Var(model.LC, bounds=(0,1))
    return model
    