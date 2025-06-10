r"""


.. math::
    :label: distflow-initialization
    :nowrap:
    
    \begin{align} 
        v_{\text{slack}}^{2} = V_{\text{ref}}^{2}
    \end{align}
"""

import pyomo.environ as pyo

def complete_model_variables(model: pyo.AbstractModel) -> pyo.AbstractModel:
    
    # Candidate-indexed branch variables.
    model.p_flow = pyo.Var(model.LC, domain=pyo.Reals)
    model.q_flow = pyo.Var(model.LC, domain=pyo.Reals)
    model.i_sq = pyo.Var(model.LC, domain=pyo.NonNegativeReals)
    model.v_sq = pyo.Var(model.N, domain=pyo.NonNegativeReals)
    # Binary variables for switch status selection and radiality.
    model.d = pyo.Var(model.LC, domain=pyo.Binary)
    model.delta = pyo.Var(model.S, domain=pyo.Binary)
    # Slack variable for voltage and current limitation constraints.
    model.slack_v_pos = pyo.Var(model.N, domain=pyo.NonNegativeReals)
    model.slack_v_neg = pyo.Var(model.N, domain=pyo.NonNegativeReals)
    model.slack_i_sq = pyo.Var(model.LC, domain=pyo.NonNegativeReals)
    return model
    