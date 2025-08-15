# %%
import os

os.chdir(os.getcwd().replace("/src", ""))

# %%
from examples import *

# %%
dap = load_dap_state(".cache/boisy_dap")
dap_fixed = load_dap_state(".cache/boisy_dap_fixed")

net = joblib.load(".cache/boisy_net.joblib")

# %%
plot_grid_from_pandapower(net=net, dap=dap, from_z=True, color_by_results=True, text_size=8, node_size=12)  # type: ignore

# %% Plot fixed switches
# plot_grid_from_pandapower(net=net, dap=dap_fixed, from_z=True, color_by_results=True, text_size=8, node_size=12)  # type: ignore
