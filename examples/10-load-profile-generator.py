# %%
import os

os.chdir(os.getcwd().replace("/src", ""))
# %%
from examples import *

download_load_allocation_data(FORCE_DOWNLOAD=False)
