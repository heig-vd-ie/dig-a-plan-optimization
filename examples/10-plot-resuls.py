# %%
import os

os.chdir(os.getcwd().replace("/src", ""))
from pymongo import MongoClient
import plotly.express as px
import pandas as pd

SERVER_MONGODB_PORT = os.getenv("SERVER_MONGODB_PORT", 27017)

# Connect to MongoDB
client = MongoClient(f"mongodb://localhost:{SERVER_MONGODB_PORT}")
db = client.optimization
collections = {
    "Expectation": db.run_20250902_110438,
    "WorstCase0.02": db.run_20250902_125013,
    "WorstCase0.05": db.run_20250902_125339,
    "WorstCase0.1": db.run_20250902_125602,
    "WorstCase0.2": db.run_20250902_125740,
    "Wasserstein0.02": db.run_20250902_125917,
    "Wasserstein0.05": db.run_20250902_130117,
    "Wasserstein0.1": db.run_20250902_130318,
    "Wasserstein0.2": db.run_20250902_130518,
}
# Fetch all documents with objectives and source file
cursors = {
    name: collection.find({}, {"objectives": 1, "_source_file": 1, "_id": 0})
    for name, collection in collections.items()
}

# Flatten into a DataFrame
data = {}
dfs = {}
for name, cursor in cursors.items():
    data[name] = []
    for doc in cursor:
        file_name = doc.get("_source_file", "unknown")
        for obj in doc.get("objectives", []):
            data[name].append(
                {
                    "objective": obj,
                    "iteration": int(file_name.split("_")[-1].split(".")[0]),
                }
            )

    dfs[name] = pd.DataFrame(data[name])
    dfs[name]["source"] = name

df_all = pd.concat(dfs.values(), ignore_index=True)

df_all = df_all[

    (df_all["source"] != "Expectation")
    | (df_all["iteration"] == df_all["iteration"].max())
]

# Plot distribution using Plotly
fig = px.histogram(
    df_all,
    x="objective",
    color="source",
    nbins=50,
    histnorm="probability density",  # Normalize to probability density
    title="Distribution of Objectives by Iteration",
    labels={
        "objective": "Objective value",
        "count": "Density",
        "iteration": "Iteration number",
    },
    barmode="group",
)

# Update layout for better appearance
fig.update_layout(
    width=800,
    height=500,
    legend=dict(title="iteration", orientation="v", x=1.02, y=1),
    margin=dict(r=150),  # Add right margin for legend
)

fig.show()
