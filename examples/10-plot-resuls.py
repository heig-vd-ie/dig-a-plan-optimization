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
    "Expectation": db.run_20250901_114205,
    "WorstCase": db.run_20250901_115710,
    "Wasserstein": db.run_20250901_115715,
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

# df_all = df_all[(df_all["source"] != "Expectation") | (df_all["iteration"])]

# Plot distribution using Plotly
fig = px.histogram(
    dfs["Expectation"],
    x="objective",
    color="source",
    nbins=50,
    title="Distribution of Objectives by Iteration",
    labels={
        "objective": "Objective value",
        "count": "Frequency",
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

# %%
