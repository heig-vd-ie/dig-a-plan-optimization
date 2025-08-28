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
collection = db.run_20250828_065031  # replace with your collection

# Fetch all documents with objectives and source file
cursor = collection.find({}, {"objectives": 1, "_source_file": 1, "_id": 0})

# Flatten into a DataFrame
data = []
for doc in cursor:
    file_name = doc.get("_source_file", "unknown")
    for obj in doc.get("objectives", []):
        data.append(
            {"objective": obj, "iteration": int(file_name.split("_")[-1].split(".")[0])}
        )

df = pd.DataFrame(data)

# Plot distribution using Plotly
fig = px.histogram(
    df,
    x="objective",
    color="iteration",
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
    legend=dict(title="Iteration number", orientation="v", x=1.02, y=1),
    margin=dict(r=150),  # Add right margin for legend
)

fig.show()

# %%
