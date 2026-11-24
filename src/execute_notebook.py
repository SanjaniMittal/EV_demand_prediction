"""
Executes the generated Jupyter notebook so that all code cells have saved outputs and rendered plots.
"""

import os
import nbformat
from nbclient import NotebookClient

notebook_path = "notebooks/EV_Demand_Prediction_Analysis.ipynb"

print(f"Reading notebook from {notebook_path}...")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name="python3")
print("Executing notebook cells...")
client.execute()

with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook executed successfully and saved to {notebook_path}.")
