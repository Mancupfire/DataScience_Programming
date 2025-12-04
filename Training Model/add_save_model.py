import json
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
notebook_path = os.path.join(script_dir, 'xgboost.ipynb')

print(f"Reading from: {notebook_path}")

# Read the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Create the new cell for saving the model
save_model_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Save the trained model\n",
        "model_save_path = \"xgboost_model.json\"\n",
        "model.save_model(model_save_path)\n",
        "print(f\"Model saved to {model_save_path}\")"
    ]
}

# Append the new cell to the notebook
nb['cells'].append(save_model_cell)

# Save the modified notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print("Added model saving code to xgboost.ipynb")
