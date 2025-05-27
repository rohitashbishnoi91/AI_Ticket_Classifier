# utils.py

import joblib
import json
import os
import pandas as pd

def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure directory exists
    joblib.dump(model, filepath)                            # Save model to file

def load_model(filepath):
    return joblib.load(filepath)                            # Load model from file

def save_label_map(series, filepath):
    label_map = {i: label for i, label in enumerate(sorted(series.unique()))}  # Create index-label map
    with open(filepath, "w") as f:
        json.dump(label_map, f)                             # Save map as JSON

def load_label_map(filepath):
    with open(filepath, "r") as f:
        return json.load(f)                                 # Load label map from JSON

def get_unique_products(file_path="cleaned_tickets.csv"):
    df = pd.read_csv(file_path)                             # Read CSV into DataFrame
    return df['product'].unique().tolist()                  # Return unique product list
