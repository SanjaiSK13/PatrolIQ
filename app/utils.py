import os
import pandas as pd
import pickle

def get_project_root():
    current = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current)

def data_path(filename):
    return os.path.join(get_project_root(), "data", filename)

def model_path(filename):
    return os.path.join(get_project_root(), "src", "models", filename)

def load_parquet(filename):
    return pd.read_parquet(data_path(filename))

def load_model(filename):
    return pickle.load(open(model_path(filename), "rb"))