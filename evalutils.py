import openml
import pandas as pd
import numpy as np
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
from sklearn.pipeline import Pipeline

from tqdm import tqdm
import matplotlib.pyplot as plt

def get_dataset(openmlid):
    ds = openml.datasets.get_dataset(openmlid)
    df = ds.get_data()[0]
    num_rows = len(df)
        
    # prepare label column as numpy array
    print(f"Read in data frame. Size is {len(df)} x {len(df.columns)}.")
    X = df.drop(columns=[ds.default_target_attribute]).values
    y = df[ds.default_target_attribute].values
    print(f"Data is of shape {X.shape}.")
    return X, y


