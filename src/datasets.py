import numpy as np
import pandas as pd

def example_from_lecture():
    X = np.array([
        [4.2, 4.0], 
        [5.9, 3.9],
        [2.8, 3.5],
        [12.0, 1.3],
        [10.0, 1.3],
        [1.1, 3.0],
        [0.0, 2.4],
        [2.4, 2.0],
        [11.5, 1.8],
        [11.0, 1.0],
        [0.9, 0.0],
        [1.0, 1.5]
    ], np.single)
    y =  np.array([1, 0, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1])
    return X, y

def jain_dataset():
    header = ["x", "y", "real_cluster"]
    df = pd.read_csv('./data/jain_dataset.txt', sep = '\t', names=header)
    X = df[["x", "y"]].values
    y = df["real_cluster"].values
    return X, y

def compound_dataset():
    header = ["x", "y", "real_cluster"]
    df = pd.read_csv('./data/compound_dataset.txt', sep = '\t', names=header)
    X = df[["x", "y"]].values
    y = df["real_cluster"].values
    return X, y