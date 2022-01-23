import numpy as np
import pandas as pd

from pathlib import Path

class Dataset:
    
    def __init__(self, path):
        self.df = self.load_data(path)
        self.X = self.df[["x", "y"]].values
        self.y = self.df["real_cluster"].values
        self.name = Path(path).stem
        self.n_dimentions = self.X.shape[1]
        self.n_rows = self.X.shape[0]
    
    def load_data(self, path):
        header = ["x", "y", "real_cluster"]
        return pd.read_csv(path, sep='\t', names=header)

