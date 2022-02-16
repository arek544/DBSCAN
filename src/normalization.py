import time
import numpy as np

class Norm:
    
    def __init__(self):
        self.runtime = None
        
    def run(self, X):
        timer_start = time.time()
        X = X/np.linalg.norm(X, axis =1, keepdims = True)
        self.runtime = (time.time() - timer_start) * 1000
        return X