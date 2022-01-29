from sklearn.preprocessing import Normalizer
import time

class Norm:
    
    def __init__(self):
        self.runtime = None
        
    def run(self, X):
        timer_start = time.time()
        X = Normalizer().fit_transform(X)
        self.runtime = (time.time() - timer_start) * 1000
        return X