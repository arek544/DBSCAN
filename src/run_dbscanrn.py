import time 
from src.dbscan import *
from src.utils import *
from src.metrics import *
from src.clusterization_performance import *
from src.datasets import Dataset
from src.dbscanrn_optimized import *
from src.dbscanrn import *
from src.normalization import *

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json


    
############################## DBSCANRN #############################

config_path = './configs/dbscan.json'
f = open(config_path)
config = json.load(f)

new_config = {}

for dataset_name in config.keys():
    conf = config[dataset_name]
    if not conf['disable']:
        dataset = Dataset(conf['path'])
        X, y = dataset.X, dataset.y
        normalization = Norm()
        X = normalization.run(X)
        #################### Clusterization ###########################
        timer_start = time.time()
        params = {
            'similarity': cosine_dissimilarity
        }
        params.update(conf['params'])

        algorithm = DBSCANRN(**params)

        name = get_name(
            algorithm_name=algorithm.name, 
            dataset_name=conf['name'], 
            n_dimentions=dataset.n_dimentions, 
            n_rows=dataset.n_rows,
            **params
        )
        print(name)
        algorithm.log_output = f'out/LOG_{name}.log'
        algorithm.fit_transform(X)
        
        display_points(algorithm.X, algorithm.y_pred, numerate=False)
        print("\n")
        
        pd.DataFrame({
            'x': algorithm.X[:, 0],
            'y': algorithm.X[:, 1]
        }).to_csv(f"out/{name}_{dataset_name}.csv")
        
        name = get_name(
            algorithm_name="algorithm", 
            dataset_name=conf['name'], 
            n_dimentions=dataset.n_dimentions, 
            n_rows=dataset.n_rows,
            **params
        )
        conf['log_out'] = f'out/LOG_{name}.log'
        conf['out_path'] = f'out/algorithm_{dataset_name}.csv'
                    
        
    new_config[dataset_name] = conf
    
with open(config_path, 'w') as outfile:
    json.dump(new_config, outfile, indent=4, sort_keys=True)
    
   