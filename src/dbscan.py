import numpy as np
import logging
import time 
import os
import pandas as pd

def get_neighbors(X, current_index, epsilon, similarity):
    '''
    X - dataset with point coordinates,
    current_index - index of given point,
    epsilon - max similarity or distance to given point,
    similarity - metric of similarity or distance bewteen two points
    '''
    neighbor_indices = []
    for neighbor_index, neighbor in enumerate(X):
        if similarity(neighbor, X[current_index]) <= epsilon:
            neighbor_indices.append(neighbor_index)
    return neighbor_indices

def dbscan(X, epsilon, minPts, similarity):
    '''
    X - dataset with point coordinates,
    epsilon - max similarity or distance to given point,
    minPts - minimum number of points that create cluster,
    similarity - metric of similarity or distance bewteen two points
    '''
    logging.info(f'start log,,,')
    
    # each data point can be in one of 3 stages
    NOT_VISITED = -1 # not visited point
    VISTED = 0 # non-core point
    CLUSTERED = 1 # core point
    
    # initial setup
    n = X.shape[0]
    cluster = np.array([-1] * n) # cluster register
    state = np.array([NOT_VISITED] * n) # state register
    cluster_id = 1

    def search(current_index, cluster_id, epsilon, minPts, similarity):
        ''' Extend cluster '''
        
        # calculation of Eps-neighborhood for current_index
        timer1 = time.time() 
        neighbor_indices = get_neighbors(X, current_index, epsilon, similarity)
        logging.info(f'Eps_time,{current_index},{(time.time() - timer1)*1000},')
        logging.info(f'|Eps_neighbors|,{current_index}, {len(neighbor_indices)},')
        neighbor_indices_str =';'.join(str(e) for e in neighbor_indices)
        logging.info(f'Eps_neighbor_id,{current_index},,{neighbor_indices_str}')
        
        if len(neighbor_indices) >= minPts:
            state[current_index] = CLUSTERED
            cluster[current_index] = cluster_id
            for neighbor_index in neighbor_indices:
                if state[neighbor_index] == NOT_VISITED:
                    state[neighbor_index] = CLUSTERED
                    cluster[neighbor_index] = cluster_id
                    
                    # number of distance/similarity calculations
                    logging.info(f'similarity_calculation, {current_index}, 1,')
                    search(neighbor_index, cluster_id, epsilon, minPts, similarity)
        else:
            state[current_index] = VISTED

    while NOT_VISITED in state:
        not_visited_ids = np.where(state==NOT_VISITED)[0]
        search(not_visited_ids[0], cluster_id, epsilon, minPts, similarity)
        cluster_id += 1
    logging.info(f'stop log,,,')
    return cluster, state


class DBSCAN:

    def __init__(self, epsilon, minPts, similarity):
        self.epsilon = epsilon
        self.minPts = minPts
        self.distance = similarity
        self.log_output = 'out.log'
        self.name = 'dbscan'
    
    def fit_transform(self, X):

        # Logger setup
        logging.basicConfig(
            level=logging.INFO, 
            filename=self.log_output, 
            filemode='w+',
            format='%(msecs)06f,%(message)s',
            datefmt='%H:%M:%S'
        )
        
        self.X = X
        result = dbscan(self.X, self.epsilon, self.minPts, self.distance)
        self.y_pred, self.state = result
        logging.shutdown()
        return self.y_pred
    
    def get_logs(self):
        logs = pd.read_csv(
            self.log_output,
            names=['time [ms]', 'operation', 'point_id', 'value', 'string']
        )
        logs['time [ms]'] -= logs['time [ms]'].min()
        return logs