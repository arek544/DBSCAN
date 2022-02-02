from turtle import down
import pandas as pd
import numpy as np
from src.metrics import cosine_dissimilarity
import logging
import time 
from IPython.display import display

def pessimistic_estimation(df, dfx, current_index, real_max, down_row, idx, k, previous_check):

    dfy = df.iloc[down_row]
    dfy['check'] = dfy['pessimistic_estimation'] < real_max
    previous_check = bool(dfy['check'])
    if not previous_check:
        return dfx
    if previous_check:
        dfy['cosine_dissimilarity'] = cosine_dissimilarity(dfy[['x','y']].values, df[df['index']==current_index][['x','y']].values[0])
        if dfy['cosine_dissimilarity'] < real_max:
            dfx = dfx[dfx['cosine_dissimilarity'] != real_max]
            dfx = dfx.append(dfy[['index', 'x','y', 'cosine_dissimilarity', 'r_distnace']])
            real_max = dfx["cosine_dissimilarity"].max()   
            return pessimistic_estimation(df, dfx, current_index, real_max, down_row, idx, k, previous_check)
    return pessimistic_estimation(df, dfx, current_index, real_max, down_row, idx, k, previous_check)


def ti_knn(k, df, current_index, all_point_indices):
   
    # calculate distance to reference point, [0,1]
    df['r_distnace'] = df.apply(
        lambda row: cosine_dissimilarity(row[['x','y']], [0,1]), 
        axis=1
    ) 
    
    timer_start = time.time()

    logging.info(f'dist_to_ref_point_time,,{(time.time() - timer_start) * 1000},')
    
    timer_start = time.time()
    logging.info(f'sorting_dist_time,{current_index},{(time.time() - timer_start) * 1000},')
    current_index_r_distance = df[df['index']==current_index]['r_distnace'].values[0]
    df['pessimistic_estimation'] = abs(current_index_r_distance-df['r_distnace'])
    df = df.sort_values(by='pessimistic_estimation')
    df.reset_index(inplace=True, drop=True)
    dfx = df.head(k+1)
    dfx = dfx[dfx['index']!=current_index]

    # calculate distance to current point
    idx = df[df['index']==current_index].index.values[0]
    down_row = dfx.iloc[[k-1]].index.values[0] + 1

    dfx['cosine_dissimilarity'] = dfx.apply(
        lambda row: cosine_dissimilarity(row[['x','y']], df[df['index']==current_index][['x','y']].values[0]
        ), axis=1) 
    real_max = dfx["cosine_dissimilarity"].max()
    last = True
    dfx = pessimistic_estimation(df, dfx, current_index, real_max, down_row, idx, k, last)
    
    return dfx

def get_tiknn(k, df, all_point_indices):
    point_tiknn = {}
    for current_index in range(0, df.shape[0]):
        result = ti_knn(k, df, current_index, all_point_indices)
        point_tiknn_result = result['index'].to_list()
        point_tiknn_result = [int(point) for point in point_tiknn_result]
        point_tiknn[current_index] = point_tiknn_result
        tiknn_indices_str = ';'.join(str(e) for e in point_tiknn_result)
        logging.info(f'tiknn_neighbors_id,{current_index},,{tiknn_indices_str}')
        logging.info(f'|tiknn_neighbors|,{current_index}, {len(point_tiknn_result)},')
        
    return point_tiknn


def get_tirnn(k, df, all_point_indices):
    point_tirnn = {}
    
    timer_start = time.time()
    point_tiknn = get_tiknn(k, df, all_point_indices)
    logging.info(f'tiknn_time,,{(time.time() - timer_start) * 1000},')
    
    for current_index in all_point_indices:
        tirnn = get_pointwise_rnn(point_tiknn, current_index)
        point_tirnn[current_index] = tirnn
        tirnn_indices_str = ';'.join(str(e) for e in tirnn)
        logging.info(f'tirnn_neighbors_id,{current_index},,{tirnn_indices_str}')
        logging.info(f'|tirnn_neighbors|,{current_index}, {len(tirnn)},')
        
    return point_tirnn, point_tiknn


def get_pointwise_rnn(point_knn, current_index):
    rnn = []
    for neighbor in point_knn[current_index]:
        if current_index in point_knn[neighbor]:
            rnn.append(neighbor)
    return rnn


def get_knn(current_index, neighbor_indices, k, similarity, X):
    '''
    current_index - index of given point,
    neighbor_indices - indices of points for searching,
    k - number of neighbors searched for current_index,
    similarity - metric of similarity or distance bewteen two points,
    X - dataset with point coordinates
    '''
    # Remove current point from neighbors
    if current_index in neighbor_indices: neighbor_indices.remove(current_index)
    
    # For each neighbor measure similarity to current point
    neighbor_similaritys = []
    for neighbor_index in neighbor_indices:
        similarity_measure = similarity(X[neighbor_index], X[current_index])
        neighbor_similaritys.append(similarity_measure) 
        
    # Get k-nearest neighbors
    sort_indices = np.argsort(neighbor_similaritys) # sort arguments and return indices
    neighbor_indices = np.array(neighbor_indices) 
    return neighbor_indices[sort_indices][:k].tolist()


def ti_dbscanrn(X, k, similarity):
    
    logging.info(f'start log,,,')
    # inidces of all points
    all_point_indices = list(range(len(X))) 
    
    df = pd.DataFrame({
    "index": all_point_indices,
    "x": X[all_point_indices][:,0],
    "y": X[all_point_indices][:,1],
    'cosine_dissimilarity': np.nan
    })
    
    # each data point can be in one of 3 stages
    NOT_VISITED = -1 # not visited point
    VISTED = 0 # non-core point
    CLUSTERED = 1 # core point
    
    # initial setup
    n = X.shape[0]
    cluster = np.array([-1] * n) # cluster register
    state = np.array([NOT_VISITED] * n) # state register
    cluster_id = 1
    
    timer_start = time.time()
    point_rnn, point_knn = get_tirnn(k, df, all_point_indices) # calculate RNN_k for all points
    logging.info(f'tirnn_time,,{(time.time() - timer_start) * 1000},')
    
    # search for clusters
    def search(current_index, k):
        if len(point_rnn[current_index]) < k:
                state[current_index] = VISTED
        else:
            state[current_index] = CLUSTERED
            cluster[current_index] = cluster_id
            for neighbor_index in point_rnn[current_index]:
                if state[neighbor_index] == NOT_VISITED:
                    search(neighbor_index, k)
                state[neighbor_index] = CLUSTERED
                cluster[neighbor_index] = cluster_id
                    
    # visit all X points
    while NOT_VISITED in state:
        not_visited_ids = np.where(state==NOT_VISITED)[0][0]
        search(not_visited_ids, k)
        cluster_id += 1
    
    # clusterize all outlier data points 
    while VISTED in state:
        not_clustered_ids = np.where(state==VISTED)[0][0]
        clustered_ids = np.where(state==CLUSTERED)[0]
        knn = get_knn(not_clustered_ids, clustered_ids, 1, similarity, X)
        cluster[not_clustered_ids] = cluster[knn[0]]
        state[not_clustered_ids] = CLUSTERED
    logging.info(f'stop log,,,')
    return cluster, state


class DBSCANRN_opt:

    def __init__(self, k, similarity):
        self.k = k
        self.similarity = similarity
        self.log_output = 'out.log'
        self.name = 'dbscanrn_opt'
    
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
        result = ti_dbscanrn(self.X, self.k, self.similarity)
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