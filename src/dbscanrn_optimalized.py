import pandas as pd
import numpy as np
from src.metrics import euclidean_distance
from src.metrics import cosine_dissimilarity 
import logging
import time 
from IPython.display import display

def check_pessimistic_estimation(df, df2, current_point):
    # calculate cosine_dissimilaritye to current point
    # df2['cosine_dissimilarity'] = df2.apply(
        # lambda row: euclidean_distance(row[['x','y']], current_point[['x','y']].values[0]), axis=1
    # ) # czy tu nie powinno być cosine dissimilarity?
    df2['cosine_dissimilarity'] = df2.apply(
        lambda row: cosine_dissimilarity(row[['x','y']], current_point[['x','y']].values[0]
    ), axis=1) # chyba powinno być tak !!! (działa wtedy ok)
    
    df.update(df2)
    # choose max cosine_dissimilarity
    max_val = df2.max()
    # get points with fake distance below max cosine_dissimilarity 
    return df[(df['pesimistic_distance'] < max_val['cosine_dissimilarity']) & (df['pesimistic_distance'] > 0) & (df['cosine_dissimilarity'].isna())]

def ti_knn(k, df, current_index, all_point_indices):
   
    # calculate distance to reference point, [0,1]
    # df['r_distnace'] = df.apply(
        # lambda row: euclidean_distance(row[['x','y']], [0,1]), 
        # axis=1
    # ) # czy tu nie powinno być cosine dissimilarity?
    df['r_distnace'] = df.apply(
        lambda row: cosine_dissimilarity(row[['x','y']], [0,1]), 
        axis=1
    ) # chyba powinno być tak !!! (działa wtedy ok)
    df = df.sort_values(by='r_distnace')

    # calculate distance to current point
    current_point = df[df['index']==current_index]
    df['pesimistic_distance'] = abs(df['r_distnace'] - current_point['r_distnace'].values[0])

    # get k-NN acording to pessimistic estimation 
    df2 = df[df['pesimistic_distance'] > 0].sort_values(by='pesimistic_distance').head(k)
    # was > 1, changed to > 0

    dfn = check_pessimistic_estimation(df, df2, current_point)
    
    # check if current df is empty
    def empty_df(df, df2, dfn):
        for n in range(0, df.shape[0]):
            if dfn.empty:
                result = df.sort_values(by='cosine_dissimilarity').head(k) 
                return result
            else:
                dfn = check_pessimistic_estimation(df, dfn, current_point)
                empty_df(df, df2, dfn)

    result = empty_df(df, df2, dfn)
    df['cosine_dissimilarity']=np.nan
    
    return result

def get_tiknn(k, df, all_point_indices):
    point_tiknn = {}
    for current_index in range(0, df.shape[0]):
        result = ti_knn(k, df, current_index, all_point_indices)
        point_tiknn_result = result['index'].to_list()
        point_tiknn_result = [int(point) for point in point_tiknn_result]
        point_tiknn[current_index] = point_tiknn_result
    return point_tiknn


def get_tirnn(k, df, all_point_indices):
    point_tirnn = {}
    point_tiknn = get_tiknn(k, df, all_point_indices)
    for current_index in all_point_indices:
        tirnn = get_pointwise_rnn(point_tiknn, current_index)
        point_tirnn[current_index] = tirnn
        
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
    point_rnn, point_knn = get_tirnn(k, df, all_point_indices) # calculate RNN_k for all points
    
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
    number_of_calc = 0 # to do
    return cluster, state, number_of_calc


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
            format='%(message)s'
        )
        
        self.X = X
        result = ti_dbscanrn(self.X, self.k, self.similarity)
        self.y_pred, self.state, self.number_of_calc = result
        return self.y_pred