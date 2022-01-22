import pandas as pd
import numpy as np
from src.metrics import euclidean_distance

def check_pessimistic_estimation(df, df2, current_point):
    # calculate real distance to current point
    df2['real'] = df2.apply(lambda row: euclidean_distance(row[['x','y']], current_point[['x','y']].values[0]), axis=1)
    df.update(df2)
    # choose max real distance
    max_val = df2.max()
    # get points with fake distance below max real distance 
    return df[(df['pesimistic_distance'] < max_val['real']) & (df['pesimistic_distance'] > 1) & (df['real'].isna())]

def ti_knn(k, df, current_index, all_point_indices):
   
    # calculate distance to reference point, [0,1]
    df['r_distnace'] = df.apply(lambda row: euclidean_distance(row[['x','y']], [0,1]), axis=1)
    df = df.sort_values(by='r_distnace')

    # calculate distance to current point
    current_point = df[df['index']==current_index]
    df['pesimistic_distance'] = abs(df['r_distnace'] - current_point['r_distnace'].values[0] )

    # get k-NN acording to pessimistic estimation 
    df2 = df[df['pesimistic_distance'] > 0].sort_values(by='pesimistic_distance').head(k)
    # was > 1, changed to > 0

    dfn = check_pessimistic_estimation(df, df2, current_point)

    # check if current df is empty
    def empty_df(df, df2, dfn):
        for n in range(0, df.shape[0]):
            if dfn.empty:
                result = df.sort_values(by='real').head(k) 
                return result
            else:
                dfn = check_pessimistic_estimation(df, dfn, current_point)
                empty_df(df, df2, dfn)

    result = empty_df(df, df2, dfn)
    df['real']=np.nan
    
    return result

def get_tiknn(k, df, all_point_indices):
    point_knn = {}
    for current_index in range(0, df.shape[0]):
        result = ti_knn(k, df, current_index, all_point_indices)
        point_knn_result = result['index'].to_list()
        point_knn_result = [int(point) for point in point_knn_result]
        point_knn[current_index] = point_knn_result
    return point_knn


def get_tirnn(k, df, all_point_indices):
    point_tirnn = {}
    point_tiknn = get_tiknn(k, df, all_point_indices)
    for current_index in all_point_indices:
        rnn = get_pointwise_rnn(point_tiknn, current_index)
        point_tirnn[current_index] = rnn
        
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
    'real': np.nan
    })
    
    # each X point can be in one of 3 stages
    NOT_VISITED = 0 # not visited point
    VISTED = 1 # non-core point
    CLUSTERED = 2 # core point
    
    # initial setup
    n = X.shape[0]
    cluster = np.array([0] * n) # cluster register
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
    
    return cluster