import numpy as np

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


def get_pointwise_rnn(point_knn, current_index):
    rnn = []
    for neighbor in point_knn[current_index]:
        if current_index in point_knn[neighbor]:
            rnn.append(neighbor)
    return rnn
    
def get_rnn(point_indices, k, similarity, X):
    '''
    point_indices - indices of points for searching,
    k - number of neighbors searched for current_index,
    similarity - metric of similarity or distance bewteen two points,
    X - dataset with point coordinates
    '''
    point_rnn = {}
    point_knn = {}
    
    for current_index in point_indices:
        knn = get_knn(current_index, point_indices[:], k, similarity, X)
        point_knn[current_index] = knn
        
    for current_index in point_indices:
        rnn = get_pointwise_rnn(point_knn, current_index)
        point_rnn[current_index] = rnn
        
    return point_rnn, point_knn

def dbscanrn(X, k, similarity):
    
    # each data point can be in one of 3 stages
    NOT_VISITED = -1 # not visited point
    VISTED = 0 # non-core point
    CLUSTERED = 1 # core point
    
    # initial setup
    n = X.shape[0]
    cluster = np.array([-1] * n) # cluster register
    state = np.array([NOT_VISITED] * n) # state register
    cluster_id = 1
    all_point_indices = list(range(len(X))) # inidces of all points
    point_rnn, point_knn = get_rnn(all_point_indices, k, similarity, X) # calculate RNN_k for all points
    
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